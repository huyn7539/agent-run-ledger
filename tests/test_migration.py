"""L1 — migration mechanism: per-record schema_version, PRAGMA user_version,
major.minor compatibility gate, and the additive _ensure_column pattern on
runs + steps. TDD red-first (Task 44, Phase 1).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent_run_ledger.core.io import load_trace
from agent_run_ledger.core.models import (
    SCHEMA_VERSION,
    RunRecord,
    TraceBundle,
    TraceValidationError,
    is_version_compatible,
)
from agent_run_ledger.core.storage import (
    USER_VERSION,
    connect,
    init_db,
    load_bundle,
    save_bundle,
)


# --- per-record schema_version ------------------------------------------------

def test_run_record_carries_schema_version_default() -> None:
    run = RunRecord(
        id="run_v",
        workflow="w",
        framework="f",
        provider="p",
        model="m",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
        success_label="passed",
    )
    assert run.schema_version == SCHEMA_VERSION


def test_schema_version_persists_and_reads_back(tmp_path: Path) -> None:
    bundle = load_trace(Path("fixtures/clean_run.json"))
    db = tmp_path / "ledger.sqlite"

    run_id = save_bundle(db, bundle)
    loaded = load_bundle(db, run_id)

    # the per-row stamp answers "what shape was this record written under"
    assert loaded.run.schema_version == SCHEMA_VERSION
    with connect(db) as conn:
        col = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    assert "schema_version" in col


# --- PRAGMA user_version on the file ------------------------------------------

def test_user_version_stamped_on_new_db(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    init_db(db)
    with connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == USER_VERSION


def test_user_version_unchanged_across_repeated_init(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    init_db(db)
    with connect(db) as conn:
        first = conn.execute("PRAGMA user_version").fetchone()[0]
    for _ in range(3):
        init_db(db)
    with connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == first


# --- compatibility policy -----------------------------------------------------

@pytest.mark.parametrize(
    ("record", "ok"),
    [
        ("0.1", True),    # same
        ("0.0", True),    # lower minor upcasts
        ("0.9", False),   # higher minor -> unknown fields, reject
        ("0.10", False),  # int compare: 10 > 1, reject (NOT lexical "0.10" < "0.9")
        ("1.0", False),   # higher major, reject
        ("9.9", False),   # higher major, reject
    ],
)
def test_version_compatibility_policy(record: str, ok: bool) -> None:
    assert is_version_compatible(record, "0.1") is ok


def test_higher_minor_rejected_at_bundle_load() -> None:
    with pytest.raises(TraceValidationError):
        TraceBundle.from_dict(
            {"schema_version": "0.9", "run": {"id": "r"}, "steps": [{"id": "s"}]}
        )


def test_lower_minor_accepted_and_upcasts() -> None:
    # a record written under 0.0 (missing newer fields) loads cleanly
    bundle = TraceBundle.from_dict(
        {
            "schema_version": "0.0",
            "run": {
                "id": "run_old",
                "workflow": "w",
                "framework": "f",
                "provider": "p",
                "model": "m",
                "started_at": "2026-05-28T00:00:00Z",
                "ended_at": "2026-05-28T00:00:01Z",
                "success_label": "passed",
            },
            "steps": [
                {
                    "id": "s1",
                    "type": "tool",
                    "name": "s",
                    "started_at": "2026-05-28T00:00:00Z",
                    "ended_at": "2026-05-28T00:00:01Z",
                }
            ],
        }
    )
    assert bundle.run.id == "run_old"


# --- additive _ensure_column pattern on runs + steps --------------------------

def test_old_db_upgrades_additively_without_rewrite(tmp_path: Path) -> None:
    """An older runs/steps table (pre-schema_version) gains the new columns via
    _ensure_column on init, with no row rewrite."""
    db = tmp_path / "ledger.sqlite"
    # Build a pre-L1 shape by hand: runs/steps WITHOUT schema_version column.
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE runs (
                id TEXT PRIMARY KEY, workflow TEXT NOT NULL, framework TEXT NOT NULL,
                provider TEXT NOT NULL, model TEXT NOT NULL, started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL, success_label TEXT NOT NULL,
                prompt_hash TEXT NOT NULL, config_hash TEXT NOT NULL,
                total_cost_usd REAL NOT NULL, total_latency_ms INTEGER NOT NULL,
                total_input_tokens INTEGER NOT NULL, total_output_tokens INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO runs VALUES ('run_old','w','f','openai','m',"
            "'2026-05-28T00:00:00Z','2026-05-28T00:00:01Z','passed','','',0.0,0,0,0)"
        )
        conn.commit()

    init_db(db)  # must ALTER-in schema_version additively, not drop the row

    with connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        row = conn.execute("SELECT id FROM runs WHERE id='run_old'").fetchone()
    assert "schema_version" in cols
    assert row is not None  # row preserved, no rewrite


# --- silent-drop guard: schema_version survives every transform -----------------

def test_schema_version_survives_dict_roundtrip() -> None:
    bundle = load_trace(Path("fixtures/clean_run.json"))
    rt = TraceBundle.from_dict(bundle.to_dict())
    assert rt.run.schema_version == bundle.run.schema_version == SCHEMA_VERSION


def test_schema_version_survives_db_roundtrip(tmp_path: Path) -> None:
    bundle = load_trace(Path("fixtures/clean_run.json"))
    db = tmp_path / "ledger.sqlite"
    save_bundle(db, bundle)
    assert load_bundle(db, bundle.run.id).run.schema_version == SCHEMA_VERSION


# --- L6: prompt_hash / config_hash are hex-digest-or-empty ----------------------

def _run_with(**kw) -> RunRecord:
    base = dict(
        id="run_h", workflow="w", framework="f", provider="p", model="m",
        started_at="2026-05-28T00:00:00Z", ended_at="2026-05-28T00:00:01Z",
        success_label="passed",
    )
    base.update(kw)
    return RunRecord(**base)


def test_prompt_hash_empty_is_valid() -> None:
    assert _run_with(prompt_hash="", config_hash="").prompt_hash == ""


def test_prompt_hash_lowercase_hex_is_valid() -> None:
    digest = "a" * 64
    assert _run_with(prompt_hash=digest, config_hash=digest).prompt_hash == digest


def test_prompt_hash_free_text_rejected() -> None:
    with pytest.raises(TraceValidationError, match="prompt_hash"):
        _run_with(prompt_hash="prompt_retry_loop_v1")


def test_config_hash_free_text_rejected() -> None:
    with pytest.raises(TraceValidationError, match="config_hash"):
        _run_with(config_hash="config_retry_5")


def test_uppercase_hex_rejected_not_canonical() -> None:
    with pytest.raises(TraceValidationError):
        _run_with(prompt_hash="A" * 64)


def test_fixtures_load_with_hex_hashes() -> None:
    # both golden fixtures must now carry real lowercase-hex digests, not labels
    for fx in ("fixtures/golden_retry_loop.json", "fixtures/clean_run.json"):
        bundle = load_trace(Path(fx))
        for h in (bundle.run.prompt_hash, bundle.run.config_hash):
            assert h == "" or (len(h) == 64 and all(c in "0123456789abcdef" for c in h))
