from pathlib import Path
import json
import sqlite3

import pytest

from agent_run_ledger.core.io import load_trace
from agent_run_ledger.core.models import RunRecord, StepRecord, TraceBundle
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.storage import (
    RunAlreadyRecorded,
    _ensure_column,
    connect,
    init_db,
    list_runs,
    load_bundle,
    save_bundle,
)


def test_sqlite_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))

    run_id = save_bundle(db, bundle)
    loaded = load_bundle(db, run_id)

    assert loaded.run.id == bundle.run.id
    assert len(loaded.steps) == len(bundle.steps)
    assert loaded.prescriptions[0].patch_type == "config_diff"
    assert loaded.prescriptions[0].patch.strip()
    assert list_runs(db)[0].id == bundle.run.id


def test_connect_enables_foreign_keys(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"

    with connect(db) as conn:
        enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]

    assert enabled == 1


def test_cascade_delete_removes_children(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    save_bundle(db, bundle)

    with connect(db) as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (bundle.run.id,))
        step_count = conn.execute("SELECT count(*) FROM steps").fetchone()[0]
        rx_count = conn.execute("SELECT count(*) FROM prescriptions").fetchone()[0]

    assert step_count == 0
    assert rx_count == 0


def test_fk_rejects_orphan_step_insert(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    init_db(db)

    with connect(db) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO steps (
                id, run_id, step_type, name, started_at, ended_at, input_tokens,
                output_tokens, cost_usd, retry_count, error, redaction_mode, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "step_orphan",
                "run_missing",
                "tool",
                "missing.parent",
                "2026-05-28T00:00:00Z",
                "2026-05-28T00:00:01Z",
                0,
                0,
                0.0,
                0,
                None,
                "metadata_only",
                "{}",
            ),
        )


def test_composite_pk_collision(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    save_bundle(db, bundle)
    step = bundle.steps[0]

    with connect(db) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO steps (
                id, run_id, step_type, name, started_at, ended_at, input_tokens,
                output_tokens, cost_usd, retry_count, error, redaction_mode, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step.id,
                bundle.run.id,
                step.step_type,
                step.name,
                step.started_at,
                step.ended_at,
                step.input_tokens,
                step.output_tokens,
                step.cost_usd,
                step.retry_count,
                step.error,
                step.redaction_mode,
                "{}",
            ),
        )


def test_init_db_idempotent_three_calls(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    bundle = load_trace(Path("fixtures/clean_run.json"))
    init_db(db)
    save_bundle(db, bundle)

    # L3: capture the file's DDL stamp after first init; it must NOT change on
    # repeat init — proving zero DDL (no CREATE/ALTER) runs on the hot path
    # (Rule 5 idempotency: same result, no state change on calls 2/3).
    with connect(db) as conn:
        user_version_after_first = conn.execute("PRAGMA user_version").fetchone()[0]

    for _ in range(3):
        init_db(db)

    assert load_bundle(db, bundle.run.id).run.id == bundle.run.id
    with connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
        assert conn.execute("PRAGMA user_version").fetchone()[0] == user_version_after_first


def test_ingest_same_bundle_three_times_no_base_mutation(tmp_path: Path) -> None:
    """L3 / Rule 5: ingesting the same bundle 3x is idempotent on the FACT base —
    (a) no base-row mutation on calls 2/3, (b) deterministic outcome (reject),
    (c) the base row is never deleted."""
    db = tmp_path / "ledger.sqlite"
    bundle = load_trace(Path("fixtures/clean_run.json"))

    save_bundle(db, bundle)
    with connect(db) as conn:
        base_after_first = conn.execute(
            "SELECT * FROM runs WHERE id = ?", (bundle.run.id,)
        ).fetchone()
        steps_after_first = conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY id", (bundle.run.id,)
        ).fetchall()

    # calls 2 and 3 are deterministic rejects, never a rewrite or delete
    for _ in range(2):
        with pytest.raises(RunAlreadyRecorded):
            save_bundle(db, bundle)

    with connect(db) as conn:
        base_after = conn.execute(
            "SELECT * FROM runs WHERE id = ?", (bundle.run.id,)
        ).fetchone()
        steps_after = conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY id", (bundle.run.id,)
        ).fetchall()
        run_count = conn.execute("SELECT count(*) FROM runs").fetchone()[0]

    assert tuple(base_after) == tuple(base_after_first)  # no mutation
    assert [tuple(r) for r in steps_after] == [tuple(r) for r in steps_after_first]
    assert run_count == 1  # base never deleted, never duplicated


def test_duplicate_run_id_rejected_append_only(tmp_path: Path) -> None:
    """L2: fact tables are INSERT-ONLY. A second save of the same run_id raises
    RunAlreadyRecorded and leaves the base row byte-for-byte unchanged — a
    provenance hash computed over a row must stay valid forever."""
    db = tmp_path / "ledger.sqlite"
    bundle = load_trace(Path("fixtures/clean_run.json"))

    save_bundle(db, bundle)
    with connect(db) as conn:
        before = conn.execute(
            "SELECT * FROM runs WHERE id = ?", (bundle.run.id,)
        ).fetchone()
        steps_before = conn.execute("SELECT count(*) FROM steps").fetchone()[0]

    with pytest.raises(RunAlreadyRecorded, match="run already recorded"):
        save_bundle(db, bundle)

    with connect(db) as conn:
        after = conn.execute(
            "SELECT * FROM runs WHERE id = ?", (bundle.run.id,)
        ).fetchone()
        steps_after = conn.execute("SELECT count(*) FROM steps").fetchone()[0]
    # base row + steps untouched by the rejected re-save
    assert tuple(after) == tuple(before)
    assert steps_after == steps_before == len(bundle.steps)


def test_load_missing_run_raises_keyerror(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="run not found"):
        load_bundle(tmp_path / "ledger.sqlite", "missing")


def test_list_runs_missing_db_returns_empty(tmp_path: Path) -> None:
    assert list_runs(tmp_path / "missing.sqlite") == []


def test_metadata_json_nested_unicode_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    run = RunRecord(
        id="run_unicode",
        workflow="unicode",
        framework="test",
        provider="test",
        model="test",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
        success_label="passed",
    )
    metadata = {
        "retry_budget_patch_target": {
            "path": "settings/retries_\U0001f600.py",
            "before": "CRM_LOOKUP_RETRIES = 4",
            "after": "CRM_LOOKUP_RETRIES = 0",
        },
        "nested": {"emoji": "ok \U0001f600", "items": [1, "two"]},
    }
    expected_metadata = {"retry_budget_patch_target": metadata["retry_budget_patch_target"]}
    bundle = TraceBundle(
        run=run,
        steps=[
            StepRecord(
                id="step_unicode",
                run_id=run.id,
                step_type="tool",
                name="unicode.step",
                started_at=run.started_at,
                ended_at=run.ended_at,
                metadata=metadata,
            )
        ],
    )

    save_bundle(db, bundle)
    loaded = load_bundle(db, run.id)

    assert json.dumps(loaded.steps[0].metadata, sort_keys=True) == json.dumps(
        expected_metadata,
        sort_keys=True,
    )


def test_ensure_column_identifier_guard(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    init_db(db)

    with connect(db) as conn, pytest.raises(AssertionError):
        _ensure_column(conn, "bad-table", "column", "TEXT")


def test_sqlite_rejects_invalid_patch_type(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    init_db(db)

    with connect(db) as conn:
        conn.execute(
            """
            INSERT INTO runs (
                id, workflow, framework, provider, model, started_at, ended_at, success_label,
                prompt_hash, config_hash, total_cost_usd, total_latency_ms,
                total_input_tokens, total_output_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run_invalid_patch_type",
                "workflow",
                "framework",
                "provider",
                "model",
                "2026-05-28T00:00:00Z",
                "2026-05-28T00:00:01Z",
                "failed",
                "",
                "",
                0.0,
                0,
                0,
                0,
            ),
        )
        try:
            conn.execute(
                """
                INSERT INTO prescriptions (
                    id, run_id, severity, root_cause, one_line_fix, evidence_json,
                    patch_type, patch, expected_impact_json, regression_test_template
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "rx_invalid_patch_type",
                    "run_invalid_patch_type",
                    "medium",
                    "bad",
                    "bad",
                    "[]",
                    "invalid",
                    "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1 +1 @@\n-a\n+b\n",
                    "{}",
                    "def test_bad(): pass",
                ),
            )
        except sqlite3.IntegrityError:
            return
        raise AssertionError("SQLite accepted invalid patch_type")
