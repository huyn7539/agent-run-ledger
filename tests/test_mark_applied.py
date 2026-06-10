"""`arl mark-applied` — the apply-gate becomes measurable.

The product's stated success metric is "users APPLY the fix" (README Status).
Nothing recorded that until now. mark-applied writes an applied-event into the
run's outcome_json attach-point (LR1) — judgment-side annotation, merge-only,
never clobbering other outcome keys. Rule 5: marking twice is a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_run_ledger.cli import app
from agent_run_ledger.core.storage import load_bundle


def _seed(tmp_path: Path) -> Path:
    db = tmp_path / "l.sqlite"
    r = CliRunner().invoke(app, ["run-demo", "--variant", "retry-loop", "--db", str(db)])
    assert r.exit_code == 0, r.output
    return db


def _mark(db: Path, run_id: str = "run_retry_loop", at: str = "2026-06-11T12:00:00Z"):
    return CliRunner().invoke(app, ["mark-applied", run_id, "--db", str(db), "--at", at])


def test_marks_run_as_applied(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    r = _mark(db)
    assert r.exit_code == 0, r.output
    outcome = json.loads(load_bundle(db, "run_retry_loop").run.outcome_json)
    assert outcome["applied"] == {"at": "2026-06-11T12:00:00Z"}


def test_three_times_idempotent(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    assert _mark(db).exit_code == 0
    first = load_bundle(db, "run_retry_loop").run.outcome_json
    for _ in range(2):
        r = _mark(db)
        assert r.exit_code == 0, r.output
        assert "already" in r.output.lower()
        assert load_bundle(db, "run_retry_loop").run.outcome_json == first


def test_unknown_run_fails_closed(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    r = _mark(db, run_id="run_nope")
    assert r.exit_code == 1
    assert "run_nope" in r.output


def test_merge_preserves_existing_outcome_keys(tmp_path: Path) -> None:
    db = _seed(tmp_path)
    import sqlite3

    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE runs SET outcome_json = ? WHERE id = ?",
            (json.dumps({"shadow_score": 0.9}), "run_retry_loop"),
        )
    assert _mark(db).exit_code == 0
    outcome = json.loads(load_bundle(db, "run_retry_loop").run.outcome_json)
    assert outcome["shadow_score"] == 0.9 and "applied" in outcome
