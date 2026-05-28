from pathlib import Path
import sqlite3

from agent_run_ledger.core.io import load_trace
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.storage import connect, init_db, list_runs, load_bundle, save_bundle


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
