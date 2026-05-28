from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent_run_ledger.core.models import PrescriptionRecord, RunRecord, StepRecord, TraceBundle


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                workflow TEXT NOT NULL,
                framework TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                success_label TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                total_cost_usd REAL NOT NULL,
                total_latency_ms INTEGER NOT NULL,
                total_input_tokens INTEGER NOT NULL,
                total_output_tokens INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS steps (
                id TEXT NOT NULL,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                step_type TEXT NOT NULL,
                name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                retry_count INTEGER NOT NULL,
                error TEXT,
                redaction_mode TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (run_id, id)
            );

            CREATE TABLE IF NOT EXISTS prescriptions (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                severity TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                one_line_fix TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                patch_type TEXT NOT NULL CHECK (
                    patch_type IN ('unified_diff', 'code_snippet', 'config_diff', 'regression_test')
                ),
                patch TEXT NOT NULL,
                expected_impact_json TEXT NOT NULL,
                regression_test_template TEXT NOT NULL
            );
            """
        )
        _ensure_column(
            conn,
            "prescriptions",
            "patch_type",
            (
                "TEXT NOT NULL DEFAULT 'code_snippet' CHECK "
                "(patch_type IN ('unified_diff', 'code_snippet', 'config_diff', 'regression_test'))"
            ),
        )


def save_bundle(db_path: Path, bundle: TraceBundle) -> str:
    bundle.validate()
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM prescriptions WHERE run_id = ?", (bundle.run.id,))
        conn.execute("DELETE FROM steps WHERE run_id = ?", (bundle.run.id,))
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                id, workflow, framework, provider, model, started_at, ended_at, success_label,
                prompt_hash, config_hash, total_cost_usd, total_latency_ms,
                total_input_tokens, total_output_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bundle.run.id,
                bundle.run.workflow,
                bundle.run.framework,
                bundle.run.provider,
                bundle.run.model,
                bundle.run.started_at,
                bundle.run.ended_at,
                bundle.run.success_label,
                bundle.run.prompt_hash,
                bundle.run.config_hash,
                bundle.run.total_cost_usd,
                bundle.run.total_latency_ms,
                bundle.run.total_input_tokens,
                bundle.run.total_output_tokens,
            ),
        )
        for step in bundle.steps:
            conn.execute(
                """
                INSERT INTO steps (
                    id, run_id, step_type, name, started_at, ended_at, input_tokens,
                    output_tokens, cost_usd, retry_count, error, redaction_mode, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.id,
                    step.run_id,
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
                    json.dumps(step.metadata, sort_keys=True),
                ),
            )
        for prescription in bundle.prescriptions:
            conn.execute(
                """
                INSERT INTO prescriptions (
                    id, run_id, severity, root_cause, one_line_fix, evidence_json, patch_type, patch,
                    expected_impact_json, regression_test_template
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prescription.id,
                    prescription.run_id,
                    prescription.severity,
                    prescription.root_cause,
                    prescription.one_line_fix,
                    json.dumps(prescription.evidence, sort_keys=True),
                    prescription.patch_type,
                    prescription.patch,
                    json.dumps(prescription.expected_impact, sort_keys=True),
                    prescription.regression_test_template,
                ),
            )
    return bundle.run.id


def load_bundle(db_path: Path, run_id: str) -> TraceBundle:
    init_db(db_path)
    with connect(db_path) as conn:
        run_row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run_row is None:
            raise KeyError(f"run not found: {run_id}")
        step_rows = conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY started_at, id", (run_id,)
        ).fetchall()
        prescription_rows = conn.execute(
            "SELECT * FROM prescriptions WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
    run = _run_from_row(run_row)
    steps = [_step_from_row(row) for row in step_rows]
    prescriptions = [_prescription_from_row(row) for row in prescription_rows]
    bundle = TraceBundle(run=run, steps=steps, prescriptions=prescriptions)
    bundle.validate()
    return bundle


def list_runs(db_path: Path) -> list[RunRecord]:
    if not db_path.exists():
        return []
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC, id DESC").fetchall()
    return [_run_from_row(row) for row in rows]


def _run_from_row(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        workflow=row["workflow"],
        framework=row["framework"],
        provider=row["provider"],
        model=row["model"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        success_label=row["success_label"],
        prompt_hash=row["prompt_hash"],
        config_hash=row["config_hash"],
        total_cost_usd=float(row["total_cost_usd"]),
        total_latency_ms=int(row["total_latency_ms"]),
        total_input_tokens=int(row["total_input_tokens"]),
        total_output_tokens=int(row["total_output_tokens"]),
    )


def _step_from_row(row: sqlite3.Row) -> StepRecord:
    return StepRecord(
        id=row["id"],
        run_id=row["run_id"],
        step_type=row["step_type"],
        name=row["name"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        cost_usd=float(row["cost_usd"]),
        retry_count=int(row["retry_count"]),
        error=row["error"],
        redaction_mode=row["redaction_mode"],
        metadata=json.loads(row["metadata_json"]),
    )


def _prescription_from_row(row: sqlite3.Row) -> PrescriptionRecord:
    return PrescriptionRecord.from_dict(
        {
            "id": row["id"],
            "severity": row["severity"],
            "root_cause": row["root_cause"],
            "one_line_fix": row["one_line_fix"],
            "evidence": json.loads(row["evidence_json"]),
            "patch_type": row["patch_type"],
            "patch": row["patch"],
            "expected_impact": json.loads(row["expected_impact_json"]),
            "regression_test_template": row["regression_test_template"],
        },
        row["run_id"],
    )


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    assert table.isidentifier()
    assert column.isidentifier()
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
