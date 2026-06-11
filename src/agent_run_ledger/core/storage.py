from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from agent_run_ledger.core.models import (
    SCHEMA_VERSION,
    PrescriptionRecord,
    RunRecord,
    StepRecord,
    TraceBundle,
)

# L11: directory names whose presence in a ledger path means a cloud-sync daemon
# (Dropbox / OneDrive / iCloud) may egress plaintext to a vendor cloud — defeating
# zero-egress not via ARL's code but via where the file lives. We WARN, not block.
_CLOUD_SYNC_MARKERS = ("Dropbox", "OneDrive", "Mobile Documents")


def cloud_sync_warning(db_path: Path | str) -> str | None:
    """Return a one-time warning string if *db_path* resolves inside a known
    cloud-sync directory, else None (L11). Pure function — caller decides how to
    surface it (the CLI prints it once)."""
    parts = set(Path(db_path).expanduser().parts)
    for marker in _CLOUD_SYNC_MARKERS:
        if marker in parts:
            return (
                f"warning: ledger path is inside a cloud-sync directory ({marker}). "
                "Local-first / zero-egress can be defeated by a sync daemon "
                "uploading plaintext. Move .arl outside synced folders, or accept "
                "that your provider's cloud will hold a copy."
            )
    return None


def _set_file_permissions(path: Path) -> None:
    """Lock the ledger dir to 0o700 and the file to 0o600 on POSIX (L11).

    Windows POSIX modes are a no-op (the cloud-sync warning covers that case).
    Best-effort: a chmod failure must never break a save."""
    if os.name != "posix":
        return
    try:
        os.chmod(path.parent, 0o700)
        if path.exists():
            os.chmod(path, 0o600)
    except OSError:
        pass


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # PRAGMA foreign_keys must be issued immediately after connect, before any
    # DML — it is a silent no-op inside an open transaction (ADR-001 C4). This
    # is what makes ON DELETE CASCADE and orphan-step rejection actually fire.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


# The DDL/migration version stamped on the FILE via PRAGMA user_version —
# "which additive DDL migrations has this file had" (distinct from the per-row
# schema_version, which records the shape an individual record was written
# under). Bump this and append an `_ensure_column` for any additive change.
# v2: runs.adapter_provenanced (P1-1 spoof hardening, 2026-06-11).
USER_VERSION = 2


class RunAlreadyRecorded(Exception):
    """Raised when a run id already exists — fact tables are append-only (L2).

    The base (runs, steps) is never deleted or overwritten, so a provenance hash
    computed over a row stays valid forever. Re-capture, if ever supported, is a
    NEW immutable row — never a rewrite of the existing one.
    """


def init_db(db_path: Path) -> None:
    """Create tables at the latest DDL and run additive migrations (idempotent).

    A brand-new file is created at the latest shape and stamped at
    ``USER_VERSION``. An existing older file gains any missing columns via
    ``_ensure_column`` (additive ALTER, no row rewrite) and is re-stamped. The
    ALTERs are no-ops on an already-current file (Rule 5 idempotency).
    """
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL DEFAULT '0.1',
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
                total_output_tokens INTEGER NOT NULL,
                ingested_at TEXT NOT NULL DEFAULT '',
                billing_mode TEXT NOT NULL DEFAULT 'unknown'
                    CHECK (billing_mode IN
                        ('pay_per_use','subscription','enterprise_contract','local','unknown')),
                price_table_version TEXT,
                provenance_hash TEXT,
                outcome_json TEXT,
                adapter_provenanced INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS steps (
                id TEXT NOT NULL,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                parent_step_id TEXT,
                span_kind TEXT,
                retry_scope TEXT,
                input_fingerprint TEXT,
                step_type TEXT NOT NULL,
                name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                provider_reported_cost_usd REAL,
                cost_usd REAL NOT NULL,
                retry_count INTEGER NOT NULL,
                error TEXT,
                error_class TEXT,
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
                regression_test_template TEXT NOT NULL,
                failure_class TEXT NOT NULL DEFAULT 'retry_loop'
                    CHECK (failure_class IN ('retry_loop', 'artifact_failure'))
            );

            -- Task 60: the governed-apply experiment registry (pre-registered
            -- metric + exact revert material). LOCAL-SECRET by classification:
            -- before/after block text MAY mirror CLAUDE.md content; this table
            -- is NEVER exported (write_trace serializes a TraceBundle only) —
            -- the io._scrub_for_share chokepoint comment names this class.
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL,
                proposal_class TEXT NOT NULL,
                tool TEXT NOT NULL,
                claudemd_path TEXT NOT NULL,
                line TEXT NOT NULL,
                before_block TEXT NOT NULL,
                after_block TEXT NOT NULL,
                assignment_basis TEXT NOT NULL,
                mde TEXT NOT NULL,
                eps_harm TEXT NOT NULL,
                min_n INTEGER NOT NULL,
                baseline_n INTEGER NOT NULL,
                baseline_k INTEGER NOT NULL,
                applied_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('applied', 'kept', 'reverted', 'review')
                )
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
        # L1/L2/L7 additive migration: bring a pre-LOCK-NOW table up to shape.
        # No-op on a file already created with the columns (IF NOT EXISTS above).
        # ALTER ADD COLUMN cannot carry a CHECK constraint, so billing_mode's
        # enum is enforced at the model layer (RunRecord.__post_init__); the CHECK
        # in the CREATE is belt-and-suspenders for fresh DBs.
        _ensure_column(conn, "runs", "schema_version", "TEXT NOT NULL DEFAULT '0.1'")
        _ensure_column(conn, "runs", "ingested_at", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "runs", "billing_mode", "TEXT NOT NULL DEFAULT 'unknown'")
        _ensure_column(conn, "runs", "price_table_version", "TEXT")
        _ensure_column(conn, "runs", "provenance_hash", "TEXT")
        _ensure_column(conn, "runs", "outcome_json", "TEXT")
        _ensure_column(conn, "steps", "cached_input_tokens", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "steps", "reasoning_tokens", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "steps", "provider_reported_cost_usd", "REAL")
        _ensure_column(conn, "steps", "parent_step_id", "TEXT")
        _ensure_column(conn, "steps", "span_kind", "TEXT")
        _ensure_column(conn, "steps", "error_class", "TEXT")
        # Trace-derived retry FACTS (content-free); additive, nullable.
        _ensure_column(conn, "steps", "retry_scope", "TEXT")
        _ensure_column(conn, "steps", "input_fingerprint", "TEXT")
        # Task 58: bounded failure class on the (recomputable) judgment side.
        # ALTER ADD COLUMN cannot carry CHECK; the model layer enforces the
        # closed vocabulary (models._failure_class) — same split as billing_mode.
        _ensure_column(
            conn, "prescriptions", "failure_class", "TEXT NOT NULL DEFAULT 'retry_loop'"
        )
        # v2 (P1-1 spoof hardening): the in-process adapter trust bit, persisted
        # because the DB is ARL's own write. DEFAULT 0 = pre-v2 rows fail CLOSED:
        # an old row re-graded by report caps artifact receipts at L0 until
        # re-captured through an adapter, never the other way around.
        _ensure_column(conn, "runs", "adapter_provenanced", "INTEGER NOT NULL DEFAULT 0")
        # Stamp the file's DDL version. Stays put once at USER_VERSION (no DDL on
        # the hot path → user_version does not increment on repeat init).
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current < USER_VERSION:
            conn.execute(f"PRAGMA user_version = {USER_VERSION}")
    # L11: lock the dir/file down once the file exists on disk.
    _set_file_permissions(Path(db_path))


def save_bundle(db_path: Path, bundle: TraceBundle) -> str:
    """Persist a bundle. Fact tables (runs, steps) are INSERT-ONLY (L2).

    A duplicate ``run.id`` raises ``RunAlreadyRecorded`` and nothing is written —
    base rows are never deleted or overwritten. Only the recomputable judgment
    side (``prescriptions``) is replaced per-run.
    """
    bundle.validate()
    init_db(db_path)
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT 1 FROM runs WHERE id = ?", (bundle.run.id,)
        ).fetchone()
        if existing is not None:
            raise RunAlreadyRecorded(f"run already recorded: {bundle.run.id!r}")
        conn.execute(
            """
            INSERT INTO runs (
                id, schema_version, workflow, framework, provider, model, started_at,
                ended_at, success_label, prompt_hash, config_hash, total_cost_usd,
                total_latency_ms, total_input_tokens, total_output_tokens, ingested_at,
                billing_mode, price_table_version, provenance_hash, outcome_json,
                adapter_provenanced
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bundle.run.id,
                bundle.run.schema_version,
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
                bundle.ingested_at,
                bundle.run.billing_mode,
                bundle.run.price_table_version,
                bundle.run.provenance_hash,
                bundle.run.outcome_json,
                1 if bundle.adapter_provenanced else 0,
            ),
        )
        for step in bundle.steps:
            conn.execute(
                """
                INSERT INTO steps (
                    id, run_id, parent_step_id, span_kind, retry_scope, input_fingerprint,
                    step_type, name, started_at,
                    ended_at, input_tokens, output_tokens, cached_input_tokens,
                    reasoning_tokens, provider_reported_cost_usd, cost_usd, retry_count,
                    error, error_class, redaction_mode, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.id,
                    step.run_id,
                    step.parent_step_id,
                    step.span_kind,
                    step.retry_scope,
                    step.input_fingerprint,
                    step.step_type,
                    step.name,
                    step.started_at,
                    step.ended_at,
                    step.input_tokens,
                    step.output_tokens,
                    step.cached_input_tokens,
                    step.reasoning_tokens,
                    step.provider_reported_cost_usd,
                    step.cost_usd,
                    step.retry_count,
                    step.error,
                    step.error_class,
                    step.redaction_mode,
                    json.dumps(step.metadata, sort_keys=True),
                ),
            )
        # Prescriptions are the recomputable judgment side — safe to replace
        # per-run (the new run inserted above had no prior prescriptions, so this
        # is a no-op on first save; it matters only on a re-analyze path).
        conn.execute("DELETE FROM prescriptions WHERE run_id = ?", (bundle.run.id,))
        for prescription in bundle.prescriptions:
            conn.execute(
                """
                INSERT INTO prescriptions (
                    id, run_id, severity, root_cause, one_line_fix, evidence_json, patch_type, patch,
                    expected_impact_json, regression_test_template, failure_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    prescription.failure_class,
                ),
            )
    # L11: re-assert tight perms after writing the bundle.
    _set_file_permissions(Path(db_path))
    return bundle.run.id


def _load_bundle_from(conn: sqlite3.Connection, run_id: str) -> TraceBundle:
    """Map one run + its steps/prescriptions from an OPEN connection. Shared by
    the writable loader and the strictly-read-only loader (Task 59) so the SQL
    and row mapping cannot drift between them."""
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
    bundle = TraceBundle(
        run=run,
        steps=steps,
        prescriptions=prescriptions,
        schema_version=run.schema_version,
        ingested_at=_row_get(run_row, "ingested_at", ""),
        # Trusted on read-back: the row was written by save_bundle from the
        # in-process bit (file imports store 0). Pre-v2 rows default 0 (closed).
        adapter_provenanced=bool(_row_get(run_row, "adapter_provenanced", 0)),
    )
    bundle.validate()
    return bundle


def load_bundle(db_path: Path, run_id: str) -> TraceBundle:
    init_db(db_path)
    with connect(db_path) as conn:
        return _load_bundle_from(conn, run_id)


def list_runs(db_path: Path) -> list[RunRecord]:
    if not db_path.exists():
        return []
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC, id DESC").fetchall()
    return [_run_from_row(row) for row in rows]


def _sqlite_ro_uri(db_path: Path) -> str:
    """A read-only SQLite URI for *db_path*. Hand-rolled minimal percent-encoding
    (%, ?, #, space) because ``urllib`` is banned package-wide by the egress
    guard; SQLite decodes %HH in URI filenames."""
    p = db_path.resolve().as_posix()
    for ch, enc in (("%", "%25"), ("?", "%3F"), ("#", "%23"), (" ", "%20")):
        p = p.replace(ch, enc)
    if not p.startswith("/"):
        p = "/" + p  # Windows drive paths: file:///C:/...
    return f"file://{p}?mode=ro"


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """A STRICTLY read-only connection for read surfaces (``arl serve``, Task 59).

    URI ``mode=ro`` + ``PRAGMA query_only=ON``; never mkdirs, never migrates,
    never chmods — a GET must not write (Codex spec-review F8: ``load_bundle``'s
    ``init_db`` runs DDL + chmod on read; serve must never take that path).
    Raises ``sqlite3.OperationalError`` when the file does not exist — the
    caller maps that to a 404/empty surface; nothing is auto-created."""
    conn = sqlite3.connect(_sqlite_ro_uri(db_path), uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def load_bundle_readonly(db_path: Path, run_id: str) -> TraceBundle:
    """Read-only ``load_bundle`` (per-request, short-lived — closes the handle)."""
    conn = connect_readonly(db_path)
    try:
        return _load_bundle_from(conn, run_id)
    finally:
        conn.close()


def list_runs_readonly(db_path: Path) -> list[RunRecord]:
    """Read-only ``list_runs`` (per-request, short-lived — closes the handle)."""
    if not db_path.exists():
        return []
    conn = connect_readonly(db_path)
    try:
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC, id DESC").fetchall()
    finally:
        conn.close()
    return [_run_from_row(row) for row in rows]


def merge_run_outcome(db_path: Path, run_id: str, key: str, value: dict) -> str:
    """Merge ``{key: value}`` into a run's ``outcome_json`` attach-point (LR1).

    outcome_json is the judgment-side annotation slot on the immutable fact row
    — the ONE runs column designed for post-hoc attachment (LR1), so this UPDATE
    does not breach the insert-only discipline that protects provenance hashes
    (which cover the captured facts, not this slot). Merge-only: other outcome
    keys survive. Idempotent (Rule 5): if ``key`` is already present with the
    same value, nothing is written and ``"already"`` is returned; a present key
    with a DIFFERENT value also returns ``"already"`` (first write wins — an
    applied-event is a fact about the past, not a mutable preference).
    Raises ``KeyError`` for an unknown run.
    """
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT outcome_json FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run not found: {run_id}")
        outcome = json.loads(row["outcome_json"]) if row["outcome_json"] else {}
        if key in outcome:
            return "already"
        outcome[key] = value
        conn.execute(
            "UPDATE runs SET outcome_json = ? WHERE id = ?",
            (json.dumps(outcome, sort_keys=True), run_id),
        )
    return "set"


# --- Task 60: experiments registry (governed-apply lane) ----------------------

def save_experiment(db_path: Path, fields: dict) -> str:
    """Insert an experiment row; idempotent on experiment_id (Rule 5): an
    existing id returns "already" and writes NOTHING (an apply is a fact about
    the past — first write wins, same discipline as merge_run_outcome)."""
    init_db(db_path)
    with connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ?", (fields["experiment_id"],)
        ).fetchone()
        if exists:
            return "already"
        conn.execute(
            """INSERT INTO experiments (
                experiment_id, proposal_id, proposal_class, tool, claudemd_path,
                line, before_block, after_block, assignment_basis, mde, eps_harm,
                min_n, baseline_n, baseline_k, applied_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fields["experiment_id"], fields["proposal_id"], fields["proposal_class"],
                fields["tool"], fields["claudemd_path"], fields["line"],
                fields["before_block"], fields["after_block"], fields["assignment_basis"],
                fields["mde"], fields["eps_harm"], fields["min_n"],
                fields["baseline_n"], fields["baseline_k"], fields["applied_at"],
                fields["status"],
            ),
        )
    return "saved"


def list_experiments(db_path: Path, status: str | None = None) -> list[dict]:
    if not db_path.exists():
        return []
    init_db(db_path)
    with connect(db_path) as conn:
        if status is None:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY applied_at, experiment_id"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE status = ? ORDER BY applied_at, experiment_id",
                (status,),
            ).fetchall()
    return [dict(row) for row in rows]


def set_experiment_status(db_path: Path, experiment_id: str, status: str) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE experiments SET status = ? WHERE experiment_id = ?",
            (status, experiment_id),
        )
        if cur.rowcount == 0:
            raise KeyError(f"experiment not found: {experiment_id}")


def _row_get(row: sqlite3.Row, key: str, default=None):
    """Read *key* from a Row, returning *default* if the column is absent.

    Lets the loader tolerate a row written under an older shape in a mixed-
    version file (L1) without raising on a missing column.
    """
    return row[key] if key in row.keys() else default


def _run_from_row(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        schema_version=_row_get(row, "schema_version", SCHEMA_VERSION),
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
        billing_mode=_row_get(row, "billing_mode", "unknown") or "unknown",
        price_table_version=_row_get(row, "price_table_version"),
        provenance_hash=_row_get(row, "provenance_hash"),
        outcome_json=_row_get(row, "outcome_json"),
    )


def _step_from_row(row: sqlite3.Row) -> StepRecord:
    _prc = _row_get(row, "provider_reported_cost_usd")
    return StepRecord(
        id=row["id"],
        run_id=row["run_id"],
        parent_step_id=_row_get(row, "parent_step_id"),
        span_kind=_row_get(row, "span_kind"),
        retry_scope=_row_get(row, "retry_scope"),
        input_fingerprint=_row_get(row, "input_fingerprint"),
        cached_input_tokens=int(_row_get(row, "cached_input_tokens", 0) or 0),
        reasoning_tokens=int(_row_get(row, "reasoning_tokens", 0) or 0),
        provider_reported_cost_usd=float(_prc) if _prc is not None else None,
        step_type=row["step_type"],
        name=row["name"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        cost_usd=float(row["cost_usd"]),
        retry_count=int(row["retry_count"]),
        error=row["error"],
        error_class=_row_get(row, "error_class"),
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
            "failure_class": _row_get(row, "failure_class", "retry_loop"),
        },
        row["run_id"],
    )


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    assert table.isidentifier()
    assert column.isidentifier()
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
