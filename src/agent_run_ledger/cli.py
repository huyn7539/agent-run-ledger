from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from agent_run_ledger.adapters.claude_code import (
    ClaudeCodeSessionError,
    bundle_from_session,
    find_recent_sessions,
    load_claude_session,
    looks_like_claude_session_file,
)
from agent_run_ledger.adapters.codex import (
    CodexRolloutError,
    bundle_from_rollout,
    find_recent_rollouts,
    load_codex_rollout,
    looks_like_jsonl,
)
from agent_run_ledger.adapters.openai import NoSpansCapturedError, bundle_from_recorded_trace
from agent_run_ledger.core.compare import compare_bundles
from agent_run_ledger.core.cost import cost_display
from agent_run_ledger.core.demo import load_demo_bundle
from agent_run_ledger.core.io import TraceParseError, load_json_object, write_trace
from agent_run_ledger.core.models import TraceBundle, TraceValidationError
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.receipt import PROOF_LEVELS, build_receipts
from agent_run_ledger.core.report import render_comparison, write_report
from agent_run_ledger.core.storage import (
    RunAlreadyRecorded,
    cloud_sync_warning,
    init_db,
    list_runs,
    load_bundle,
    save_bundle,
)

app = typer.Typer(help="Agent Run Ledger CLI")
console = Console()
T = TypeVar("T")


def default_db() -> Path:
    return Path(os.environ.get("ARL_DB", ".arl/ledger.sqlite"))


def _warn_cloud_sync(db: Path) -> None:
    """L11: print a one-time warning if the ledger lives in a cloud-sync dir."""
    warning = cloud_sync_warning(db)
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")


@app.command("init")
def init(db: Path = typer.Option(default_factory=default_db, help="SQLite database path.")) -> None:
    _warn_cloud_sync(db)
    init_db(db)
    console.print(f"initialized ledger: {db}")


@app.command("run-demo")
def run_demo(
    variant: str = typer.Option("retry-loop", help="Demo variant: retry-loop or clean."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    bundle = load_demo_bundle(variant)
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    run_id = save_bundle(db, bundle)
    console.print(f"stored demo run: {run_id}")


def _load_any_trace(path: Path) -> TraceBundle:
    """Route an import by FORMAT (provider-neutral detection):

    * a line-delimited JSON log whose lines carry the Claude Code session shape
      (top-level ``sessionId`` + ``uuid``) -> the Claude Code adapter;
    * any other line-delimited JSON log -> the Codex rollout adapter;
    * a single JSON object carrying ``trace`` + ``spans`` (and no ``run``) -> a
      recorded OpenAI-SDK trace export -> the OpenAI adapter's
      ``bundle_from_recorded_trace`` (pure parsing; no SDK dependency);
    * any other single JSON object -> the neutral TraceBundle path.

    The detection names no provider field beyond the adapters' own probes — only
    file SHAPE — so the core stays provider-neutral; provider-specific parsing
    lives entirely in the adapters. All single-object reads share
    ``load_json_object``'s defensive size/depth/encoding bounds."""
    if looks_like_jsonl(path):
        if looks_like_claude_session_file(path):
            return bundle_from_session(load_claude_session(path))
        return bundle_from_rollout(load_codex_rollout(path))
    data = load_json_object(path)
    if "run" not in data and "trace" in data and "spans" in data:
        return bundle_from_recorded_trace(data)
    try:
        return TraceBundle.from_dict(data)
    except TraceValidationError as exc:
        raise TraceParseError(f"invalid trace bundle: {exc}") from exc


@app.command("import")
def import_trace(
    path: Path = typer.Argument(..., help="Trace file: a single-object trace JSON, or a Codex .jsonl rollout log."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    bundle = _friendly_or_exit(lambda: _load_any_trace(path))
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    # Rule 5: importing the same run twice is safe and quiet. The fact tables are
    # append-only (L2), so a re-import is a no-op, not a crash — print the existing
    # run id and the next command instead of a stack trace.
    try:
        run_id = save_bundle(db, bundle)
    except RunAlreadyRecorded:
        run_id = bundle.run.id
        console.print(f"already imported: {run_id}")
        console.print(f"  view it:  arl report --run {run_id}")
        return
    console.print(f"imported run: {run_id}")


@app.command("export")
def export_trace(
    run: str = typer.Option(..., "--run", help="Run id."),
    out: Path = typer.Option(..., "--out", help="Output JSON path."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    bundle = _load_bundle_or_exit(db, run)
    _friendly_or_exit(lambda: write_trace(bundle, out))
    console.print(f"wrote trace: {out}")


@app.command("report")
def report(
    run: str = typer.Option(..., "--run", help="Run id."),
    out: Path | None = typer.Option(None, "--out", help="Output HTML path."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    bundle = _load_bundle_or_exit(db, run)
    output = out or Path(".arl") / "reports" / f"{run}.html"
    _friendly_or_exit(lambda: write_report(bundle, output))
    console.print(f"wrote report: {output}")


@app.command("compare")
def compare(
    left: str = typer.Option(..., "--left", help="Left/baseline run id."),
    right: str = typer.Option(..., "--right", help="Right/candidate run id."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    comparison = compare_bundles(_load_bundle_or_exit(db, left), _load_bundle_or_exit(db, right))
    console.print(render_comparison(comparison))


@app.command("list-runs")
def list_runs_cmd(
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    rows = list_runs(db)
    table = Table(title="Agent Run Ledger")
    table.add_column("Run")
    table.add_column("Workflow")
    table.add_column("Outcome")
    table.add_column("Cost")
    table.add_column("Tokens")
    for run in rows:
        # L7/LR2: show the cost computed on read from the FACTS, never the cached
        # total_cost_usd (which a price-table change — or a $0 capture cache —
        # makes stale). Load the bundle per run; linear in run count.
        bundle = load_bundle(db, run.id)
        # Use the DISCLOSURE form (A2): an unpriced real-token run reads as
        # "unpriced (...)" here too, never a misleading bare $0 — consistent with the
        # demo + HTML report (was: cost_on_read -> a silent $0 on this surface).
        table.add_row(
            run.id,
            run.workflow,
            run.success_label,
            cost_display(bundle),
            str(run.total_input_tokens + run.total_output_tokens),
        )
    console.print(table)


# The loop contract (Task 57): a hook / CI step / Ralph-style loop gates on these.
# 0 = clean, 3 = receipt(s) fired, 1 = error (fail closed: unreadable is NOT clean).
# 2 stays reserved for click/typer usage errors.
EXIT_CLEAN = 0
EXIT_RECEIPTS = 3
VERDICT_SCHEMA = "arl.verdict/v1"


@app.command("verdict")
def verdict(
    path: Path | None = typer.Argument(
        None,
        help="Trace file: single-object trace JSON or a Codex .jsonl rollout. Omit with --latest.",
    ),
    latest: bool = typer.Option(
        False, "--latest", help="Grade the newest local Codex session rollout instead of a path."
    ),
    latest_claude: bool = typer.Option(
        False, "--latest-claude", help="Grade the newest local Claude Code session instead of a path."
    ),
    sessions_root: Path | None = typer.Option(
        None, "--sessions-root", help="Codex sessions root (default: ~/.codex/sessions)."
    ),
    claude_projects_root: Path | None = typer.Option(
        None,
        "--claude-projects-root",
        help="Claude Code projects root (default: ~/.claude/projects).",
    ),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
    json_out: bool = typer.Option(
        False, "--json", help="Machine-readable arl.verdict/v1 JSON on stdout."
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Record the run in the ledger (idempotent)."
    ),
) -> None:
    """Grade a run for a loop: exit 0 = clean, 3 = receipt(s) fired, 1 = error.

    This is the verdict layer for autonomous loops: the graded repair receipt as a
    machine-consumable exit, so "the run finished" and "the run is verified clean"
    stop being the same claim. An unreadable run exits 1 — fail closed, never
    silently clean."""
    if latest and latest_claude:
        console.print("error: pass --latest OR --latest-claude, not both")
        raise typer.Exit(1)
    if latest:
        rollouts = find_recent_rollouts(sessions_root)
        if not rollouts:
            root_label = sessions_root or "~/.codex/sessions"
            console.print(f"error: no Codex session rollouts found under {root_label}")
            raise typer.Exit(1)
        path = rollouts[0]
    if latest_claude:
        sessions = find_recent_sessions(claude_projects_root)
        if not sessions:
            root_label = claude_projects_root or "~/.claude/projects"
            console.print(f"error: no Claude Code sessions found under {root_label}")
            raise typer.Exit(1)
        path = sessions[0]
    if path is None:
        console.print(
            "error: pass a trace path, or use --latest (Codex) / --latest-claude (Claude Code)"
        )
        raise typer.Exit(1)

    bundle = _friendly_or_exit(lambda: _load_any_trace(path))
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    receipts = build_receipts(bundle)

    if save:
        # Idempotent by design (Rule 5): a verdict on an already-recorded run is a
        # read, not a conflict.
        try:
            save_bundle(db, bundle)
        except RunAlreadyRecorded:
            pass

    if json_out:
        max_level = (
            max((r.proof_level for r in receipts), key=PROOF_LEVELS.index) if receipts else None
        )
        payload = {
            "schema": VERDICT_SCHEMA,
            "run_id": bundle.run.id,
            "verdict": "receipts" if receipts else "clean",
            "receipt_count": len(receipts),
            "max_proof_level": max_level,
            "receipts": [asdict(r) for r in receipts],
        }
        # Plain stdout JSON (typer.echo, not rich): hooks parse this byte-for-byte.
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(EXIT_RECEIPTS if receipts else EXIT_CLEAN)

    if not receipts:
        console.print(
            f"verdict: clean — no structural failure detected in {bundle.run.id} "
            "(the honest answer; ARL does not invent receipts)"
        )
        raise typer.Exit(EXIT_CLEAN)

    console.print(f"verdict: {len(receipts)} repair receipt(s) fired on {bundle.run.id}")
    for r in receipts:
        console.print(
            f"  [{r.proof_level} | confidence {r.confidence}] {r.observed_failure}: "
            f"{r.repair_artifact.get('one_line_fix', '')}"
        )
    console.print("  review before applying — ARL advises, you apply:")
    console.print(f"    arl report --run {bundle.run.id}")
    raise typer.Exit(EXIT_RECEIPTS)


def _load_bundle_or_exit(db: Path, run_id: str):
    try:
        return load_bundle(db, run_id)
    except KeyError as exc:
        console.print(f"error: {exc.args[0]}")
        raise typer.Exit(1) from exc


def _friendly_or_exit(action: Callable[[], T]) -> T:
    try:
        return action()
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        TraceParseError,
        TraceValidationError,
        CodexRolloutError,
        ClaudeCodeSessionError,
        NoSpansCapturedError,
    ) as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
