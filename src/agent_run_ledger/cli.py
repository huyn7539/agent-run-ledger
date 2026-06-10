from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from agent_run_ledger.adapters.codex import (
    CodexRolloutError,
    bundle_from_rollout,
    load_codex_rollout,
    looks_like_jsonl,
)
from agent_run_ledger.core.compare import compare_bundles
from agent_run_ledger.core.cost import cost_display
from agent_run_ledger.core.demo import load_demo_bundle
from agent_run_ledger.core.io import TraceParseError, load_trace, write_trace
from agent_run_ledger.core.models import TraceBundle, TraceValidationError
from agent_run_ledger.core.prescriptions import analyze_bundle
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

    * a line-delimited JSON log (``.jsonl`` / multiple JSON objects) -> the Codex
      rollout adapter, which maps the provider log into the neutral TraceBundle;
    * a single JSON object -> the existing TraceBundle path (``load_trace``).

    The detection names no provider field — only the file shape — so the core
    stays provider-neutral; the Codex-specific parsing lives entirely in the
    adapter."""
    if looks_like_jsonl(path):
        return bundle_from_rollout(load_codex_rollout(path))
    return load_trace(path)


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
    ) as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
