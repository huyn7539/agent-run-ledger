from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agent_run_ledger.core.compare import compare_bundles
from agent_run_ledger.core.demo import load_demo_bundle
from agent_run_ledger.core.io import load_trace, write_trace
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.report import render_comparison, write_report
from agent_run_ledger.core.storage import init_db, list_runs, load_bundle, save_bundle

app = typer.Typer(help="Agent Run Ledger CLI")
console = Console()


def default_db() -> Path:
    return Path(os.environ.get("ARL_DB", ".arl/ledger.sqlite"))


@app.command("init")
def init(db: Path = typer.Option(default_factory=default_db, help="SQLite database path.")) -> None:
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


@app.command("import")
def import_trace(
    path: Path = typer.Argument(..., help="Trace JSON file."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    bundle = load_trace(path)
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    run_id = save_bundle(db, bundle)
    console.print(f"imported run: {run_id}")


@app.command("export")
def export_trace(
    run: str = typer.Option(..., "--run", help="Run id."),
    out: Path = typer.Option(..., "--out", help="Output JSON path."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    bundle = _load_bundle_or_exit(db, run)
    write_trace(bundle, out)
    console.print(f"wrote trace: {out}")


@app.command("report")
def report(
    run: str = typer.Option(..., "--run", help="Run id."),
    out: Path | None = typer.Option(None, "--out", help="Output HTML path."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    bundle = _load_bundle_or_exit(db, run)
    output = out or Path(".arl") / "reports" / f"{run}.html"
    write_report(bundle, output)
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
        table.add_row(
            run.id,
            run.workflow,
            run.success_label,
            f"${run.total_cost_usd:.6f}",
            str(run.total_input_tokens + run.total_output_tokens),
        )
    console.print(table)


def _load_bundle_or_exit(db: Path, run_id: str):
    try:
        return load_bundle(db, run_id)
    except KeyError as exc:
        console.print(f"error: {exc.args[0]}")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
