from pathlib import Path
import json
import os

from typer.testing import CliRunner

from agent_run_ledger.cli import app
from agent_run_ledger.core.io import write_trace
from agent_run_ledger.core.models import TraceValidationError


def test_cli_demo_report_compare(tmp_path: Path) -> None:
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"
    report_path = tmp_path / "report.html"

    result = runner.invoke(app, ["init", "--db", str(db)])
    assert result.exit_code == 0

    result = runner.invoke(app, ["run-demo", "--variant", "retry-loop", "--db", str(db)])
    assert result.exit_code == 0
    assert "run_retry_loop" in result.output

    result = runner.invoke(app, ["run-demo", "--variant", "clean", "--db", str(db)])
    assert result.exit_code == 0
    assert "run_clean" in result.output

    result = runner.invoke(
        app,
        ["report", "--run", "run_retry_loop", "--out", str(report_path), "--db", str(db)],
    )
    assert result.exit_code == 0
    assert "wrote report" in result.output
    assert "Non-runnable config diff" in report_path.read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        ["compare", "--left", "run_retry_loop", "--right", "run_clean", "--db", str(db)],
    )
    assert result.exit_code == 0
    assert "failed -> passed" in result.output


def test_cli_import_external_json(tmp_path: Path) -> None:
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"

    result = runner.invoke(
        app,
        ["import", "fixtures/golden_retry_loop.json", "--db", str(db)],
    )

    assert result.exit_code == 0
    assert "imported run: run_retry_loop" in result.output


def test_cli_missing_run_fails_closed(tmp_path: Path) -> None:
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"

    result = runner.invoke(app, ["report", "--run", "missing", "--db", str(db)])

    assert result.exit_code != 0
    assert "run not found: missing" in result.output


def test_cli_import_malformed_json_clean_error(tmp_path: Path) -> None:
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")

    result = runner.invoke(app, ["import", str(bad_json), "--db", str(db)])

    assert result.exit_code == 1
    assert "error:" in result.output
    assert not isinstance(result.exception, json.JSONDecodeError)


def test_cli_import_missing_file_clean_error(tmp_path: Path) -> None:
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"

    result = runner.invoke(app, ["import", str(tmp_path / "missing.json"), "--db", str(db)])

    assert result.exit_code == 1
    assert "error:" in result.output
    assert not isinstance(result.exception, FileNotFoundError)


def test_cli_import_bad_schema_clean_error(tmp_path: Path) -> None:
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"
    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text('{"schema_version": "0.1", "steps": []}', encoding="utf-8")

    result = runner.invoke(app, ["import", str(bad_schema), "--db", str(db)])

    assert result.exit_code == 1
    assert "error:" in result.output
    assert not isinstance(result.exception, TraceValidationError)


def test_cli_run_demo_bad_variant(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["run-demo", "--variant", "missing", "--db", str(tmp_path / "db.sqlite")])

    assert result.exit_code != 0


def test_cli_export_happy_roundtrip(tmp_path: Path) -> None:
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"
    out = tmp_path / "trace.json"
    runner.invoke(app, ["run-demo", "--variant", "clean", "--db", str(db)])

    result = runner.invoke(app, ["export", "--run", "run_clean", "--out", str(out), "--db", str(db)])

    assert result.exit_code == 0
    assert "wrote trace" in result.output
    assert json.loads(out.read_text(encoding="utf-8"))["run"]["id"] == "run_clean"


def test_cli_export_missing_run_clean_error(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["export", "--run", "missing", "--out", str(tmp_path / "out.json"), "--db", str(tmp_path / "db.sqlite")],
    )

    assert result.exit_code == 1
    assert "error: run not found: missing" in result.output


def test_cli_list_runs_empty_and_populated(tmp_path: Path) -> None:
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"

    empty = runner.invoke(app, ["list-runs", "--db", str(db)])
    runner.invoke(app, ["run-demo", "--variant", "clean", "--db", str(db)])
    populated = runner.invoke(app, ["list-runs", "--db", str(db)])

    assert empty.exit_code == 0
    assert populated.exit_code == 0
    assert "run_clean" in populated.output


def test_cli_import_non_demo_shape(tmp_path: Path, non_demo_bundle) -> None:
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"
    trace = tmp_path / "non_demo.json"
    write_trace(non_demo_bundle, trace)

    result = runner.invoke(app, ["import", str(trace), "--db", str(db)])

    assert result.exit_code == 0
    assert "imported run: run_invoice_audit" in result.output


def test_cli_arl_db_env_override(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    db = tmp_path / "env.sqlite"
    monkeypatch.setenv("ARL_DB", os.fspath(db))

    result = runner.invoke(app, ["run-demo", "--variant", "clean"])

    assert result.exit_code == 0
    assert db.exists()
