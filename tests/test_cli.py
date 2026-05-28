from pathlib import Path

from typer.testing import CliRunner

from agent_run_ledger.cli import app


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
