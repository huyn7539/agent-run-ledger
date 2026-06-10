"""Release-surface version checks (pre-first-user sweep, 2026-06-11).

A first user reporting a misfire needs `arl --version` to tell us what they
ran — without it every bug report starts with archaeology. And the version it
prints must be THE version: `__version__` drifted to 0.1.0 while pyproject said
0.2.0, so anything reading the package attribute reported a release that the
wheel was not.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from typer.testing import CliRunner

from agent_run_ledger import __version__
from agent_run_ledger.cli import app


def test_dunder_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert __version__ == declared


def test_cli_version_flag_prints_version_and_exits_zero() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output
    assert "agent-run-ledger" in result.output
