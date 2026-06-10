"""Legacy-console encoding safety (full-suite-audit P0, 2026-06-11).

The CLEAN verdict path crashed with UnicodeEncodeError + exit 1 on a default
Windows console (cp1252/cp437, PYTHONUTF8 unset) because of a non-encodable glyph
in a status line — on the product's MOST-COMMON outcome, mis-reported to a gating
loop as "unreadable". The 442-test suite never saw it: CliRunner captures through
a UTF-8 buffer, structurally blind to the real terminal code page.

These tests close that blindspot two ways:
  1. a fast unit guard — no human-output string contains a glyph that cp1252
     cannot encode (the exact crash condition);
  2. a real subprocess test under a forced cp1252 stdout, exercising the clean
     AND fired verdict paths, asserting the true exit code (0 / 3) with no
     traceback — the only configuration that reproduces real first-user reality.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "agent_run_ledger"


def _human_output_strings(text: str) -> list[str]:
    """Console.print / typer.echo string literals — the human-output surface."""
    out = []
    for m in re.finditer(r'(?:console\.print|typer\.echo)\(\s*(["\'])(.*?)\1', text, re.S):
        out.append(m.group(2))
    # f-strings too (the quoted body still carries the literal glyphs)
    for m in re.finditer(r'(?:console\.print|typer\.echo)\(\s*f(["\'])(.*?)\1', text, re.S):
        out.append(m.group(2))
    return out


def test_no_human_output_glyph_crashes_cp1252() -> None:
    """The exact P0 crash condition: a human-output literal containing a glyph
    cp1252 cannot encode. cli.py is the only module that prints to the user."""
    cli = (SRC / "cli.py").read_text(encoding="utf-8")
    offenders = []
    for s in _human_output_strings(cli):
        for ch in s:
            if ord(ch) > 0x7F:
                try:
                    ch.encode("cp1252")
                except UnicodeEncodeError:
                    offenders.append((hex(ord(ch)), ch, s[:60]))
    assert not offenders, (
        "human-output strings contain cp1252-unencodable glyphs (would crash a "
        f"default Windows console): {offenders}"
    )


def _run(args: list[str], env_extra: dict) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, "-m", "agent_run_ledger", *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
    )


def test_clean_verdict_survives_cp1252_stdout(tmp_path: Path) -> None:
    """The real-terminal repro: force a cp1252 stdout and run the CLEAN verdict
    path as a subprocess. Must exit 0 with no traceback (was: UnicodeEncodeError
    + exit 1, the loop-contract 'unreadable' code, on a perfectly clean run)."""
    proc = _run(
        ["verdict", "fixtures/clean_run.json", "--db", str(tmp_path / "l.sqlite")],
        {"PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"},
    )
    assert proc.returncode == 0, (
        f"clean verdict crashed under cp1252 (exit {proc.returncode}): "
        f"{proc.stderr.decode('cp1252', 'replace')}"
    )
    assert b"Traceback" not in proc.stderr
    assert b"UnicodeEncodeError" not in proc.stderr


def test_fired_verdict_survives_cp1252_stdout(tmp_path: Path) -> None:
    proc = _run(
        ["verdict", "fixtures/golden_retry_loop.json", "--db", str(tmp_path / "l.sqlite")],
        {"PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"},
    )
    assert proc.returncode == 3, (
        f"fired verdict wrong exit under cp1252 ({proc.returncode}): "
        f"{proc.stderr.decode('cp1252', 'replace')}"
    )
    assert b"Traceback" not in proc.stderr


def test_selftest_survives_cp1252_stdout() -> None:
    proc = _run(["selftest"], {"PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"})
    assert proc.returncode == 0, proc.stderr.decode("cp1252", "replace")
    assert b"Traceback" not in proc.stderr
