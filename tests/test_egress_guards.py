"""L10/L11/L12/L13 — egress guards, file permissions, and the locked contracts.

L10: no module under src/agent_run_ledger may import a network-capable module
     (the safest possible state — makes a naive requests.post() impossible to
     merge under deadline without first building the fail-closed chokepoint).
     Folds in the provider-neutrality grep (no 'openai' in core/).
L11: the .arl dir is 0o700 and files 0o600 on POSIX; a one-time warning fires
     if .arl resolves inside a known cloud-sync path.
L12/L13: the ShapeEvent closed-type invariant + the one-sentence redaction
     contract are LOCKED as text (ADR + docstring), build deferred (D2).
TDD red-first (Task 44, Phase 4 — Rule 8 surface).
"""

from __future__ import annotations

import ast
import os
import stat
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "agent_run_ledger"

_NETWORK_MODULES = {
    "requests", "httpx", "urllib", "urllib2", "urllib3", "aiohttp",
    "socket", "http", "http.client", "ftplib", "telnetlib", "websocket",
    "websockets", "smtplib", "asyncio",
}


def _iter_py_files():
    return sorted(_SRC.rglob("*.py"))


def _imported_names(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module.split(".")[0], node.module


# --- L10: no network imports in src/ ------------------------------------------

def test_no_network_imports_in_src() -> None:
    offenders = []
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for top, full in _imported_names(tree):
            if top in _NETWORK_MODULES or full in _NETWORK_MODULES:
                offenders.append(f"{path.relative_to(_SRC.parent.parent)}: {full}")
    assert offenders == [], (
        "network-capable import found under src/ — all egress MUST route "
        f"through core/telemetry.emit() (fail-closed). Offenders: {offenders}"
    )


# --- L10 fold-in: provider neutrality — no 'openai' under core/ ----------------

def test_core_is_provider_neutral_no_openai() -> None:
    core = _SRC / "core"
    offenders = []
    for path in core.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        if "openai" in text:
            offenders.append(str(path.relative_to(_SRC.parent.parent)))
    assert offenders == [], f"'openai' must not appear under core/: {offenders}"


# --- L11: file permissions on the ledger directory + files --------------------

@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")
def test_ledger_dir_and_file_are_locked_down(tmp_path: Path) -> None:
    from agent_run_ledger.core.storage import init_db

    db = tmp_path / "vault" / "ledger.sqlite"
    init_db(db)

    dir_mode = stat.S_IMODE(os.stat(db.parent).st_mode)
    file_mode = stat.S_IMODE(os.stat(db).st_mode)
    assert dir_mode == 0o700, oct(dir_mode)
    assert file_mode == 0o600, oct(file_mode)


def test_cloud_sync_path_warns(tmp_path, capsys, monkeypatch) -> None:
    from agent_run_ledger.core import storage

    # simulate .arl resolving inside a cloud-sync dir
    fake_home = tmp_path / "home"
    dropbox_db = fake_home / "Dropbox" / ".arl" / "ledger.sqlite"
    warning = storage.cloud_sync_warning(dropbox_db)
    assert warning is not None
    assert "Dropbox" in warning or "cloud-sync" in warning.lower()

    safe_db = tmp_path / "proj" / ".arl" / "ledger.sqlite"
    assert storage.cloud_sync_warning(safe_db) is None


# --- L12 / L13: the locked contracts exist as text ----------------------------

def test_adr_stub_locks_telemetry_chokepoint_and_shapeevent() -> None:
    adr = _SRC.parent.parent / ".agentbus" / "ADR-002-egress-chokepoint.md"
    assert adr.exists(), "L10/L12 ADR stub missing"
    text = adr.read_text(encoding="utf-8")
    assert "core/telemetry.emit()" in text  # L10 chokepoint
    assert "fail" in text.lower() and "consent" in text.lower()  # fail-closed on consent
    assert "ShapeEvent" in text  # L12 closed-type invariant
    assert "TraceBundle" in text  # L12: never redact-then-ship a bundle


def test_redaction_contract_sentence_in_schema_docstring() -> None:
    from agent_run_ledger.core import models

    src = Path(models.__file__).read_text(encoding="utf-8")
    # L13: one sentence near the outcome slot + provenance_hash
    assert "sensitivity tag" in src
    assert "redaction" in src.lower() and "consent" in src.lower()
