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
#
# Task 59 amendment (Codex spec review, adopted): EXACTLY ONE module —
# core/serve.py — may import EXACTLY ONE network module — http.server (the
# INBOUND stdlib server). The allow is (file, full-module-name) keyed; http
# anything-else (http.client!) stays banned everywhere including serve.py.

_SERVE_ALLOW = {("core/serve.py", "http.server")}


def _serve_rel(path: Path) -> str:
    return path.relative_to(_SRC).as_posix()


def test_no_network_imports_in_src() -> None:
    offenders = []
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for top, full in _imported_names(tree):
            if (_serve_rel(path), full) in _SERVE_ALLOW:
                continue
            if top in _NETWORK_MODULES or full in _NETWORK_MODULES:
                offenders.append(f"{path.relative_to(_SRC.parent.parent)}: {full}")
    assert offenders == [], (
        "network-capable import found under src/ — all egress MUST route "
        f"through core/telemetry.emit() (fail-closed). Offenders: {offenders}"
    )


def test_serve_allowlist_is_exactly_http_server() -> None:
    """The allowlist itself is pinned closed: one file, one inbound module."""
    assert _SERVE_ALLOW == {("core/serve.py", "http.server")}


# --- Task 59: bind predicate — the HTTPServer constructor's literal address ----

def _bind_violations(src_text: str) -> list[str]:
    """AST scan: every call to *HTTPServer-named* constructors must pass a
    literal ('127.0.0.1', ...) tuple as the address. Variables, other literals
    (0.0.0.0, ::, localhost), and missing tuples are violations. Also flags
    server_bind / address_family overrides (rebinding escape hatches)."""
    tree = ast.parse(src_text)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "server_bind":
            violations.append("server_bind override")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id == "address_family":
                    violations.append("address_family override")
                if isinstance(t, ast.Attribute) and t.attr == "address_family":
                    violations.append("address_family override")
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if not name.endswith("HTTPServer"):
                continue
            # superclass __init__ delegation: HTTPServer.__init__/super().__init__
            args = node.args
            if not args:
                violations.append(f"{name}: no address argument")
                continue
            addr = args[0]
            if not (
                isinstance(addr, ast.Tuple)
                and addr.elts
                and isinstance(addr.elts[0], ast.Constant)
                and addr.elts[0].value == "127.0.0.1"
            ):
                violations.append(f"{name}: address is not the literal ('127.0.0.1', ...)")
    return violations


def _super_init_bind_violations(src_text: str) -> list[str]:
    """The serve module binds via super().__init__((...)); scan those too."""
    tree = ast.parse(src_text)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "__init__":
            continue
        if not node.args:
            continue
        addr = node.args[0]
        if isinstance(addr, ast.Tuple):
            if not (
                addr.elts
                and isinstance(addr.elts[0], ast.Constant)
                and addr.elts[0].value == "127.0.0.1"
            ):
                violations.append("__init__ address tuple is not ('127.0.0.1', ...)")
    return violations


def test_serve_binds_loopback_literal_only() -> None:
    serve = _SRC / "core" / "serve.py"
    assert serve.exists(), "Task 59 P1 locks the guard BEFORE the server ships"
    src = serve.read_text(encoding="utf-8")
    assert _bind_violations(src) == []
    assert _super_init_bind_violations(src) == []
    # at least one loopback bind actually exists (the predicate is not vacuous)
    assert "127.0.0.1" in src


def test_bind_predicate_bites_on_violation_fixtures() -> None:
    """The lock must BITE (vacuous-lock class, FAILURE-INDEX Category 4)."""
    assert _bind_violations("HTTPServer(('0.0.0.0', 0), H)") != []
    assert _bind_violations("host='127.0.0.1'\nHTTPServer((host, 0), H)") != []
    assert _bind_violations("ThreadingHTTPServer(('::1', 0), H)") != []
    assert _bind_violations("class S:\n    def server_bind(self): pass") != []
    assert _bind_violations("S.address_family = 23") != []
    assert _super_init_bind_violations("super().__init__(('0.0.0.0', port), H)") != []
    # and the clean shapes pass
    assert _bind_violations("HTTPServer(('127.0.0.1', 0), H)") == []
    assert _super_init_bind_violations("super().__init__(('127.0.0.1', port), H)") == []


# --- Task 59: call-site bans — dynamic import / exec / spawn / outbound prims --

_BANNED_CALL_NAMES = {"__import__", "eval", "exec"}
_BANNED_ATTR_CALLS = {
    "import_module",        # importlib dynamic import
    "system",               # os.system
    "create_connection",    # socket outbound
    "sendto",
    "sendmsg",
    "open_connection",      # asyncio outbound
    "urlopen",              # urllib via any indirection
    "getaddrinfo",          # DNS
}
_SPAWN_MODULES = {"subprocess", "ctypes"}


def _dynamic_violations(src_text: str) -> list[str]:
    tree = ast.parse(src_text)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _BANNED_CALL_NAMES:
                out.append(f.id)
            if isinstance(f, ast.Attribute) and f.attr in _BANNED_ATTR_CALLS:
                out.append(f.attr)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _SPAWN_MODULES:
                    out.append(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in _SPAWN_MODULES:
                out.append(node.module)
    return out


def test_no_dynamic_import_exec_spawn_or_outbound_callsites_in_src() -> None:
    """Closes the Codex F5/F7 evasion set: the import ban alone misses
    __import__/importlib/eval/exec/subprocess/ctypes and outbound primitives
    reached through attribute calls. With dynamic import ALSO banned, a banned
    module cannot be smuggled in at runtime, so the attr-name set need not
    chase every alias. Applies to the whole package INCLUDING core/serve.py
    (the allowlisted module is the natural place to hide outbound code —
    Codex F19)."""
    offenders = []
    for path in _iter_py_files():
        for hit in _dynamic_violations(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(_SRC.parent.parent)}: {hit}")
    assert offenders == [], f"banned call-site/spawn-import found: {offenders}"


def test_dynamic_scan_bites_on_violation_fixtures() -> None:
    assert _dynamic_violations("__import__('socket')") != []
    assert _dynamic_violations("import importlib\nimportlib.import_module('x')") != []
    assert _dynamic_violations("eval('1')") != []
    assert _dynamic_violations("import subprocess") != []
    assert _dynamic_violations("import ctypes") != []
    assert _dynamic_violations("os.system('curl x')") != []
    assert _dynamic_violations("s.sendto(b'x', addr)") != []
    assert _dynamic_violations("x = 1 + 1") == []


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
    adr = _SRC.parent.parent / "docs" / "adr" / "ADR-002-egress-chokepoint.md"
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
