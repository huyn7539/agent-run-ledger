"""``arl serve`` — READ-ONLY loopback dashboard over the existing ledger (Task 59).

SECURITY CONTRACT (Codex spec review 2026-06-11, adopted in full):
  * Binds ``127.0.0.1`` ONLY — the literal in the ``HTTPServer`` constructor is
    AST-asserted by ``tests/test_egress_guards.py`` (no variables, no overrides
    of ``server_bind``/``address_family``).
  * GET/HEAD only; every other method answers 405. No mutating route exists.
  * Host-header validation: exactly ``127.0.0.1`` or ``127.0.0.1:<bound port>``,
    anything else 403 (DNS-rebinding defense — a hostile origin can point its
    own hostname at 127.0.0.1; the Host header betrays it).
  * No CORS header is EVER sent; ``X-Content-Type-Options: nosniff`` always;
    precise content types with charset.
  * Every ledger-derived string rendered here is ``html.escape``d — session
    logs are hostile by design. ``/run/<id>`` reuses ``render_report`` (its
    escaping is leak-matrix-tested); ``/verdicts`` is served as text/plain.
  * Storage access is STRICTLY read-only and per-request short-lived
    (``storage.connect_readonly``: URI mode=ro + ``PRAGMA query_only=ON``;
    never ``init_db``/mkdir/chmod — a GET must not write, Codex F8). A
    locked/busy DB answers 503; a missing DB answers an honest empty page/404.
  * ``/verdicts`` refuses a symlinked file or parent (404) and never follows
    one — the path is caller-controlled CWD state.
  * No reverse DNS: ``log_message``/``address_string`` overridden; access
    logging is suppressed entirely.
  * The egress guard allows ONLY ``http.server`` in THIS module; outbound
    primitives stay banned package-wide INCLUDING here (call-site scan in
    ``tests/test_egress_guards.py``).
"""

from __future__ import annotations

import html
import re
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent_run_ledger.core.receipt import build_receipts
from agent_run_ledger.core.report import render_report
from agent_run_ledger.core.storage import list_runs_readonly, load_bundle_readonly

# Run-id segment for /run/<id>: closed charset, no separators, no percent
# escapes (anything percent-encoded is rejected wholesale — stricter than
# decoding; encoded separators %2f/%5c can never smuggle through).
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_INDEX_RUN_CAP = 100  # most-recent cap; disclosed in the page when it truncates
_VERDICTS_TAIL_LINES = 50

_PAGE_TEMPLATE = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<meta http-equiv='refresh' content='5'>"
    "<title>arl serve</title>"
    "<style>body{{font-family:monospace;margin:2em}}table{{border-collapse:collapse}}"
    "td,th{{border:1px solid #999;padding:4px 8px;text-align:left}}</style>"
    "</head><body>{body}</body></html>"
)


class _ReadOnlyServer(HTTPServer):
    # Codex F17: stdlib HTTPServer sets allow_reuse_address=1; on Windows
    # SO_REUSEADDR lets another process bind the same port. Ephemeral-port
    # default makes reuse unnecessary — disable it.
    allow_reuse_address = False

    def __init__(self, db_path: Path, verdicts_path: Path, port: int) -> None:
        self.db_path = db_path
        self.verdicts_path = verdicts_path
        # The loopback literal below is load-bearing and AST-asserted; do not
        # replace it with a variable, constant reference, or config value.
        super().__init__(("127.0.0.1", port), _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: _ReadOnlyServer  # narrowed type for attribute access
    server_version = "arl-serve"
    sys_version = ""

    # -- no access log, no reverse DNS (Codex F10/F20) ----------------------
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def address_string(self) -> str:
        return self.client_address[0]

    # -- responses -----------------------------------------------------------
    def _send(self, code: int, body: str, content_type: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        # Codex F2: nosniff always; NEVER an Access-Control-Allow-Origin header.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_html(self, code: int, body: str) -> None:
        self._send(code, _PAGE_TEMPLATE.format(body=body), "text/html; charset=utf-8")

    def _send_text(self, code: int, body: str) -> None:
        self._send(code, body, "text/plain; charset=utf-8")

    # -- method gates ---------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        self._serve()

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve()

    def _reject_method(self) -> None:
        self._send_text(405, "405: read-only server (GET/HEAD only)\n")

    do_POST = _reject_method  # noqa: N815
    do_PUT = _reject_method  # noqa: N815
    do_DELETE = _reject_method  # noqa: N815
    do_PATCH = _reject_method  # noqa: N815
    do_OPTIONS = _reject_method  # noqa: N815

    # -- routing --------------------------------------------------------------
    def _host_ok(self) -> bool:
        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        return host in ("127.0.0.1", f"127.0.0.1:{port}")

    def _serve(self) -> None:
        if not self._host_ok():
            self._send_text(403, "403: bad Host header (loopback only)\n")
            return
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if "%" in path:
            # reject ALL percent-encoding rather than decode it (Codex F13)
            self._send_text(404, "404: not found\n")
            return
        try:
            if path == "/":
                self._index()
            elif path == "/verdicts":
                self._verdicts()
            elif path.startswith("/run/") and _RUN_ID_RE.match(path[5:]):
                self._run_page(path[5:])
            else:
                self._send_text(404, "404: not found\n")
        except sqlite3.OperationalError as exc:
            # busy/locked or unreadable: honest 503, never a write/retry (F9)
            self._send_text(503, f"503: ledger unavailable ({html.escape(str(exc))})\n")

    # -- pages ----------------------------------------------------------------
    def _index(self) -> None:
        db = self.server.db_path
        if not db.exists():
            self._send_html(200, f"<p>no ledger found at <code>{html.escape(str(db))}</code></p>")
            return
        runs = list_runs_readonly(db)
        shown = runs[:_INDEX_RUN_CAP]
        rows = []
        for run in shown:
            receipts = 0
            try:
                receipts = len(build_receipts(load_bundle_readonly(db, run.id)))
            except (KeyError, sqlite3.OperationalError, ValueError):
                pass  # badge is best-effort; the run row itself still renders
            badge = f"{receipts} receipt(s)" if receipts else "clean*"
            rows.append(
                "<tr>"
                f"<td><a href='/run/{html.escape(run.id, quote=True)}'>{html.escape(run.id)}</a></td>"
                f"<td>{html.escape(run.workflow)}</td>"
                f"<td>{html.escape(run.started_at)}</td>"
                f"<td>{html.escape(run.success_label)}</td>"
                f"<td>{html.escape(badge)}</td>"
                "</tr>"
            )
        truncated = (
            f"<p>showing {len(shown)} of {len(runs)} runs (most recent first)</p>"
            if len(runs) > len(shown)
            else ""
        )
        body = (
            f"<h1>arl ledger</h1><p><code>{html.escape(str(db))}</code> — read-only · "
            "<a href='/verdicts'>verdicts</a></p>"
            "<table><tr><th>run</th><th>workflow</th><th>started</th><th>label</th>"
            "<th>receipts</th></tr>" + "\n".join(rows) + "</table>" + truncated +
            "<p>clean* = clean for the checked classes only "
            "(see <code>arl verdict --json</code> coverage)</p>"
        )
        self._send_html(200, body)

    def _run_page(self, run_id: str) -> None:
        db = self.server.db_path
        if not db.exists():
            self._send_text(404, "404: no ledger\n")
            return
        try:
            bundle = load_bundle_readonly(db, run_id)
        except KeyError:
            self._send_text(404, "404: run not found\n")
            return
        # render_report escapes every bundle string (leak-matrix-tested).
        self._send(200, render_report(bundle), "text/html; charset=utf-8")

    def _verdicts(self) -> None:
        vp = self.server.verdicts_path
        # Codex F11: refuse symlinks/reparse points on the file AND its parent —
        # the path derives from caller-controlled CWD state.
        if vp.is_symlink() or vp.parent.is_symlink():
            self._send_text(404, "404: not found\n")
            return
        if not vp.exists():
            self._send_text(200, "no verdicts file ({})\n".format(vp.as_posix()))
            return
        try:
            lines = vp.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            self._send_text(503, "503: verdicts file unreadable\n")
            return
        tail = lines[-_VERDICTS_TAIL_LINES:]
        self._send_text(200, "\n".join(tail) + ("\n" if tail else ""))


def make_server(
    db_path: Path, port: int = 0, verdicts_path: Path | None = None
) -> _ReadOnlyServer:
    """Construct (and bind) the read-only loopback server. Port 0 = ephemeral;
    read the bound port from ``server.server_address[1]`` AFTER construction
    (Codex F15: never print a URL before bind)."""
    if verdicts_path is None:
        verdicts_path = Path(".arl") / "verdicts.jsonl"
    return _ReadOnlyServer(db_path, verdicts_path, port)
