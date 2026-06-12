"""Task 59 — `arl serve` read-only loopback dashboard.

What is pinned here (Codex spec-review amendments, all adopted):
  * read-only: GETs never create/migrate/chmod the DB; POST/PUT/DELETE → 405
  * Host-header validation (DNS-rebinding defense): non-loopback Host → 403
  * headers: X-Content-Type-Options: nosniff, NO Access-Control-Allow-Origin
  * XSS: hostile ledger strings render escaped on every dashboard surface
  * routing: exact match only; traversal and percent-encoding rejected
  * /verdicts symlink refusal; missing-DB and missing-run honesty
  * storage: connect_readonly cannot write and never creates files
  * CLI: default port is FIXED (8765) and documented; a busy port fails CLOSED
    with direction (the ephemeral default produced servers nobody could find
    on release night — operator, reviewing agent, and browser all assumed a
    stable port)
"""

from __future__ import annotations

import http.client
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from agent_run_ledger.core.models import (
    PrescriptionRecord,
    RunRecord,
    StepRecord,
    TraceBundle,
)
from agent_run_ledger.core.serve import make_server
from agent_run_ledger.core.storage import (
    connect_readonly,
    load_bundle_readonly,
    save_bundle,
)

XSS = "<script>alert(1)</script>"


def _bundle(run_id: str = "run_serve") -> TraceBundle:
    run = RunRecord(
        id=run_id,
        workflow=f"wf-{XSS}",
        framework="neutral",
        provider="openai",
        model="gpt-4o-mini",
        started_at="2026-06-11T00:00:00Z",
        ended_at="2026-06-11T00:00:10Z",
        success_label="failed",
    )
    steps = [
        StepRecord(
            id=f"s{i}",
            run_id=run_id,
            step_type="function",
            name=f"crm.lookup-{XSS}",
            started_at=f"2026-06-11T00:00:0{i}Z",
            ended_at=f"2026-06-11T00:00:0{i}Z",
            parent_step_id=f"turn_{i}",
            span_kind="function",
            retry_scope="agent_root",
            input_fingerprint="fp",
            error="Error running tool",
            error_class="Other",
        )
        for i in range(1, 4)
    ]
    rx = PrescriptionRecord(
        id="rx1",
        run_id=run_id,
        severity="high",
        root_cause="retry loop",
        one_line_fix="Set crm.lookup retry budget and fail closed.",
        evidence=["step_id=s1", "retry_count=2 additional attempts"],
        patch_type="config_diff",
        patch=(
            "--- a/agent/config.yaml\n+++ b/agent/config.yaml\n@@ -1 +1 @@\n"
            "-crm_lookup_retries: 5\n+crm_lookup_retries: 0\n"
        ),
        expected_impact={},
        regression_test_template="",
    )
    return TraceBundle(run=run, steps=steps, prescriptions=[rx])


@pytest.fixture()
def served(tmp_path: Path):
    """A live server over a real tmp ledger; yields (host, port, db_path)."""
    db = tmp_path / "dir with space" / "ledger.sqlite"
    db.parent.mkdir(parents=True)
    save_bundle(db, _bundle())
    server = make_server(db, verdicts_path=tmp_path / ".arl" / "verdicts.jsonl")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", server.server_address[1], db
    finally:
        server.shutdown()
        server.server_close()


def _get(host: str, port: int, path: str, *, method: str = "GET", host_header: str | None = None):
    conn = http.client.HTTPConnection(host, port, timeout=10)
    headers = {}
    if host_header is not None:
        headers["Host"] = host_header
    conn.request(method, path, headers=headers)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", errors="replace")
    result = (resp.status, dict(resp.getheaders()), body)
    conn.close()
    return result


def test_index_lists_runs_escaped(served) -> None:
    host, port, _ = served
    status, headers, body = _get(host, port, "/")
    assert status == 200
    assert "run_serve" in body
    assert XSS not in body  # hostile workflow name must be escaped
    assert "&lt;script&gt;" in body
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert "Access-Control-Allow-Origin" not in headers


def test_run_page_renders_report_escaped(served) -> None:
    host, port, _ = served
    status, _, body = _get(host, port, "/run/run_serve")
    assert status == 200
    assert XSS not in body  # render_report escapes step names
    assert "receipt" in body.lower() or "prescription" in body.lower()


def test_mutating_methods_are_405(served) -> None:
    host, port, _ = served
    for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS"):
        status, _, _ = _get(host, port, "/", method=method)
        assert status == 405, method


def test_bad_host_header_is_403(served) -> None:
    host, port, _ = served
    status, _, _ = _get(host, port, "/", host_header="evil.example:80")
    assert status == 403
    # exact loopback host WITH the bound port is accepted
    status, _, _ = _get(host, port, "/", host_header=f"127.0.0.1:{port}")
    assert status == 200


def test_unknown_traversal_and_percent_paths_are_404(served) -> None:
    host, port, _ = served
    for path in ("/nope", "/run/../secrets", "/run/%2e%2e", "/run/a/b", "/run/"):
        status, _, _ = _get(host, port, path)
        assert status == 404, path


def test_missing_run_is_404(served) -> None:
    host, port, _ = served
    status, _, _ = _get(host, port, "/run/does_not_exist")
    assert status == 404


def test_verdicts_absent_file_is_honest_200(served) -> None:
    host, port, _ = served
    status, headers, body = _get(host, port, "/verdicts")
    assert status == 200
    assert headers.get("Content-Type", "").startswith("text/plain")
    assert "no verdicts file" in body


def test_get_does_not_create_or_migrate_the_db(tmp_path: Path) -> None:
    """THE Codex F8 pin: serving against a missing DB must not create it
    (load_bundle's init_db path runs DDL + chmod on read — serve must never)."""
    db = tmp_path / "absent" / "ledger.sqlite"
    server = make_server(db, verdicts_path=tmp_path / "v.jsonl")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        status, _, body = _get("127.0.0.1", port, "/")
        assert status == 200
        assert "no ledger found" in body
        status, _, _ = _get("127.0.0.1", port, "/run/x")
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()
    assert not db.exists()
    assert not db.parent.exists()


def test_connect_readonly_cannot_write_and_never_creates(tmp_path: Path) -> None:
    missing = tmp_path / "nope.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        connect_readonly(missing)
    assert not missing.exists()

    db = tmp_path / "dir with space" / "ledger.sqlite"
    db.parent.mkdir(parents=True)
    save_bundle(db, _bundle("run_ro"))
    conn = connect_readonly(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO runs (id) VALUES ('evil')")
        assert load_bundle_readonly(db, "run_ro").run.id == "run_ro"
    finally:
        conn.close()


def test_verdicts_symlink_is_refused(served, tmp_path: Path) -> None:
    host, port, db = served
    target = tmp_path / "real.jsonl"
    target.write_text('{"v":1}\n', encoding="utf-8")
    link_dir = tmp_path / ".arl"
    link_dir.mkdir(exist_ok=True)
    link = link_dir / "verdicts.jsonl"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlinks unavailable (Windows without Developer Mode)")
    status, _, body = _get(host, port, "/verdicts")
    assert status == 404
    assert '"v":1' not in body


# --- CLI port contract: fixed default, fail-closed on busy port ---------------


def test_cli_serve_default_port_is_fixed_and_documented() -> None:
    from typer.testing import CliRunner

    from agent_run_ledger.cli import app

    result = CliRunner().invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "8765" in result.output  # the documented default a browser can find


def test_cli_serve_busy_port_fails_closed_with_direction(tmp_path: Path) -> None:
    """A busy port must produce exit 1 and an actionable message, never a
    traceback and never a silent fallback to a different port."""
    import socket

    from typer.testing import CliRunner

    from agent_run_ledger.cli import app

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        busy_port = blocker.getsockname()[1]
        result = CliRunner().invoke(
            app, ["serve", "--port", str(busy_port), "--db", str(tmp_path / "l.sqlite")]
        )
        assert result.exit_code == 1
        assert "cannot bind" in result.output
        assert "--port 0" in result.output  # the escape hatch is named
    finally:
        blocker.close()
