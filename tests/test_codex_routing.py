"""Part (a) — JSONL import routing + defensive parsing of UNTRUSTED Codex logs.

A Codex rollout is a ``.jsonl`` file (line-delimited JSON), NOT a single
TraceBundle JSON object. ``arl import`` must DETECT the format and route a
``.jsonl`` / line-delimited file to the Codex adapter, while keeping the existing
single-object TraceBundle path for ``.json``. The format detection in
``io.py``/``cli.py`` stays provider-neutral (extension + line-delimited content),
never naming Codex fields.

The Codex log is UNTRUSTED input (Rule 8-adjacent). The JSONL loader reuses the
defensive guards: a hard size ceiling, a hard line-count ceiling, never
eval/execute log content, and a malformed line raises a typed error (never a
crash).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_run_ledger.adapters.codex import (
    MAX_ROLLOUT_BYTES,
    MAX_ROLLOUT_LINES,
    CodexRolloutError,
    load_codex_rollout,
)
from agent_run_ledger.cli import app

FIXTURES = Path(__file__).parent / "fixtures" / "codex"


# --------------------------------------------------------------------------- #
# CLI routing — the spec's RED-first: ``arl import <jsonl>`` produces a run
# --------------------------------------------------------------------------- #
def test_import_jsonl_routes_to_codex_adapter_and_produces_a_run(tmp_path: Path) -> None:
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"

    result = runner.invoke(
        app, ["import", str(FIXTURES / "fire_no_edit_retry.jsonl"), "--db", str(db)]
    )

    assert result.exit_code == 0, result.output
    assert "imported run:" in result.output


def test_import_jsonl_fire_case_stores_a_prescription(tmp_path: Path) -> None:
    """End-to-end through the CLI: the no-edit retry fixture imports and, on read,
    yields a run whose report carries the retry prescription (proves the routed
    bundle is analyzable, not just parsed)."""
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"
    runner.invoke(app, ["import", str(FIXTURES / "fire_no_edit_retry.jsonl"), "--db", str(db)])

    report = tmp_path / "r.html"
    # the run id is provider-stamped; list-runs then report the only run
    from agent_run_ledger.core.storage import list_runs

    runs = list_runs(db)
    assert len(runs) == 1
    result = runner.invoke(app, ["report", "--run", runs[0].id, "--out", str(report), "--db", str(db)])
    assert result.exit_code == 0
    assert "retry" in report.read_text(encoding="utf-8").lower()


def test_existing_json_trace_path_is_unchanged(tmp_path: Path) -> None:
    """A regular single-object .json TraceBundle still imports via the original
    path — the routing must not break the existing behavior."""
    runner = CliRunner()
    db = tmp_path / "ledger.sqlite"

    result = runner.invoke(app, ["import", "fixtures/golden_retry_loop.json", "--db", str(db)])

    assert result.exit_code == 0
    assert "imported run: run_retry_loop" in result.output


# --------------------------------------------------------------------------- #
# Defensive parsing — UNTRUSTED log, typed errors, hard ceilings
# --------------------------------------------------------------------------- #
def test_oversize_rollout_is_rejected_before_parse(tmp_path: Path) -> None:
    p = tmp_path / "big.jsonl"
    # one giant line over the byte cap
    p.write_text('{"x":"' + "A" * (MAX_ROLLOUT_BYTES + 1) + '"}\n', encoding="utf-8")

    with pytest.raises(CodexRolloutError, match="too large"):
        load_codex_rollout(p)


def test_too_many_lines_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "many.jsonl"
    p.write_text("{}\n" * (MAX_ROLLOUT_LINES + 1), encoding="utf-8")

    with pytest.raises(CodexRolloutError, match="lines|too many"):
        load_codex_rollout(p)


def test_malformed_line_errors_safely(tmp_path: Path) -> None:
    """A single malformed line raises a typed error, never an uncaught
    JSONDecodeError or a crash."""
    p = tmp_path / "bad.jsonl"
    p.write_text('{"type": "session_meta"}\n{not json\n', encoding="utf-8")

    with pytest.raises(CodexRolloutError):
        load_codex_rollout(p)


def test_non_object_line_errors_safely(tmp_path: Path) -> None:
    """A line that parses to a non-object (array/scalar) is not a rollout record ->
    typed error, no attribute-access crash."""
    p = tmp_path / "arr.jsonl"
    p.write_text('{"type":"session_meta"}\n[1,2,3]\n', encoding="utf-8")

    with pytest.raises(CodexRolloutError):
        load_codex_rollout(p)


def test_non_utf8_rollout_errors_safely(tmp_path: Path) -> None:
    p = tmp_path / "bin.jsonl"
    p.write_bytes(b'{"type":"session_meta"}\n\xff\xfe not utf-8\n')

    with pytest.raises(CodexRolloutError):
        load_codex_rollout(p)


def test_rollout_with_no_tool_calls_errors_safely(tmp_path: Path) -> None:
    """A rollout that contains zero tool calls cannot form a run with steps -> a
    typed error, never an empty/invalid bundle that crashes downstream."""
    p = tmp_path / "empty.jsonl"
    p.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "x"}}) + "\n", encoding="utf-8"
    )

    with pytest.raises(CodexRolloutError):
        from agent_run_ledger.adapters.codex import bundle_from_rollout

        bundle_from_rollout(load_codex_rollout(p))


def test_hostile_command_text_is_inert(tmp_path: Path) -> None:
    """A command/argument that looks like code injection is stored as inert data —
    never evaluated. The adapter only fingerprints + records facts; nothing in a
    rollout is ever executed."""
    p = tmp_path / "hostile.jsonl"
    recs = [
        {"type": "session_meta", "payload": {"id": "s1"}},
        {"type": "turn_context", "payload": {"model": "gpt-5.5"}},
        {
            "timestamp": "2026-05-29T12:00:02.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "$(__import__('os').system('echo pwned'))"}),
                "call_id": "c1",
            },
        },
        {
            "timestamp": "2026-05-29T12:00:03.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "Process exited with code 0\nOutput:\n[x]",
            },
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    from agent_run_ledger.adapters.codex import bundle_from_rollout

    bundle = bundle_from_rollout(load_codex_rollout(p))
    # one function step, no error (exit 0), and the hostile string never executed:
    # it survives only as a one-way fingerprint, not as stored raw content.
    fn = [s for s in bundle.steps if s.span_kind == "function"]
    assert len(fn) == 1
    assert fn[0].error is None
    assert fn[0].input_fingerprint is not None
