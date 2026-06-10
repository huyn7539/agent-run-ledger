"""Constraint 4 — defensive parsing of UNTRUSTED trace files.

ARL imports trace JSON authored elsewhere (exported from another tool, pasted by
a user). The parser must treat every byte as hostile: bounded size, bounded
nesting depth, never eval/execute, and a malformed/adversarial trace must error
SAFELY (a typed error), never crash the process or exploit the parser.

These tests extend the leak-matrix mindset to the parse boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_run_ledger.core.io import (
    MAX_TRACE_BYTES,
    MAX_TRACE_DEPTH,
    TraceParseError,
    load_trace,
)


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "trace.json"
    p.write_text(text, encoding="utf-8")
    return p


def test_oversize_trace_is_rejected_before_parse(tmp_path: Path) -> None:
    """A trace larger than the cap is rejected WITHOUT loading it into a JSON
    object (the size check reads the file size, not the parsed tree)."""
    big = _write(tmp_path, '{"x":"' + "A" * (MAX_TRACE_BYTES + 1) + '"}')

    with pytest.raises(TraceParseError, match="too large"):
        load_trace(big)


def test_deeply_nested_trace_is_rejected(tmp_path: Path) -> None:
    """A pathologically nested structure (billion-laughs / stack-blowing depth) is
    rejected with a typed error, not a RecursionError crash."""
    depth = MAX_TRACE_DEPTH + 50
    payload = "[" * depth + "]" * depth
    nested = _write(tmp_path, payload)

    with pytest.raises(TraceParseError, match="nesting|depth"):
        load_trace(nested)


def test_malformed_json_errors_safely(tmp_path: Path) -> None:
    bad = _write(tmp_path, '{"trace": {"trace_id": "x",,,}')

    with pytest.raises(TraceParseError):
        load_trace(bad)


def test_non_object_top_level_errors_safely(tmp_path: Path) -> None:
    """A top-level array / scalar is not a trace bundle -> typed error, no crash
    on attribute access."""
    arr = _write(tmp_path, "[1, 2, 3]")

    with pytest.raises(TraceParseError):
        load_trace(arr)


def test_trace_fields_are_treated_as_inert_strings(tmp_path: Path) -> None:
    """A field that LOOKS like code/template injection is stored/handled as inert
    string data — never evaluated. Here a workflow name containing a format-string
    and a path-traversal must not be executed or resolved."""
    hostile = {
        "run": {
            "id": "trace_hostile_0123",
            "workflow": "{__import__('os').system('echo pwned')}",
            "framework": "synthetic",
            "provider": "synthetic",
            "model": "demo-model",
            "started_at": "2026-05-31T10:00:00Z",
            "ended_at": "2026-05-31T10:00:01Z",
            "success_label": "failed",
        },
        "steps": [
            {
                "id": "../../etc/passwd",
                "type": "tool",
                "name": "x",
                "started_at": "2026-05-31T10:00:00Z",
                "ended_at": "2026-05-31T10:00:01Z",
                "retry_count": 0,
                "cost_usd": 0,
            }
        ],
    }
    p = _write(tmp_path, json.dumps(hostile))

    bundle = load_trace(p)

    # the hostile strings round-trip as inert data, unexecuted/unresolved
    assert bundle.run.workflow == "{__import__('os').system('echo pwned')}"
    assert bundle.steps[0].id == "../../etc/passwd"


def test_non_utf8_trace_errors_safely(tmp_path: Path) -> None:
    """A trace file with invalid UTF-8 must raise the typed TraceParseError, never
    an uncaught UnicodeDecodeError (the 'always a typed, caught error' contract)."""
    p = tmp_path / "trace.json"
    p.write_bytes(b'{"run": "\xff\xfe not utf-8"}')

    with pytest.raises(TraceParseError):
        load_trace(p)


def test_valid_trace_still_loads(tmp_path: Path) -> None:
    """The hardening must not break the happy path."""
    good = {
        "run": {"id": "t", "workflow": "w", "framework": "synthetic", "provider": "synthetic", "model": "demo-model", "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:01Z", "success_label": "passed"},
        "steps": [{"id": "s1", "type": "tool", "name": "x", "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:01Z", "retry_count": 0, "cost_usd": 0}],
    }
    p = _write(tmp_path, json.dumps(good))

    bundle = load_trace(p)

    assert bundle.run.id == "t"


# --- JSONL adapter depth guard (cold-review finding, 2026-06-11) -----------------
# The single-object path bounds nesting BEFORE json.loads; the two JSONL adapters
# parsed each line unguarded, so one pathologically nested line could stress the
# C stack. Both loaders must reject a depth-bomb line with their typed error.


def test_codex_loader_rejects_depth_bomb_line(tmp_path: Path) -> None:
    from agent_run_ledger.adapters.codex import CodexRolloutError, load_codex_rollout

    bomb = "[" * 5000 + "1" + "]" * 5000
    p = tmp_path / "rollout-2026-06-11T00-00-00-x.jsonl"
    p.write_text('{"type":"session_meta"}\n{"payload":' + bomb + "}\n", encoding="utf-8")
    with pytest.raises(CodexRolloutError, match="depth"):
        load_codex_rollout(p)


def test_claude_loader_rejects_depth_bomb_line(tmp_path: Path) -> None:
    from agent_run_ledger.adapters.claude_code import (
        ClaudeCodeSessionError,
        load_claude_session,
    )

    bomb = "[" * 5000 + "1" + "]" * 5000
    p = tmp_path / "session.jsonl"
    p.write_text('{"sessionId":"s","uuid":"u","type":"user"}\n{"x":' + bomb + "}\n", encoding="utf-8")
    with pytest.raises(ClaudeCodeSessionError, match="depth"):
        load_claude_session(p)
