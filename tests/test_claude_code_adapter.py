"""Claude Code session-log adapter (Task 57 P4).

Mirrors the Codex adapter's hardened semantics — turn synthesis, user-boundary
segments, first-write-wins results, incomplete-call honesty — against the
Claude Code line shape (top-level sessionId/uuid/timestamp; tool_use blocks in
assistant messages; tool_result blocks with is_error in user messages; usage
per message.id).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_run_ledger.adapters.claude_code import (
    ClaudeCodeSessionError,
    bundle_from_session,
    find_recent_sessions,
    load_claude_session,
    looks_like_claude_session_file,
)
from agent_run_ledger.cli import app
from agent_run_ledger.core.prescriptions import analyze_bundle

FIXTURES = Path(__file__).parent / "fixtures" / "claude_code"
CODEX_FIXTURES = Path(__file__).parent / "fixtures" / "codex"


def _bundle(name: str):
    return bundle_from_session(load_claude_session(FIXTURES / name))


# ------------------------------------------------------------------ mapping


def test_session_maps_to_run_with_function_steps() -> None:
    bundle = _bundle("retry_session.jsonl")
    assert bundle.run.id == "cc_f00dfeed-0000-4000-8000-000000000001"
    assert bundle.run.framework == "claude-code"
    assert bundle.run.provider == "anthropic"
    assert bundle.run.model == "claude-sonnet-4-6"
    assert bundle.run.workflow == "sample"
    assert len(bundle.steps) == 3
    assert all(s.span_kind == "function" for s in bundle.steps)
    assert all(s.name == "Bash" for s in bundle.steps)
    # provenance is stamped locally over the raw steps
    assert bundle.run.provenance_hash


def test_error_results_mark_steps_failed_with_content_free_marker() -> None:
    """is_error=true results mark the step failed; the models layer then redacts
    the marker itself (metadata_only mode) — what must hold is has_error=True and
    zero stored content from the result."""
    bundle = _bundle("retry_session.jsonl")
    for step in bundle.steps:
        assert step.error is not None
        assert step.error_class is not None
        # the result's text must never be stored anywhere on the step
        assert "FAILED" not in (step.error or "")
        assert "FAILED" not in json.dumps(step.metadata)


def test_identical_inputs_share_a_fingerprint() -> None:
    bundle = _bundle("retry_session.jsonl")
    prints = {s.input_fingerprint for s in bundle.steps}
    assert len(prints) == 1
    assert None not in prints


def test_token_totals_dedupe_by_message_id_not_per_line() -> None:
    """A streaming response repeats usage across lines under one message.id —
    only the LAST usage per message id may count."""
    base = json.loads((FIXTURES / "clean_session.jsonl").read_text().splitlines()[1])
    partial = json.loads(json.dumps(base))
    partial["uuid"] = "a-101-partial"
    partial["message"]["usage"] = {"input_tokens": 10, "output_tokens": 1}
    final = json.loads(json.dumps(base))
    final["uuid"] = "a-101-final"
    final["message"]["usage"] = {"input_tokens": 10, "output_tokens": 22}
    result_line = json.loads((FIXTURES / "clean_session.jsonl").read_text().splitlines()[2])
    bundle = bundle_from_session([partial, final, result_line])
    assert bundle.run.total_input_tokens == 10  # not 20
    assert bundle.run.total_output_tokens == 22  # last write wins, not 23


def test_retry_loop_fires_on_autonomous_repeats() -> None:
    bundle = _bundle("retry_session.jsonl")
    prescriptions = analyze_bundle(bundle)
    assert prescriptions, "3 identical failing autonomous attempts must fire the detector"


def test_clean_session_emits_no_prescriptions() -> None:
    bundle = _bundle("clean_session.jsonl")
    assert analyze_bundle(bundle) == []


def test_user_instruction_between_repeats_blocks_false_retry() -> None:
    """A human instruction between identical failing calls = a DIRECTED rerun,
    not an autonomous loop — the segment boundary must prevent collapse."""
    lines = [json.loads(ln) for ln in (FIXTURES / "retry_session.jsonl").read_text().splitlines()]
    instruction = {
        "parentUuid": "x",
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": "try running it again please"},
        "uuid": "u-extra",
        "timestamp": "2026-06-10T01:00:22.000Z",
        "cwd": "C:\\proj\\sample",
        "sessionId": "f00dfeed-0000-4000-8000-000000000001",
        "version": "2.1.170",
        "gitBranch": "main",
    }
    # insert a human instruction between EVERY attempt -> every rerun is directed
    with_boundaries = (
        lines[:3] + [dict(instruction, uuid="u-x1")] + lines[3:5]
        + [dict(instruction, uuid="u-x2")] + lines[5:]
    )
    bundle = bundle_from_session(with_boundaries)
    scopes = {s.retry_scope for s in bundle.steps}
    assert len(scopes) == 3, "each directed attempt must land in its own segment scope"
    assert analyze_bundle(bundle) == [], "directed reruns must not read as an autonomous loop"


def test_unterminated_call_is_incomplete_never_clean() -> None:
    lines = [json.loads(ln) for ln in (FIXTURES / "clean_session.jsonl").read_text().splitlines()]
    # drop the final tool_result -> the Bash call has no result (interrupted run)
    truncated = [ln for ln in lines if ln.get("uuid") != "u-103"]
    bundle = bundle_from_session(truncated)
    bash_step = next(s for s in bundle.steps if s.name == "Bash")
    # the models layer redacts the marker text; the load-bearing fact is that an
    # unterminated call carries an error (has_error=True), never a clean success
    assert bash_step.error is not None
    read_step = next(s for s in bundle.steps if s.name == "Read")
    assert read_step.error is None


def test_orphan_result_does_not_close_the_turn() -> None:
    lines = [json.loads(ln) for ln in (FIXTURES / "clean_session.jsonl").read_text().splitlines()]
    orphan = {
        "parentUuid": "x",
        "isSidechain": False,
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_never_seen", "content": "?"}],
        },
        "uuid": "u-orphan",
        "timestamp": "2026-06-10T02:00:06.000Z",
        "cwd": "C:\\proj\\sample",
        "sessionId": "c1ea0000-0000-4000-8000-000000000002",
        "version": "2.1.170",
        "gitBranch": "main",
    }
    # place the orphan between the two real calls, REMOVING the first call's real
    # result so the turn stays open: if the orphan wrongly closed it, the calls
    # would land in different turns.
    doctored = [ln for ln in lines if ln.get("uuid") != "u-102"]
    doctored.insert(2, orphan)
    bundle = bundle_from_session(doctored)
    turns = {s.parent_step_id for s in bundle.steps}
    assert len(turns) == 1, "an orphan result must be inert (must not split the turn)"


# ------------------------------------------------------------------ defensive


def test_malformed_line_raises_typed_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"type":"user"}\n{not json\n', encoding="utf-8")
    try:
        load_claude_session(bad)
    except ClaudeCodeSessionError as exc:
        assert "line 2" in str(exc)
    else:
        raise AssertionError("malformed line must raise ClaudeCodeSessionError")


def test_session_with_no_tool_calls_is_a_typed_error() -> None:
    records = [
        {
            "type": "user",
            "uuid": "u",
            "sessionId": "s",
            "timestamp": "2026-06-10T00:00:00Z",
            "message": {"role": "user", "content": "hi"},
        }
    ]
    try:
        bundle_from_session(records)
    except ClaudeCodeSessionError as exc:
        assert "no tool calls" in str(exc)
    else:
        raise AssertionError("chat-only session must raise (no run to record)")


# ------------------------------------------------------------------ routing


def test_probe_recognizes_claude_session_and_rejects_codex() -> None:
    assert looks_like_claude_session_file(FIXTURES / "retry_session.jsonl") is True
    assert looks_like_claude_session_file(CODEX_FIXTURES / "fire_no_edit_retry.jsonl") is False


def test_probe_sees_past_leading_meta_lines(tmp_path: Path) -> None:
    """REAL main-session files open with meta lines (sessionId but NO uuid:
    summary/mode/permission lines) before the first user/assistant line — the
    probe must scan past them, not judge line 1 alone (found on first real
    dogfood, 2026-06-10)."""
    f = tmp_path / "main-session.jsonl"
    meta = (
        '{"type":"summary","leafUuid":"x","sessionId":"s-1"}\n'
        '{"type":"mode","mode":"default","sessionId":"s-1"}\n'
    )
    f.write_text(meta + (FIXTURES / "retry_session.jsonl").read_text(), encoding="utf-8")
    assert looks_like_claude_session_file(f) is True


def test_probe_rejects_workflow_journal(tmp_path: Path) -> None:
    """A workflow journal.jsonl lives in the same tree but is not a session."""
    f = tmp_path / "journal.jsonl"
    f.write_text('{"type":"j","key":"k","agentId":"a"}\n' * 3, encoding="utf-8")
    assert looks_like_claude_session_file(f) is False


def test_find_recent_sessions_skips_non_session_jsonl(tmp_path: Path) -> None:
    import os
    import shutil

    root = tmp_path / "projects"
    (root / "p").mkdir(parents=True)
    journal = root / "p" / "journal.jsonl"
    journal.write_text('{"type":"j","key":"k","agentId":"a"}\n', encoding="utf-8")
    session = root / "p" / "real.jsonl"
    shutil.copy(FIXTURES / "clean_session.jsonl", session)
    # journal is NEWER — discovery must still return only the real session
    os.utime(session, (1_700_000_000, 1_700_000_000))
    os.utime(journal, (1_800_000_000, 1_800_000_000))
    found = find_recent_sessions(root)
    assert [p.name for p in found] == ["real.jsonl"]


def test_cli_routes_claude_session_by_shape(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    result = CliRunner().invoke(
        app, ["import", str(FIXTURES / "retry_session.jsonl"), "--db", str(db)]
    )
    assert result.exit_code == 0, result.output
    assert "imported run: cc_f00dfeed" in result.output


def test_verdict_fires_on_claude_retry_session(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    result = CliRunner().invoke(
        app, ["verdict", str(FIXTURES / "retry_session.jsonl"), "--db", str(db), "--json"]
    )
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["verdict"] == "receipts"
    assert payload["run_id"].startswith("cc_")


def test_verdict_clean_on_claude_clean_session(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    result = CliRunner().invoke(
        app, ["verdict", str(FIXTURES / "clean_session.jsonl"), "--db", str(db)]
    )
    assert result.exit_code == 0, result.output
    assert "clean" in result.output.lower()


# ------------------------------------------------------------------ discovery


def test_find_recent_sessions_newest_first_by_mtime(tmp_path: Path) -> None:
    import os
    import shutil

    root = tmp_path / "projects"
    (root / "proj-a").mkdir(parents=True)
    (root / "proj-b" / "subagents").mkdir(parents=True)
    older = root / "proj-a" / "aaaa.jsonl"
    newer = root / "proj-b" / "subagents" / "agent-bbbb.jsonl"
    shutil.copy(FIXTURES / "clean_session.jsonl", older)
    shutil.copy(FIXTURES / "retry_session.jsonl", newer)
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_800_000_000, 1_800_000_000))
    found = find_recent_sessions(root)
    assert [p.name for p in found] == ["agent-bbbb.jsonl", "aaaa.jsonl"]


def test_find_recent_sessions_missing_root_is_empty(tmp_path: Path) -> None:
    assert find_recent_sessions(tmp_path / "nope") == []


def test_verdict_latest_claude(tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "projects" / "proj-a"
    root.mkdir(parents=True)
    shutil.copy(FIXTURES / "retry_session.jsonl", root / "s1.jsonl")
    db = tmp_path / "ledger.sqlite"
    result = CliRunner().invoke(
        app,
        [
            "verdict", "--latest-claude",
            "--claude-projects-root", str(tmp_path / "projects"),
            "--db", str(db), "--json",
        ],
    )
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["run_id"].startswith("cc_")
