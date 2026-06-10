"""Claude Code session-log adapter (ONE provider adapter; Claude Code only).

Claude Code writes a JSONL session log at
``~/.claude/projects/<munged-project-dir>/<session-uuid>.jsonl`` (subagent
sessions land under ``<project>/subagents/agent-*.jsonl`` with the same line
shape). One line = one JSON object:

  * top-level: ``type`` (user/assistant/system/…), ``uuid``, ``parentUuid``,
    ``timestamp`` (ISO), ``sessionId``, ``cwd``, ``gitBranch``, ``version``;
  * an ASSISTANT line carries ``message.model``, ``message.id`` (one per API
    response — several lines may share it while a response streams), per-response
    ``message.usage`` token counts, and ``message.content`` blocks including
    ``tool_use`` ({id, name, input});
  * a USER line may carry ``tool_result`` blocks ({tool_use_id, is_error,
    content}) — the paired result of an earlier ``tool_use`` — or be a real
    human instruction (string content / ``text`` blocks).

UNTRUSTED INPUT (Rule 8-adjacent): a session file is hostile. Size + line count
are hard-capped; every field is treated as inert string/scalar data (json.loads
only produces data — nothing is ever evaluated); malformed input raises a typed
``ClaudeCodeSessionError`` (never a crash). Raw tool inputs/outputs are NEVER
stored — only one-way fingerprints and content-free error markers.

SEMANTICS MIRROR ``adapters.codex`` deliberately (the retry detector depends on
them):

  * turn synthesis — a paired tool result closes the open turn; the next call
    opens a fresh one; consecutive calls before a result share the turn (fan-out);
    an ORPHAN result (unknown id) is inert;
  * user-boundary segments — a REAL human instruction between repeated calls
    means the rerun was DIRECTED, not autonomous: the segment bumps, partitioning
    ``retry_scope`` so core never collapses across the boundary;
  * first-write-wins outputs — a duplicate/late result for an already-paired
    call id cannot overwrite the original (a hostile append must not erase a
    recorded failure);
  * incomplete calls — a ``tool_use`` with NO paired result is an interrupted
    run, recorded with an error marker, never as a clean success.

TOKEN HONESTY: usage is recorded per API response (``message.id``), and several
session lines may repeat/accumulate usage for the same response while it
streams — so totals are summed over the LAST usage seen per ``message.id``,
never per line (per-line summing double-counts). Cache-creation/cache-read
token counts are EXCLUDED from the totals: ``total_input_tokens`` is the
uncached input actually sent per response. That is a deliberate UNDERCOUNT of
processed tokens, disclosed here, chosen over inventing a blended number the
trace does not state. Per-step usage is 0 (honest: the log does not attribute
tokens to tool calls).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_run_ledger.core.io import TraceParseError, _check_depth
from agent_run_ledger.core.models import (
    RunRecord,
    StepRecord,
    TraceBundle,
    classify_error,
    utc_now_iso,
)
from agent_run_ledger.core.provenance import compute_provenance_hash

# Untrusted-input ceilings — mirrors adapters.codex / core.io discipline.
MAX_SESSION_BYTES = 64 * 1024 * 1024
MAX_SESSION_LINES = 500_000

# Where Claude Code writes session logs.
DEFAULT_PROJECTS_ROOT = Path.home() / ".claude" / "projects"


class ClaudeCodeSessionError(ValueError):
    """Raised when an untrusted Claude Code session cannot be parsed SAFELY (too
    large, too many lines, malformed, not a session, or empty of tool calls).
    Always a typed, caught error — never a crash."""


# --------------------------------------------------------------------------- #
# Loading (defensive)
# --------------------------------------------------------------------------- #
def load_claude_session(path: Path) -> list[dict[str, Any]]:
    """Parse an untrusted session ``.jsonl`` into a list of record dicts, safely.

    Same defensive contract as the Codex loader: bounded size and line count,
    typed errors, every line must be a JSON object, nothing evaluated."""
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_SESSION_BYTES:
        raise ClaudeCodeSessionError(
            f"session file is too large: {size} bytes > {MAX_SESSION_BYTES}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ClaudeCodeSessionError(f"session file is not valid UTF-8: {exc}") from exc

    records: list[dict[str, Any]] = []
    line_count = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        line_count += 1
        if line_count > MAX_SESSION_LINES:
            raise ClaudeCodeSessionError(f"session has too many lines: > {MAX_SESSION_LINES}")
        # Per-line depth bound (cold-review 2026-06-11) — see adapters.codex note.
        try:
            _check_depth(line)
        except TraceParseError as exc:
            raise ClaudeCodeSessionError(f"hostile nesting at line {lineno}: {exc}") from exc
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClaudeCodeSessionError(f"malformed JSONL at line {lineno}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ClaudeCodeSessionError(f"session line {lineno} is not a JSON object")
        records.append(obj)
    return records


# Real main-session files open with META lines (summary/mode/permission — they
# carry sessionId but no uuid) before the first user/assistant line, so the probe
# must scan a bounded PREFIX, never judge line 1 alone (first-dogfood finding,
# 2026-06-10: line-1-only routed a real session to the Codex adapter).
_PROBE_MAX_LINES = 50


def looks_like_claude_session_file(path: Path) -> bool:
    """Format probe for the CLI router: True if any of the first ~50 non-empty
    lines has the Claude Code conversation-line shape (top-level ``sessionId`` +
    ``uuid`` + ``type``). Codex rollout lines nest under ``payload`` and carry
    neither; workflow journals carry no ``sessionId`` at all. Any read/parse
    failure returns False (the caller falls through to the Codex route, whose
    typed errors fail closed)."""
    try:
        with Path(path).open(encoding="utf-8") as fh:
            seen = 0
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                seen += 1
                if seen > _PROBE_MAX_LINES:
                    return False
                if len(line) > 1_000_000:  # one hostile mega-line must not stall the probe
                    return False
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    return False
                if not isinstance(obj, dict):
                    return False
                if "sessionId" in obj and "uuid" in obj and "type" in obj:
                    return True
                # a meta/journal line: keep scanning the bounded prefix
    except (OSError, UnicodeDecodeError):
        return False
    return False


def find_recent_sessions(root: Path | None = None, limit: int = 15) -> list[Path]:
    """Newest-first session files under *root* (default ``~/.claude/projects``).

    Claude Code session filenames are bare UUIDs (no embedded timestamp), so —
    unlike Codex rollouts — modification time is the only ordering signal. mtime
    can lie on copied/synced files; acceptable for a local convenience picker,
    documented here. The projects tree also holds NON-session ``.jsonl`` files
    (workflow journals, queues), so each candidate is shape-probed and
    non-sessions are skipped — lazily, newest first, stopping at *limit*.
    Defensive: missing root returns ``[]``."""
    base = root if root is not None else DEFAULT_PROJECTS_ROOT
    if not base.is_dir():
        return []
    candidates = sorted(
        base.glob("**/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    found: list[Path] = []
    for path in candidates:
        if looks_like_claude_session_file(path):
            found.append(path)
            if len(found) >= limit:
                break
    return found


# --------------------------------------------------------------------------- #
# Mapping session -> TraceBundle (facts only)
# --------------------------------------------------------------------------- #
def bundle_from_session(records: list[dict[str, Any]]) -> TraceBundle:
    """Map an ordered list of session records into a neutral ``TraceBundle``.

    One ``StepRecord`` per ``tool_use`` block (raw facts; the retry collapse is a
    judgment computed on read). Token totals are summed over the last usage per
    ``message.id`` (see module docstring). ``provenance_hash`` is stamped locally
    over the raw steps."""
    session_id = _session_id(records)
    run_id = f"cc_{session_id}"
    model = _first_model(records)
    workflow = _workflow(records)

    steps = _steps_from_records(records, run_id, session_id)
    if not steps:
        raise ClaudeCodeSessionError(
            f"session {session_id!r} contains no tool calls (no run to record)"
        )

    started_at = steps[0].started_at
    ended_at = max(s.ended_at for s in steps)
    total_input, total_output = _run_token_totals(records)
    run = RunRecord(
        id=run_id,
        workflow=workflow,
        framework="claude-code",
        provider="anthropic",
        model=model,
        started_at=started_at,
        ended_at=ended_at,
        # L14: a session log has no single terminal-status field; "unknown" is the
        # honest self-report, never a guess.
        success_label="unknown",
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )
    bundle = TraceBundle(run=run, steps=steps)
    # L5: stamp provenance locally over the RAW steps — un-backfillable.
    return replace(bundle, run=replace(run, provenance_hash=compute_provenance_hash(bundle)))


def _session_id(records: list[dict[str, Any]]) -> str:
    for rec in records:
        sid = rec.get("sessionId")
        if isinstance(sid, str) and sid:
            return sid
    return f"cc_{uuid4().hex[:12]}"


def _first_model(records: list[dict[str, Any]]) -> str:
    for rec in records:
        if rec.get("type") != "assistant":
            continue
        message = rec.get("message")
        if isinstance(message, dict):
            model = message.get("model")
            if isinstance(model, str) and model:
                return model
    return "unknown"


def _workflow(records: list[dict[str, Any]]) -> str:
    """The workflow label: the project directory NAME from ``cwd`` (metadata, not
    content — the full path is never stored)."""
    for rec in records:
        cwd = rec.get("cwd")
        if isinstance(cwd, str) and cwd:
            name = Path(cwd).name
            if name:
                return name
    return "claude-code-session"


def _steps_from_records(
    records: list[dict[str, Any]], run_id: str, session_id: str
) -> list[StepRecord]:
    outputs = _results_by_tool_use_id(records)

    turn_index = 0
    turn_open = False
    open_turn_call_ids: set[str] = set()
    segment = 0
    steps: list[StepRecord] = []
    call_seq = 0

    for rec in records:
        rtype = rec.get("type")
        if rtype == "user":
            result_ids = _tool_result_ids(rec)
            if result_ids:
                # A paired tool result closes the open turn — only when it pairs to
                # a call actually emitted in THIS turn (orphans are inert; B3).
                if turn_open and any(rid in open_turn_call_ids for rid in result_ids):
                    turn_open = False
                    open_turn_call_ids = set()
                continue
            if _is_human_instruction(rec):
                # A real human instruction: directed rerun boundary (A1).
                segment += 1
                turn_open = False
                open_turn_call_ids = set()
            continue
        if rtype != "assistant":
            continue
        for block in _content_blocks(rec):
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if not turn_open:
                turn_index += 1
                turn_open = True
                open_turn_call_ids = set()
            call_id = str(block.get("id") or f"call_{call_seq}")
            open_turn_call_ids.add(call_id)
            call_seq += 1
            steps.append(
                _step_from_tool_use(
                    block=block,
                    rec=rec,
                    run_id=run_id,
                    session_id=session_id,
                    turn_id=f"{session_id}:turn{turn_index}",
                    segment=segment,
                    seq=call_seq,
                    result=outputs.get(call_id),
                )
            )
    return steps


def _step_from_tool_use(
    *,
    block: dict[str, Any],
    rec: dict[str, Any],
    run_id: str,
    session_id: str,
    turn_id: str,
    segment: int,
    seq: int,
    result: dict[str, Any] | None,
) -> StepRecord:
    name = str(block.get("name") or "unknown-tool")
    started_at = str(rec.get("timestamp") or "")
    ended_at, is_error = _result_facts(result)
    if not ended_at:
        ended_at = started_at or utc_now_iso()
    if not started_at:
        started_at = ended_at

    # Error marker is CONTENT-FREE: the result's text is never stored. A missing
    # result is an INTERRUPTED call (Task 54 honesty) — never a clean success.
    if result is None:
        raw_error = "incomplete: tool call has no result (run interrupted before completion)"
    elif is_error:
        raw_error = "tool_result.is_error=true"
    else:
        raw_error = None

    return StepRecord(
        id=f"step_{session_id[:8]}_{seq:04d}",
        run_id=run_id,
        step_type="function",
        # 'function' is the only kind ``retries._is_tool`` groups.
        span_kind="function",
        name=name,
        started_at=started_at,
        ended_at=ended_at,
        parent_step_id=turn_id,
        retry_scope=f"{session_id}:seg{segment}",
        input_fingerprint=_input_fingerprint(block),
        # The log does not attribute tokens to tool calls — 0 is the honest fact.
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        error=raw_error,
        error_class=classify_error(raw_error),
        redaction_mode="metadata_only",
    )


def _content_blocks(rec: dict[str, Any]) -> list[Any]:
    message = rec.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    return content if isinstance(content, list) else []


def _tool_result_ids(rec: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for block in _content_blocks(rec):
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tu = block.get("tool_use_id")
            if tu is not None:
                ids.add(str(tu))
    return ids


def _is_human_instruction(rec: dict[str, Any]) -> bool:
    """True for a REAL user instruction (string content or a ``text`` block),
    never for tool-result carriers or meta/system-injected lines."""
    if rec.get("isMeta"):
        return False
    message = rec.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "text" for b in content)
    return False


def _results_by_tool_use_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map tool_use_id -> the FIRST result line carrying it (first-write-wins: a
    hostile/corrupt late duplicate must not erase a recorded failure)."""
    by_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        if rec.get("type") != "user":
            continue
        for block in _content_blocks(rec):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tu = block.get("tool_use_id")
            if tu is None:
                continue
            key = str(tu)
            if key not in by_id:
                by_id[key] = {"timestamp": rec.get("timestamp"), "is_error": block.get("is_error")}
    return by_id


def _result_facts(result: dict[str, Any] | None) -> tuple[str, bool]:
    if not isinstance(result, dict):
        return "", False
    ended_at = str(result.get("timestamp") or "")
    return ended_at, result.get("is_error") is True


def _input_fingerprint(block: dict[str, Any]) -> str | None:
    """Stable one-way digest of the tool input (canonicalized JSON), or None when
    absent. Content-free: the digest is not reversible; the raw input is never
    stored."""
    raw = block.get("input")
    if raw is None:
        return None
    try:
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        canonical = str(raw)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_token_totals(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Sum the LAST usage seen per ``message.id`` (a streaming response repeats
    usage across lines; per-line summing double-counts). Cache token classes are
    excluded — see the module docstring's disclosed undercount."""
    last_by_message: dict[str, tuple[int, int]] = {}
    for rec in records:
        if rec.get("type") != "assistant":
            continue
        message = rec.get("message")
        if not isinstance(message, dict):
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        key = str(message.get("id") or rec.get("uuid") or len(last_by_message))
        last_by_message[key] = (
            _as_int(usage.get("input_tokens")),
            _as_int(usage.get("output_tokens")),
        )
    total_in = sum(v[0] for v in last_by_message.values())
    total_out = sum(v[1] for v in last_by_message.values())
    return total_in, total_out


def _as_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0
