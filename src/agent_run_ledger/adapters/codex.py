"""Codex rollout-log adapter (the ONE provider adapter; Codex only).

The Codex CLI writes a JSONL rollout log — one JSON object per line — at
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``. This adapter maps that
provider-specific log into the NEUTRAL ``TraceBundle`` schema (``core.models``).
ALL Codex-specific field knowledge lives HERE; the core package never names a
Codex field.

WHAT THIS ADAPTER DOES NOT DO (invariant — facts only):
  * No pricing. Per-step token usage is 0 (Codex logs do not break model usage
    per call); the run total is the FINAL cumulative ``token_count``. Cost is
    computed ON READ by ``core.cost``.
  * No "this was a retry" judgment. It records the content-free FACTS the
    detector keys on (``span_kind``, ``retry_scope``, the per-turn parent, an
    input fingerprint, the parsed error presence); the retry COLLAPSE is a
    JUDGMENT computed on read by ``core.prescriptions``.
  * No proof grades. ``provenance_hash`` is stamped locally over the raw steps.

UNTRUSTED INPUT (Rule 8-adjacent): a rollout file is hostile. Size + line count
are hard-capped; every field is treated as inert string/scalar data (json.loads
only produces data — nothing is ever evaluated); malformed input raises a typed
``CodexRolloutError`` (never a crash).

THE TWO LOAD-BEARING MAPPINGS (confirmed against ``core.prescriptions``
derive_retry_steps lines 38-50 + ``core.retries``):

  retry_scope  <- the SESSION id (``session_meta.payload.id``). One Codex session
                  is one agent on one workspace; the session id is the only stable
                  ancestor available, so it is the retry scope. This is COARSE BY
                  DESIGN: every tool call in a session shares one scope, so
                  discrimination between a genuine retry and legitimate repetition
                  comes from tool NAME + input FINGERPRINT + per-turn parent, NOT
                  from scope. ``retries`` requires a non-null scope on both sides
                  (else it abstains); a constant non-null scope satisfies that
                  without ever FALSELY separating two real attempts.

  parent_step_id (the "turn_id" the detector reads) <- a SYNTHESIZED per-turn id.
                  A Codex rollout has NO per-inference turn id on tool-call records
                  (one ``task_started`` / one ``turn_context`` for the whole
                  session, but many calls). A real model re-invocation is marked by
                  a tool-call OUTPUT: the model runs, calls a tool, sees the output,
                  then is re-invoked. So a tool-call OUTPUT is a TURN BOUNDARY:
                  consecutive tool calls emitted BEFORE the next output share one
                  turn; the first call AFTER an output starts a new turn. This makes
                  a cross-turn retry have DISTINCT turn ids per attempt (fires) while
                  a same-turn fan-out of identical calls shares ONE turn id (the B3
                  guard rejects it). Getting this backwards (unique id per call)
                  false-fires same-turn fan-out; a single id for all calls
                  (no boundary) collapses fan-out into a false retry.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_run_ledger.core.models import (
    RunRecord,
    StepRecord,
    TraceBundle,
    classify_error,
    utc_now_iso,
)
from agent_run_ledger.core.provenance import compute_provenance_hash

# Untrusted-input ceilings. Generous for a real rollout (a multi-hour session is a
# few hundred KB to a few MB and a few thousand records); the point is a HARD
# ceiling against a hostile file, not a tight fit. Mirrors ``core.io`` discipline.
MAX_ROLLOUT_BYTES = 64 * 1024 * 1024
MAX_ROLLOUT_LINES = 500_000

# Tool-call response items. ``exec_command`` arrives as ``function_call``;
# ``apply_patch`` arrives as ``custom_tool_call`` (a different response-item shape
# carrying the patch in ``input`` instead of ``arguments``). BOTH map to a
# span_kind='function' step — apply_patch's different NAME is what breaks a retry
# run between two identical commands (the fix-then-rerun abstain), so it must be a
# function step, never dropped or demoted to a non-function kind.
_TOOL_CALL_TYPES = ("function_call", "custom_tool_call")
_TOOL_OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")
# A user/instruction message between repeated tool calls means the human DIRECTED
# the rerun -> NOT an autonomous (blind) retry loop. The adapter bumps the retry
# "segment" on this boundary so calls before vs. after land in DIFFERENT
# retry_scopes; core then refuses to collapse across the boundary (retries.py:
# "a handoff to a DIFFERENT scope must NOT collapse"). Core stays provider-neutral
# -- the user-boundary concept lives ONLY here in the adapter (A1, vault-CC 2026-06-05).
_USER_MESSAGE_TYPES = ("user_message",)

# Exit-status phrasings seen on real outputs:
#   function_call_output:      "Process exited with code N"
#   custom_tool_call_output:   "Exit code: N"
# An output with NO recognizable status line is treated as NO error (conservative:
# never manufacture an error that would trigger a false retry).
_EXIT_RE = re.compile(r"(?:Process exited with code|Exit code:)\s+(\d+)")


class CodexRolloutError(ValueError):
    """Raised when an untrusted Codex rollout cannot be parsed SAFELY (too large,
    too many lines, malformed, not a rollout, or empty of tool calls). Always a
    typed, caught error — never a crash or an uncaught decode/recursion error."""


# --------------------------------------------------------------------------- #
# Loading (defensive)
# --------------------------------------------------------------------------- #
def load_codex_rollout(path: Path) -> list[dict[str, Any]]:
    """Parse an untrusted ``.jsonl`` rollout into a list of record dicts, safely.

    Size and line count are bounded BEFORE building the list; each line must be a
    JSON object (a non-object line is rejected); a malformed line raises a typed
    ``CodexRolloutError``. Nothing in a record is ever evaluated — json.loads only
    produces inert data."""
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_ROLLOUT_BYTES:
        raise CodexRolloutError(f"rollout file is too large: {size} bytes > {MAX_ROLLOUT_BYTES}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CodexRolloutError(f"rollout file is not valid UTF-8: {exc}") from exc

    records: list[dict[str, Any]] = []
    line_count = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        line_count += 1
        if line_count > MAX_ROLLOUT_LINES:
            raise CodexRolloutError(
                f"rollout has too many lines: > {MAX_ROLLOUT_LINES}"
            )
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CodexRolloutError(f"malformed JSONL at line {lineno}: {exc}") from exc
        if not isinstance(obj, dict):
            raise CodexRolloutError(f"rollout line {lineno} is not a JSON object")
        records.append(obj)
    return records


def looks_like_jsonl(path: Path) -> bool:
    """Provider-neutral format probe: True if *path* is a Codex-style line-delimited
    JSON log (NOT a single TraceBundle object). Used by the CLI router. Names NO
    Codex field — it only inspects the extension and, as a fallback, whether the
    first two non-empty lines are each a JSON object (a single pretty-printed
    TraceBundle has a ``{`` alone on line 1, so it is NOT mistaken for JSONL)."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return True
    try:
        with path.open(encoding="utf-8") as fh:
            objs = 0
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    return False
                if not isinstance(parsed, dict):
                    return False
                objs += 1
                if objs >= 2:
                    return True
        return False
    except (OSError, UnicodeDecodeError):
        return False


# --------------------------------------------------------------------------- #
# Mapping rollout -> TraceBundle (facts only)
# --------------------------------------------------------------------------- #
def bundle_from_rollout(records: list[dict[str, Any]]) -> TraceBundle:
    """Map an ordered list of rollout records into a neutral ``TraceBundle``.

    One ``StepRecord`` per tool call (raw facts; no collapse — that is on read).
    Token totals come from the FINAL cumulative ``token_count``; per-step token
    usage is 0 (honest). ``provenance_hash`` is stamped locally over the raw
    steps."""
    session = _session_meta(records)
    session_id = str(session.get("id") or f"codex_{uuid4().hex[:12]}")
    run_id = f"codex_{session_id}"
    model = _model(records)
    workflow = _workflow(session, session_id)

    steps = _steps_from_records(records, run_id, session_id)
    if not steps:
        raise CodexRolloutError(
            f"rollout {session_id!r} contains no tool calls (no run to record)"
        )

    started_at = steps[0].started_at
    ended_at = max(s.ended_at for s in steps)
    total_input, total_output = _run_token_totals(records)
    run = RunRecord(
        id=run_id,
        workflow=workflow,
        framework="codex-cli",
        provider="codex",
        model=model,
        started_at=started_at,
        ended_at=ended_at,
        # L14: success_label is the adapter's PROVISIONAL self-report, not a
        # verdict. A rollout has no single terminal-status field, so we record the
        # honest "unknown" rather than guessing from the last exit code (which is a
        # judgment, not a captured fact).
        success_label="unknown",
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )
    bundle = TraceBundle(run=run, steps=steps)
    # L5: stamp the provenance hash locally over the RAW steps (no derived
    # collapse) — the un-backfillable seed of proof-of-real.
    return replace(bundle, run=replace(run, provenance_hash=compute_provenance_hash(bundle)))


def _steps_from_records(
    records: list[dict[str, Any]], run_id: str, session_id: str
) -> list[StepRecord]:
    """Build one StepRecord per tool call, pairing each call to its output by
    ``call_id`` and synthesizing the per-turn parent from output-delimited turn
    boundaries (see module docstring)."""
    outputs = _outputs_by_call_id(records)

    # Synthesize turn ids: a tool-call OUTPUT closes the current turn; the next
    # tool call opens a fresh turn. Consecutive calls before the next output share
    # the open turn (a same-turn fan-out). Walk the records in order.
    turn_index = 0
    turn_open = False  # has a call been emitted in the current (un-closed) turn?
    segment = 0  # bumped on a user-message boundary; partitions retry_scope (A1)
    steps: list[StepRecord] = []
    call_seq = 0

    for rec in records:
        payload = rec.get("payload")
        if not isinstance(payload, dict):
            continue
        ptype = payload.get("type")
        if ptype in _USER_MESSAGE_TYPES:
            # A user instruction. If it falls between repeated tool calls, the human
            # DIRECTED the rerun -> not an autonomous retry. Bump the segment so the
            # next call's retry_scope differs from the prior call's; core then won't
            # collapse across this boundary. Also close the turn (model re-engages
            # after the user). Core is untouched (provider-neutral invariant).
            segment += 1
            turn_open = False
            continue
        if ptype in _TOOL_OUTPUT_TYPES:
            # A tool result: the model will be re-invoked -> close the turn so the
            # NEXT tool call starts a new one. (Multiple outputs in a row — e.g. a
            # same-turn fan-out's two outputs — only close once; already closed is a
            # no-op.)
            if turn_open:
                turn_open = False
            continue
        if ptype not in _TOOL_CALL_TYPES:
            continue
        # A tool call. Open a new turn only if the previous one was closed (by an
        # output) — consecutive calls with no output between share the open turn.
        if not turn_open:
            turn_index += 1
            turn_open = True
        call_seq += 1
        steps.append(
            _step_from_call(
                payload=payload,
                rec=rec,
                run_id=run_id,
                session_id=session_id,
                turn_id=f"{session_id}:turn{turn_index}",
                segment=segment,
                seq=call_seq,
                outputs=outputs,
            )
        )
    return steps


def _step_from_call(
    *,
    payload: dict[str, Any],
    rec: dict[str, Any],
    run_id: str,
    session_id: str,
    turn_id: str,
    segment: int,
    seq: int,
    outputs: dict[str, dict[str, Any]],
) -> StepRecord:
    name = str(payload.get("name") or "unknown-tool")
    call_id = payload.get("call_id")
    started_at = str(rec.get("timestamp") or "")
    output_rec = outputs.get(str(call_id)) if call_id is not None else None
    ended_at, exit_code = _output_facts(output_rec)
    # ended_at falls back to the call's own start when no output was paired (so the
    # non-overlap time check still has a real, non-empty ``ended_at``).
    if not ended_at:
        ended_at = started_at or utc_now_iso()
    if not started_at:
        started_at = ended_at

    # has_error is PARSED from the exit status, NOT defaulted. exit_code is None
    # when no status line was present -> treat as no error (conservative). A
    # content-free marker is stored as the error so ``error is not None`` (the
    # detector's has_error) reflects the real failure; the raw message is never
    # stored (models.sanitize_error redacts to a constant; classify_error drops it).
    raw_error = f"exit_code={exit_code}" if (exit_code is not None and exit_code != 0) else None

    return StepRecord(
        id=f"step_{session_id[:8]}_{seq:04d}",
        run_id=run_id,
        step_type="function",
        # span_kind MUST be 'function' — the only kind ``retries._is_tool`` groups.
        span_kind="function",
        name=name,
        started_at=started_at,
        ended_at=ended_at,
        # parent_step_id is the per-turn id the detector reads as turn_id.
        parent_step_id=turn_id,
        # retry_scope is the session id PARTITIONED BY user-message segment: calls
        # separated by a user instruction land in different scopes so core won't
        # collapse a user-DIRECTED rerun as an autonomous retry loop (A1). Within one
        # segment it is the stable session scope (coarse-by-design; see module doc).
        retry_scope=f"{session_id}:seg{segment}",
        # input_fingerprint: a one-way digest of the raw tool input — content-free,
        # so a genuine retry (same input) is distinguishable from repetition without
        # storing the input.
        input_fingerprint=_input_fingerprint(payload),
        # Per-step model usage is 0 (honest): Codex logs do not break usage per
        # call; the run total is the final cumulative token_count.
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        error=raw_error,
        error_class=classify_error(raw_error),
        redaction_mode="metadata_only",
    )


def _output_facts(output_rec: dict[str, Any] | None) -> tuple[str, int | None]:
    """Return (ended_at, exit_code) for a paired tool-output record. exit_code is
    None when the output carries no recognizable status line."""
    if not isinstance(output_rec, dict):
        return "", None
    ended_at = str(output_rec.get("timestamp") or "")
    payload = output_rec.get("payload")
    output_text = payload.get("output") if isinstance(payload, dict) else None
    return ended_at, _parse_exit_code(output_text)


def _parse_exit_code(output: Any) -> int | None:
    """Extract the integer exit code from a tool-output text blob, or None if no
    recognizable status line is present (-> treated as no error by the caller).

    CRITICAL: this is the guard against marking every output a success. A non-zero
    code means the command FAILED (``has_error=True``)."""
    if not isinstance(output, str):
        return None
    match = _EXIT_RE.search(output)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # pragma: no cover - regex already constrains to digits
        return None


def _outputs_by_call_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map each call_id to its FIRST output (B2). On untrusted logs a duplicate
    output must NOT overwrite the original: a hostile/corrupt rollout appending a
    later exit-0 for a call_id that originally FAILED would otherwise erase the
    failure (last-write-wins) and silently suppress retry detection. First write wins
    — the original recorded result is authoritative; later duplicates are ignored."""
    by_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        payload = rec.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("type") in _TOOL_OUTPUT_TYPES:
            call_id = payload.get("call_id")
            if call_id is not None and str(call_id) not in by_id:
                by_id[str(call_id)] = rec
    return by_id


def _input_fingerprint(payload: dict[str, Any]) -> str | None:
    """Stable one-way digest of the tool's raw input, or None when absent.

    ``exec_command`` carries the input in ``arguments`` (a JSON string);
    ``apply_patch`` carries it in ``input`` (a patch string). A digest is not
    reversible to content, so it is safe to store as a retry-grouping FACT; the raw
    input itself is never stored."""
    raw = payload.get("arguments")
    if raw is None:
        raw = payload.get("input")
    if raw is None:
        return None
    if isinstance(raw, str):
        # ``exec_command.arguments`` is a JSON string, so two semantically
        # identical commands with reordered keys / different whitespace would
        # otherwise hash differently and a genuine retry would never collapse (a
        # false negative). Canonicalize before hashing; if it is NOT valid JSON
        # (e.g. an ``apply_patch`` patch string), fall back to the raw string —
        # behavior unchanged. Still content-free: a one-way digest, never the
        # raw input.
        try:
            canonical = json.dumps(
                json.loads(raw), sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError):
            canonical = raw
    else:
        try:
            canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            canonical = str(raw)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Run-level metadata extraction
# --------------------------------------------------------------------------- #
def _session_meta(records: list[dict[str, Any]]) -> dict[str, Any]:
    for rec in records:
        if rec.get("type") == "session_meta":
            payload = rec.get("payload")
            if isinstance(payload, dict):
                return payload
    return {}


def _model(records: list[dict[str, Any]]) -> str:
    """The model is on ``turn_context.payload.model`` (e.g. ``gpt-5.5``). Fall back
    to ``unknown-model`` when no turn_context carries it."""
    for rec in records:
        if rec.get("type") == "turn_context":
            payload = rec.get("payload")
            if isinstance(payload, dict) and payload.get("model"):
                return str(payload["model"])
    return "unknown-model"


def _workflow(session: dict[str, Any], session_id: str) -> str:
    originator = session.get("originator")
    if originator:
        return f"codex-{originator}"
    return f"codex-session-{session_id[:8]}"


def _run_token_totals(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from the FINAL cumulative
    ``token_count`` event.

    ``token_count.info.total_token_usage`` is a RUNNING TOTAL emitted repeatedly;
    summing every event over-counts ~N x. We take the LAST event's total. None
    present -> (0, 0), the honest default."""
    last_total: dict[str, Any] | None = None
    for rec in records:
        payload = rec.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        info = payload.get("info")
        if not isinstance(info, dict):
            continue
        total = info.get("total_token_usage")
        if isinstance(total, dict):
            last_total = total
    if last_total is None:
        return 0, 0
    return _as_int(last_total.get("input_tokens")), _as_int(last_total.get("output_tokens"))


def _as_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0
