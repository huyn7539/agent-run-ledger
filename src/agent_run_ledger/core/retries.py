"""Trace-derived retry detection (provider-neutral).

Agent traces generally do NOT emit a ``retry_count`` field. A real retry loop is
N repeated executions of the SAME operation on the SAME input, in sequence, with
at least one failure. This module collapses such a run of attempts into a single
logical step so the existing retry/cost detector
(``prescriptions.detect_retry_cost_loops``) fires on a REAL trace.

THE LOAD-BEARING REQUIREMENT (NEW-4): distinguish a genuine retry loop from
LEGITIMATE repeated tool calls (the same tool invoked on different inputs as
normal work). A false positive makes the prescription WRONG on a real trace —
the worst failure mode — so every tie resolves toward NOT collapsing (a
conservative false-negative).

This module is provider-neutral: it operates on ``AttemptFacts`` (a neutral
projection the adapter builds). It never imports any provider SDK and never sees
raw content — only a precomputed input *fingerprint* (a digest), the error
PRESENCE, the bounded error class, and timing. The adapter is responsible for
fingerprinting raw input transiently and never storing it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttemptFacts:
    """A neutral, content-free projection of one span, for retry grouping.

    ``retry_scope`` is the STABLE ancestor that groups retries of one call site
    across turns. A real agentic retry loop spans multiple turns, so the immediate
    parent differs between attempts; the scope (the agent-span ancestor, resolved
    by the adapter) is stable. Keying on the immediate parent silently no-ops on
    every real multi-turn retry — the scope is what makes cross-turn detection
    correct. None when no scope could be resolved.

    ``input_fingerprint`` is a digest of the raw tool input (or None when input
    was not captured). It is used ONLY to test same-vs-different input; it carries
    no recoverable content.

    ``turn_id`` is the IMMEDIATE parent span id (the per-turn ``turn_span`` for a
    real SDK function call). A genuine agentic retry loop spans MULTIPLE turns, so
    the immediate parent DIFFERS between attempts even though ``retry_scope`` (the
    agent ancestor) is stable. The grouper uses this to reject a SAME-TURN
    sequential fan-out (3 calls in one turn) that would otherwise look identical to
    a retry under scope+name+input alone (NEW-4 false-positive guard, B3). None
    when no immediate parent was captured.
    """

    index: int
    name: str
    span_kind: str | None
    retry_scope: str | None
    started_at: str
    ended_at: str
    has_error: bool
    error_class: str | None
    input_fingerprint: str | None
    turn_id: str | None = None


def _is_tool(a: AttemptFacts) -> bool:
    """Only function/tool spans are eligible to form a retry loop — they carry the
    name + input needed to tell a genuine loop from legitimate repetition."""
    return a.span_kind == "function"


def _is_retry_continuation(prev: AttemptFacts, cur: AttemptFacts) -> bool:
    """True iff tool attempt *cur* is another attempt of the SAME operation as the
    previous tool attempt *prev* in an open run.

    ALL conditions must hold:
      - both are function/tool spans
      - identical tool name
      - identical retry_scope (same stable ancestor/call site across turns); a
        handoff to a DIFFERENT agent calling the same tool has a different scope
        and must NOT collapse. Both scopes must be present (None -> abstain).
      - both carry a captured input fingerprint, and they are EQUAL (same input)
      - sequential, non-overlapping in time (cur starts at/after prev ended);
        overlap means parallelism (concurrent fan-out), not retry -> reject
    """
    if not _is_tool(prev) or not _is_tool(cur):
        return False
    if prev.name != cur.name:
        return False
    if prev.retry_scope is None or cur.retry_scope is None:
        return False
    if prev.retry_scope != cur.retry_scope:
        return False
    if prev.input_fingerprint is None or cur.input_fingerprint is None:
        return False
    if prev.input_fingerprint != cur.input_fingerprint:
        return False
    # ISO-8601 Zulu timestamps sort lexically, so string compare is a correct time
    # compare. cur.started_at < prev.ended_at -> overlapping windows -> reject.
    if cur.started_at < prev.ended_at:
        return False
    return True


def _spans_multiple_turns(group: list[AttemptFacts]) -> bool:
    """True iff the attempts in *group* span MORE THAN ONE distinct turn (immediate
    parent). A genuine cross-turn retry has a different turn parent per attempt; a
    same-turn fan-out shares one. Missing turn ids collapse to a single ``{None}``
    set -> False, so unknown structure ABSTAINS (never a false positive)."""
    return len({a.turn_id for a in group}) > 1


def collapse_retry_groups(attempts: list[AttemptFacts]) -> list[list[int]]:
    """Group *attempts* (sorted by the caller) into retry loops.

    A REAL agentic retry loop interleaves a model/response turn before each tool
    retry: ``response, fn(fail), response, fn(fail), response, fn(ok)``. So a
    NON-tool span between two same-target tool attempts is a TURN BOUNDARY — it
    does NOT break the retry run, and it is never part of the tool group (it stays
    its own singleton). Only a DIFFERENT tool between attempts is real interleaved
    work that breaks the run.

    Returns groups of original ``index`` values, in order. A tool group of length
    >= 2 with at least one error is a retry loop; everything else stays a
    singleton. Callers sort attempts deterministically (by ``(started_at, index)``)
    before calling — this function trusts the given order for adjacency.
    """
    if not attempts:
        return []

    # `open_run` accumulates consecutive same-target tool attempts (model-turn
    # spans between them are tolerated and emitted separately as singletons).
    open_run: list[AttemptFacts] = []
    emitted: list[list[AttemptFacts]] = []  # closed runs + singletons, in order

    for cur in attempts:
        if not _is_tool(cur):
            # A model/response/agent turn span: tolerated between tool attempts —
            # it neither joins nor breaks an open tool run. Emit it standalone.
            emitted.append([cur])
            continue
        # cur is a tool span.
        if open_run and _is_retry_continuation(open_run[-1], cur):
            open_run.append(cur)
        else:
            if open_run:
                emitted.append(open_run)
            open_run = [cur]
    if open_run:
        emitted.append(open_run)

    result: list[list[int]] = []
    for group in emitted:
        # A multi-attempt tool run is a RETRY loop only if BOTH hold:
        #   1. at least one attempt FAILED — a repeated-same-input all-success run
        #      (idempotent re-fetch) is not a loop.
        #   2. it spans MORE THAN ONE turn (B3 / NEW-4): a genuine agentic retry
        #      interleaves a model turn before each retry, so the attempts have
        #      DIFFERENT immediate (turn) parents. A SAME-TURN sequential fan-out
        #      (>=2 same-input calls in ONE turn, e.g. the model emitting several
        #      tool calls at once) shares one turn parent and is NOT a retry loop —
        #      flagging it would tell a builder "you have a retry loop" when they
        #      do not. Abstain (conservative false-negative) when turn ids are
        #      missing, so we never falsely collapse on unknown structure.
        if (
            len(group) >= 2
            and any(a.has_error for a in group)
            and _spans_multiple_turns(group)
        ):
            result.append([a.index for a in group])
        else:
            result.extend([a.index] for a in group)
    # Preserve original capture order across all emitted groups by first index.
    result.sort(key=lambda g: g[0])
    return result
