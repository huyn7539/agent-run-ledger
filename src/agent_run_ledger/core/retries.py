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

    ``input_fingerprint`` is a digest of the raw tool input (or None when input
    was not captured). It is used ONLY to test same-vs-different input; it carries
    no recoverable content and is never persisted.
    """

    index: int
    name: str
    span_kind: str | None
    parent_id: str | None
    started_at: str
    ended_at: str
    has_error: bool
    error_class: str | None
    input_fingerprint: str | None


def _is_retry_continuation(prev: AttemptFacts, cur: AttemptFacts) -> bool:
    """True iff *cur* is another attempt of the SAME operation as *prev*.

    ALL conditions must hold (any failure -> not a continuation -> the run ends):
      - both are function/tool spans (only these carry name+input to compare)
      - identical tool name
      - identical parent (same call site)
      - both have a captured input fingerprint, and they are EQUAL (same input)
      - sequential, non-overlapping in time (cur starts at/after prev ends);
        overlap means parallelism, not retry
    """
    if prev.span_kind != "function" or cur.span_kind != "function":
        return False
    if prev.name != cur.name:
        return False
    if prev.parent_id != cur.parent_id:
        return False
    if prev.input_fingerprint is None or cur.input_fingerprint is None:
        return False
    if prev.input_fingerprint != cur.input_fingerprint:
        return False
    # Sequential, non-overlapping: cur must start at or after prev ended. ISO-8601
    # Zulu timestamps sort lexically, so a string compare is a correct time
    # compare. cur.started_at < prev.ended_at means the windows overlap ->
    # parallelism (concurrent fan-out), NOT retry -> reject (conservative).
    if cur.started_at < prev.ended_at:
        return False
    return True


def collapse_retry_groups(attempts: list[AttemptFacts]) -> list[list[int]]:
    """Group *attempts* (in capture order) into runs that are retry loops.

    Returns a list of groups, each a list of the original ``index`` values in
    order. A group of length >= 2 with at least one error is a retry loop;
    everything else stays a singleton. Callers sort attempts deterministically
    (by ``(started_at, index)``) BEFORE calling — this function trusts the given
    order for adjacency.
    """
    if not attempts:
        return []

    groups: list[list[AttemptFacts]] = [[attempts[0]]]
    for cur in attempts[1:]:
        prev = groups[-1][-1]
        if _is_retry_continuation(prev, cur):
            groups[-1].append(cur)
        else:
            groups.append([cur])

    result: list[list[int]] = []
    for group in groups:
        # A multi-attempt run is only a RETRY loop if at least one attempt failed.
        # A repeated-same-input all-success run (idempotent re-fetch) is not a loop
        # -> split back into singletons so it derives retry_count 0.
        if len(group) >= 2 and any(a.has_error for a in group):
            result.append([a.index for a in group])
        else:
            result.extend([a.index] for a in group)
    return result
