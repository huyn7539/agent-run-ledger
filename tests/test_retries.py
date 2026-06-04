"""Task 45 / NEW-4 — trace-derived retry detection.

The gate under everything: can ARL DETECT a retry loop from a REAL trace, where
the SDK emits NO retry_count field? A real loop is N repeated same-target tool
spans. The HARD requirement (operator): distinguish a genuine loop (same input +
repeated failure + temporal adjacency) from legitimate repeated tool calls. A
false positive makes the demo WRONG on a real trace, so the negative test is the
load-bearing one.

Two layers tested:
  - core/retries.py: a provider-neutral pure grouper over AttemptFacts (so the
    correctness-critical AND-rule is unit-testable without the OpenAI span path).
  - adapters/openai.py: the end-to-end path that fingerprints raw input transiently
    and collapses repeated function spans into one StepRecord(retry_count=N-1).
"""

from __future__ import annotations

from agent_run_ledger.adapters.openai import bundle_from_recorded_trace
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.retries import AttemptFacts, collapse_retry_groups


def _attempt(
    index: int,
    *,
    name: str = "crm.lookup",
    parent_id: str = "p1",
    fingerprint: str | None = "fp_same",
    has_error: bool = True,
    error_class: str | None = "Timeout",
    started_at: str | None = None,
    ended_at: str | None = None,
) -> AttemptFacts:
    base = index * 2
    return AttemptFacts(
        index=index,
        name=name,
        span_kind="function",
        parent_id=parent_id,
        started_at=started_at or f"2026-05-31T10:00:{base:02d}Z",
        ended_at=ended_at or f"2026-05-31T10:00:{base + 1:02d}Z",
        has_error=has_error,
        error_class=error_class,
        input_fingerprint=fingerprint,
    )


# --- core grouper: the discriminating logic in isolation ----------------------


def test_genuine_loop_collapses_to_one_group_retry_count_two() -> None:
    """3 attempts, same name/parent/fingerprint, sequential, all failing -> ONE
    group with member indices [0,1,2] (retry_count = 2 downstream)."""
    attempts = [_attempt(0), _attempt(1), _attempt(2)]

    groups = collapse_retry_groups(attempts)

    assert len(groups) == 1
    assert groups[0] == [0, 1, 2]


def test_legitimate_repetition_different_inputs_does_not_collapse() -> None:
    """THE DISCRIMINATOR. Same tool called 3x on 3 DIFFERENT inputs, no errors —
    legitimate work, NOT a retry loop. Must NOT collapse. A name+adjacency-only
    rule would wrongly merge these (the NEW-4 false-positive failure mode)."""
    attempts = [
        _attempt(0, fingerprint="fp_a", has_error=False, error_class=None),
        _attempt(1, fingerprint="fp_b", has_error=False, error_class=None),
        _attempt(2, fingerprint="fp_c", has_error=False, error_class=None),
    ]

    groups = collapse_retry_groups(attempts)

    # every attempt is its own singleton group -> retry_count 0 everywhere
    assert groups == [[0], [1], [2]]


def test_same_input_but_all_success_does_not_collapse() -> None:
    """Same input repeated but with NO failure is not a retry loop (idempotent
    re-fetch, cache warmup, etc). The >=1-error gate must hold."""
    attempts = [
        _attempt(0, has_error=False, error_class=None),
        _attempt(1, has_error=False, error_class=None),
        _attempt(2, has_error=False, error_class=None),
    ]

    groups = collapse_retry_groups(attempts)

    assert groups == [[0], [1], [2]]


def test_interleaved_other_tool_breaks_the_run() -> None:
    """A different-target span between two same-target attempts breaks adjacency:
    the two attempts are NOT consecutive, so no collapse."""
    attempts = [
        _attempt(0, name="crm.lookup"),
        _attempt(1, name="other.tool", fingerprint="fp_other"),
        _attempt(2, name="crm.lookup"),
    ]

    groups = collapse_retry_groups(attempts)

    assert groups == [[0], [1], [2]]


def test_overlapping_windows_do_not_collapse() -> None:
    """Concurrent attempts (parallel fan-out sharing a parent) overlap in time —
    that is parallelism, not sequential retry. Conservative: do not collapse."""
    attempts = [
        _attempt(0, started_at="2026-05-31T10:00:00Z", ended_at="2026-05-31T10:00:05Z"),
        _attempt(1, started_at="2026-05-31T10:00:01Z", ended_at="2026-05-31T10:00:06Z"),
    ]

    groups = collapse_retry_groups(attempts)

    assert groups == [[0], [1]]


def test_different_parent_does_not_collapse() -> None:
    attempts = [_attempt(0, parent_id="p1"), _attempt(1, parent_id="p2")]

    groups = collapse_retry_groups(attempts)

    assert groups == [[0], [1]]


def test_missing_fingerprint_does_not_collapse() -> None:
    """Input not captured -> we cannot prove same-input -> abstain (false-negative
    by design, never a false-positive)."""
    attempts = [
        _attempt(0, fingerprint=None),
        _attempt(1, fingerprint=None),
        _attempt(2, fingerprint=None),
    ]

    groups = collapse_retry_groups(attempts)

    assert groups == [[0], [1], [2]]


def test_non_function_spans_never_collapse() -> None:
    """Response spans have no name/input -> cannot distinguish genuine vs
    legitimate -> abstain (only function/tool spans are eligible)."""
    attempts = [
        AttemptFacts(0, "resp", "response", "p1", "2026-05-31T10:00:00Z", "2026-05-31T10:00:01Z", True, "Timeout", None),
        AttemptFacts(1, "resp", "response", "p1", "2026-05-31T10:00:02Z", "2026-05-31T10:00:03Z", True, "Timeout", None),
    ]

    groups = collapse_retry_groups(attempts)

    assert groups == [[0], [1]]


# --- adapter end-to-end: real span shape -> derived retry_count ---------------


def _function_span(
    span_id: str,
    *,
    tool_input: str,
    started_at: str,
    ended_at: str,
    parent_id: str = "span_agent_root",
    error: dict | None = None,
) -> dict:
    return {
        "object": "trace.span",
        "id": span_id,
        "trace_id": "trace_retry_0123456789ab",
        "parent_id": parent_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "span_data": {
            "type": "function",
            "name": "crm.lookup",
            "input": tool_input,
            "output": None,
        },
        "error": error,
    }


def _trace(spans: list[dict]) -> dict:
    return {
        "trace": {
            "trace_id": "trace_retry_0123456789ab",
            "workflow_name": "retry-loop-agent",
            "started_at": "2026-05-31T10:00:00Z",
            "ended_at": "2026-05-31T10:00:20Z",
        },
        "spans": spans,
    }


# The REAL SDK function-tool SpanError shape (verified against tool.py:1428 /
# run_internal/tool_actions.py:141): message is a GENERIC constant and data.error
# is free text. There is NO bounded exception-type token, so a live tool error
# HONESTLY classifies as "Other" — the error-class chokepoint refuses to parse
# free text (that would re-open the content-leak vector Task 44 closed). Bounded
# class precision needs app instrumentation; disclosed as a receipt Limit.
_TOOL_ERROR = {"message": "Error running tool", "data": {"tool_name": "crm.lookup", "error": "details redacted"}}


def test_adapter_derives_retry_loop_from_repeated_function_spans() -> None:
    """END-TO-END NEW-4: 3 repeated failing crm.lookup spans with the SAME input
    and NO app-supplied retry_count -> ARL DERIVES retry_count=2 -> 1 prescription,
    high severity. ARL detected what the SDK never labeled."""
    same_input = "lookup customer 42"
    bundle = bundle_from_recorded_trace(
        _trace(
            [
                _function_span("s1", tool_input=same_input, started_at="2026-05-31T10:00:00Z", ended_at="2026-05-31T10:00:02Z", error=_TOOL_ERROR),
                _function_span("s2", tool_input=same_input, started_at="2026-05-31T10:00:02Z", ended_at="2026-05-31T10:00:04Z", error=_TOOL_ERROR),
                _function_span("s3", tool_input=same_input, started_at="2026-05-31T10:00:04Z", ended_at="2026-05-31T10:00:06Z", error=_TOOL_ERROR),
            ]
        ),
        model="gpt-4o-mini",
    )

    # 3 attempts collapsed into ONE step
    assert len(bundle.steps) == 1
    step = bundle.steps[0]
    assert step.retry_count == 2  # N-1 additional attempts
    # A live tool error is present but free-text -> classifies as "Other" (the
    # honest result). The class is truthy, so severity is still "high".
    assert step.error_class == "Other"
    assert step.span_kind == "function"

    prescriptions = analyze_bundle(bundle)
    assert len(prescriptions) == 1
    assert prescriptions[0].severity == "high"


def test_adapter_does_not_derive_retry_for_legitimate_repetition() -> None:
    """END-TO-END NEGATIVE: same tool, 3 DIFFERENT inputs, all succeeding ->
    NO collapse -> retry_count 0 everywhere -> ZERO prescriptions. The demo must
    not invent a retry loop where there is none."""
    bundle = bundle_from_recorded_trace(
        _trace(
            [
                _function_span("s1", tool_input="customer 1", started_at="2026-05-31T10:00:00Z", ended_at="2026-05-31T10:00:01Z"),
                _function_span("s2", tool_input="customer 2", started_at="2026-05-31T10:00:01Z", ended_at="2026-05-31T10:00:02Z"),
                _function_span("s3", tool_input="customer 3", started_at="2026-05-31T10:00:02Z", ended_at="2026-05-31T10:00:03Z"),
            ]
        ),
        model="gpt-4o-mini",
    )

    assert len(bundle.steps) == 3
    assert all(step.retry_count == 0 for step in bundle.steps)
    assert analyze_bundle(bundle) == []


def test_adapter_sums_cost_and_tokens_across_collapsed_attempts() -> None:
    """The collapsed step must SUM tokens/cost across attempts — the wasted-cost
    estimate divides cost_usd by total attempts, so keeping one attempt's cost is
    wrong by a factor of N."""
    same_input = "lookup customer 42"
    spans = []
    for i in range(3):
        s = _function_span(
            f"s{i}",
            tool_input=same_input,
            started_at=f"2026-05-31T10:00:{i * 2:02d}Z",
            ended_at=f"2026-05-31T10:00:{i * 2 + 1:02d}Z",
            error=_TOOL_ERROR,
        )
        s["span_data"]["data"] = {"cost_usd": 0.01}  # provider-reported per attempt
        spans.append(s)

    bundle = bundle_from_recorded_trace(_trace(spans), model="gpt-4o-mini")

    assert len(bundle.steps) == 1
    # 3 attempts x 0.01 summed
    assert bundle.steps[0].provider_reported_cost_usd == 0.03


def test_explicit_app_supplied_retry_count_still_works() -> None:
    """SCOPE GUARD: the pre-labeled custom-span path is NOT removed. An app that
    instruments retry_count directly still yields the loop (Task-44 behavior)."""
    recorded = {
        "trace": {
            "trace_id": "trace_explicit_retry",
            "workflow_name": "explicit-agent",
            "started_at": "2026-05-31T10:00:00Z",
            "ended_at": "2026-05-31T10:00:10Z",
        },
        "spans": [
            {
                "span_id": "span_custom",
                "started_at": "2026-05-31T10:00:00Z",
                "ended_at": "2026-05-31T10:00:10Z",
                "span_data": {
                    "type": "custom",
                    "name": "demo.flaky_tool",
                    "data": {"arl_step_type": "tool", "retry_count": 3, "cost_usd": 0.04},
                },
            }
        ],
    }

    bundle = bundle_from_recorded_trace(recorded, model="gpt-4o-mini")

    assert any(step.retry_count == 3 for step in bundle.steps)
    assert len(analyze_bundle(bundle)) == 1
