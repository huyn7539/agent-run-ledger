"""Task 45 / NEW-4 — trace-derived retry detection.

The gate under everything: can ARL DETECT a retry loop from a REAL trace, where
the SDK emits NO retry_count field? A real loop is N repeated same-scope, same-
input tool attempts with >=1 failure. The HARD requirement (operator): distinguish
a genuine loop from legitimate repeated tool calls. A false positive makes the
demo WRONG on a real trace, so the negative tests are load-bearing.

Two layers tested:
  - core/retries.py: a provider-neutral pure grouper over AttemptFacts (the
    correctness-critical AND-rule, unit-testable without the OpenAI span path).
  - adapters/openai.py + prescriptions.derive_retry_steps: the end-to-end path —
    the base stores one raw step per span; the collapse is computed ON READ.

NOTE on grouping key: retries are grouped by ``retry_scope`` (the stable agent-
span ancestor), NOT the immediate parent — a real agentic retry spans turns, so
the immediate parent differs. Cross-turn realism is covered in
tests/test_cross_turn_retry.py; here we use a fixed scope to isolate the AND-rule.
"""

from __future__ import annotations

from agent_run_ledger.adapters.openai import bundle_from_recorded_trace
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.retries import AttemptFacts, collapse_retry_groups


def _attempt(
    index: int,
    *,
    name: str = "crm.lookup",
    scope: str = "agent_1",
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
        retry_scope=scope,
        started_at=started_at or f"2026-05-31T10:00:{base:02d}Z",
        ended_at=ended_at or f"2026-05-31T10:00:{base + 1:02d}Z",
        has_error=has_error,
        error_class=error_class,
        input_fingerprint=fingerprint,
    )


def _model_attempt(index: int, *, scope: str = "agent_1", started_at: str, ended_at: str) -> AttemptFacts:
    """A response/model-turn span: not a tool, no input fingerprint."""
    return AttemptFacts(
        index=index,
        name=f"response_{index}",
        span_kind="response",
        retry_scope=scope,
        started_at=started_at,
        ended_at=ended_at,
        has_error=False,
        error_class=None,
        input_fingerprint=None,
    )


# --- core grouper: the discriminating logic in isolation ----------------------


def test_genuine_loop_collapses_to_one_group_retry_count_two() -> None:
    attempts = [_attempt(0), _attempt(1), _attempt(2)]
    assert collapse_retry_groups(attempts) == [[0, 1, 2]]


def test_legitimate_repetition_different_inputs_does_not_collapse() -> None:
    """THE DISCRIMINATOR. Same tool, same scope, 3 DIFFERENT inputs, no errors —
    legitimate work, NOT a retry loop. Must NOT collapse."""
    attempts = [
        _attempt(0, fingerprint="fp_a", has_error=False, error_class=None),
        _attempt(1, fingerprint="fp_b", has_error=False, error_class=None),
        _attempt(2, fingerprint="fp_c", has_error=False, error_class=None),
    ]
    assert collapse_retry_groups(attempts) == [[0], [1], [2]]


def test_same_input_but_all_success_does_not_collapse() -> None:
    """Same input repeated with NO failure is not a retry loop (idempotent
    re-fetch). The >=1-error gate must hold."""
    attempts = [
        _attempt(0, has_error=False, error_class=None),
        _attempt(1, has_error=False, error_class=None),
        _attempt(2, has_error=False, error_class=None),
    ]
    assert collapse_retry_groups(attempts) == [[0], [1], [2]]


def test_interleaved_model_turns_do_not_break_a_tool_retry_loop() -> None:
    """THE REAL AGENTIC RETRY SHAPE: a model/response turn span between same-scope
    tool attempts is a turn boundary; it must NOT break the run. The 3 tool
    attempts collapse; the response spans stay singletons."""
    attempts = [
        _model_attempt(0, started_at="2026-05-31T10:00:00Z", ended_at="2026-05-31T10:00:01Z"),
        _attempt(1, started_at="2026-05-31T10:00:01Z", ended_at="2026-05-31T10:00:02Z"),
        _model_attempt(2, started_at="2026-05-31T10:00:02Z", ended_at="2026-05-31T10:00:03Z"),
        _attempt(3, started_at="2026-05-31T10:00:03Z", ended_at="2026-05-31T10:00:04Z"),
        _model_attempt(4, started_at="2026-05-31T10:00:04Z", ended_at="2026-05-31T10:00:05Z"),
        _attempt(5, started_at="2026-05-31T10:00:05Z", ended_at="2026-05-31T10:00:06Z"),
    ]
    groups = collapse_retry_groups(attempts)
    assert [1, 3, 5] in groups
    assert [0] in groups and [2] in groups and [4] in groups
    assert len(groups) == 4


def test_interleaved_different_tool_breaks_the_run() -> None:
    """A DIFFERENT tool between same-target attempts is real interleaved work, not
    a retry — it breaks the run (only model-turn spans are tolerated between)."""
    attempts = [
        _attempt(0, name="crm.lookup"),
        _attempt(1, name="other.tool", fingerprint="fp_other"),
        _attempt(2, name="crm.lookup"),
    ]
    assert collapse_retry_groups(attempts) == [[0], [1], [2]]


def test_overlapping_windows_do_not_collapse() -> None:
    """Concurrent attempts (parallel fan-out) overlap in time — parallelism, not
    sequential retry. Conservative: do not collapse."""
    attempts = [
        _attempt(0, started_at="2026-05-31T10:00:00Z", ended_at="2026-05-31T10:00:05Z"),
        _attempt(1, started_at="2026-05-31T10:00:01Z", ended_at="2026-05-31T10:00:06Z"),
    ]
    assert collapse_retry_groups(attempts) == [[0], [1]]


def test_different_scope_does_not_collapse() -> None:
    """Different retry_scope (e.g. a handoff to another agent) -> not a retry of
    the same call site -> no collapse."""
    attempts = [_attempt(0, scope="agent_1"), _attempt(1, scope="agent_2")]
    assert collapse_retry_groups(attempts) == [[0], [1]]


def test_missing_fingerprint_does_not_collapse() -> None:
    """Input not captured -> cannot prove same-input -> abstain (false-negative by
    design, never a false-positive)."""
    attempts = [_attempt(0, fingerprint=None), _attempt(1, fingerprint=None), _attempt(2, fingerprint=None)]
    assert collapse_retry_groups(attempts) == [[0], [1], [2]]


def test_missing_scope_does_not_collapse() -> None:
    """No resolvable scope -> abstain (never falsely group)."""
    attempts = [_attempt(0, scope=None), _attempt(1, scope=None)]  # type: ignore[arg-type]
    assert collapse_retry_groups(attempts) == [[0], [1]]


def test_non_function_spans_never_collapse() -> None:
    attempts = [
        AttemptFacts(0, "resp", "response", "agent_1", "2026-05-31T10:00:00Z", "2026-05-31T10:00:01Z", True, "Timeout", None),
        AttemptFacts(1, "resp", "response", "agent_1", "2026-05-31T10:00:02Z", "2026-05-31T10:00:03Z", True, "Timeout", None),
    ]
    assert collapse_retry_groups(attempts) == [[0], [1]]


# --- adapter + on-read end-to-end: real span shape -> derived retry_count ------

# The REAL SDK function-tool SpanError shape (tool.py:1428): message is a generic
# constant and data.error is free text -> a live tool error HONESTLY classifies as
# "Other" (the chokepoint refuses to parse free text). All these spans share one
# agent scope so the AND-rule's scope key is satisfied via the agent ancestor.
_TOOL_ERROR = {"message": "Error running tool", "data": {"tool_name": "crm.lookup", "error": "details redacted"}}


def _agent_span():
    return {
        "object": "trace.span", "id": "agent_root", "trace_id": "trace_retry_0123456789ab",
        "parent_id": None, "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:20Z",
        "span_data": {"type": "agent", "name": "Support Agent"}, "error": None,
    }


def _function_span(span_id, *, tool_input, started_at, ended_at, parent_id="agent_root", error=None):
    return {
        "object": "trace.span", "id": span_id, "trace_id": "trace_retry_0123456789ab",
        "parent_id": parent_id, "started_at": started_at, "ended_at": ended_at,
        "span_data": {"type": "function", "name": "crm.lookup", "input": tool_input, "output": None},
        "error": error,
    }


def _response_span(span_id, *, started_at, ended_at, parent_id="agent_root"):
    return {
        "object": "trace.span", "id": span_id, "trace_id": "trace_retry_0123456789ab",
        "parent_id": parent_id, "started_at": started_at, "ended_at": ended_at,
        "span_data": {"type": "response", "response_id": f"resp_{span_id}", "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}},
        "error": None,
    }


def _trace(spans):
    return {
        "trace": {"trace_id": "trace_retry_0123456789ab", "workflow_name": "retry-loop-agent", "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:20Z"},
        "spans": [_agent_span(), *spans],
    }


def test_adapter_derives_retry_loop_from_repeated_function_spans() -> None:
    """END-TO-END NEW-4: 3 repeated failing crm.lookup spans, SAME input, NO
    app-supplied retry_count -> ARL DERIVES the loop on read -> 1 prescription."""
    same = "lookup customer 42"
    bundle = bundle_from_recorded_trace(
        _trace([
            _function_span("s1", tool_input=same, started_at="2026-05-31T10:00:00Z", ended_at="2026-05-31T10:00:02Z", error=_TOOL_ERROR),
            _function_span("s2", tool_input=same, started_at="2026-05-31T10:00:02Z", ended_at="2026-05-31T10:00:04Z", error=_TOOL_ERROR),
            _function_span("s3", tool_input=same, started_at="2026-05-31T10:00:04Z", ended_at="2026-05-31T10:00:06Z", error=_TOOL_ERROR),
        ]),
        model="gpt-4o-mini",
    )
    prescriptions = analyze_bundle(bundle)
    assert len(prescriptions) == 1
    assert prescriptions[0].severity == "high"


def test_adapter_derives_retry_loop_from_real_interleaved_agentic_shape() -> None:
    """END-TO-END, REAL SHAPE: the agentic retry loop interleaves a response span
    before each tool retry. ARL still derives 1 prescription on read."""
    same = "lookup customer 42"
    bundle = bundle_from_recorded_trace(
        _trace([
            _response_span("r1", started_at="2026-05-31T10:00:00Z", ended_at="2026-05-31T10:00:01Z"),
            _function_span("f1", tool_input=same, started_at="2026-05-31T10:00:01Z", ended_at="2026-05-31T10:00:02Z", error=_TOOL_ERROR),
            _response_span("r2", started_at="2026-05-31T10:00:02Z", ended_at="2026-05-31T10:00:03Z"),
            _function_span("f2", tool_input=same, started_at="2026-05-31T10:00:03Z", ended_at="2026-05-31T10:00:04Z", error=_TOOL_ERROR),
            _response_span("r3", started_at="2026-05-31T10:00:04Z", ended_at="2026-05-31T10:00:05Z"),
            _function_span("f3", tool_input=same, started_at="2026-05-31T10:00:05Z", ended_at="2026-05-31T10:00:06Z", error=_TOOL_ERROR),
        ]),
        model="gpt-4o-mini",
    )
    prescriptions = analyze_bundle(bundle)
    assert len(prescriptions) == 1
    assert prescriptions[0].severity == "high"


def test_adapter_does_not_derive_retry_for_legitimate_repetition() -> None:
    """END-TO-END NEGATIVE: same tool, 3 DIFFERENT inputs, all succeeding -> NO
    collapse -> ZERO prescriptions. The demo must not invent a retry loop."""
    bundle = bundle_from_recorded_trace(
        _trace([
            _function_span("s1", tool_input="customer 1", started_at="2026-05-31T10:00:00Z", ended_at="2026-05-31T10:00:01Z"),
            _function_span("s2", tool_input="customer 2", started_at="2026-05-31T10:00:01Z", ended_at="2026-05-31T10:00:02Z"),
            _function_span("s3", tool_input="customer 3", started_at="2026-05-31T10:00:02Z", ended_at="2026-05-31T10:00:03Z"),
        ]),
        model="gpt-4o-mini",
    )
    assert analyze_bundle(bundle) == []


def test_derived_retry_step_sums_cost_across_attempts() -> None:
    """The on-read collapsed step SUMS cost across attempts — the wasted-cost
    estimate divides by attempt count, so keeping one attempt's cost is wrong."""
    from agent_run_ledger.core.prescriptions import derive_retry_steps

    same = "lookup customer 42"
    spans = []
    for i in range(3):
        s = _function_span(f"s{i}", tool_input=same, started_at=f"2026-05-31T10:00:{i * 2:02d}Z", ended_at=f"2026-05-31T10:00:{i * 2 + 1:02d}Z", error=_TOOL_ERROR)
        s["span_data"]["data"] = {"cost_usd": 0.01}
        spans.append(s)
    bundle = bundle_from_recorded_trace(_trace(spans), model="gpt-4o-mini")

    collapsed_fn = [s for s in derive_retry_steps(bundle) if s.span_kind == "function"]
    assert len(collapsed_fn) == 1
    assert collapsed_fn[0].provider_reported_cost_usd == 0.03


def test_explicit_app_supplied_retry_count_still_works() -> None:
    """SCOPE GUARD: the pre-labeled custom-span path is NOT removed."""
    recorded = {
        "trace": {"trace_id": "trace_explicit_retry", "workflow_name": "explicit-agent", "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:10Z"},
        "spans": [
            {"span_id": "span_custom", "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:10Z",
             "span_data": {"type": "custom", "name": "demo.flaky_tool", "data": {"arl_step_type": "tool", "retry_count": 3, "cost_usd": 0.04}}},
        ],
    }
    bundle = bundle_from_recorded_trace(recorded, model="gpt-4o-mini")
    assert any(step.retry_count == 3 for step in bundle.steps)
    assert len(analyze_bundle(bundle)) == 1
