"""C1 + H1 — the retry loop is detected on the REAL cross-turn span topology, and
the collapse is a JUDGMENT computed ON READ (never baked into the immutable base).

C1 (correctness): a real OpenAI Agents SDK retry loop spans MULTIPLE turns. Each
turn opens a fresh turn_span (parent of that turn's function/response spans) under
a stable agent_span. So across turns the retried tool spans have DIFFERENT
immediate parents (turn spans) but the SAME agent-span ancestor. Verified against
the SDK: provider.py:391 (parent_id = current span) + run.py:1045-1050 (agent_span
marked current) + run.py:1166-1278 (per-turn turn_span marked current then reset).
The detector must key on the agent-span ANCESTOR (retry_scope), not the immediate
turn parent — else it no-ops on every real multi-turn retry.

H1 (boundary): the raw per-attempt spans must persist as FACTS; retry_count is
DERIVED on read. The immutable base stores one StepRecord per span; provenance is
hashed over the raw spans, so a future detector fix can re-derive from the corpus.
"""

from __future__ import annotations

from agent_run_ledger.adapters.openai import bundle_from_recorded_trace
from agent_run_ledger.core.prescriptions import analyze_bundle, derive_retry_steps
from agent_run_ledger.core.storage import load_bundle, save_bundle


# A REAL nested span tree: agent_span -> per-turn turn_span -> function/response.
# The adapter must WALK function -> turn -> agent to resolve the stable retry_scope.
def _agent_span():
    return {
        "object": "trace.span", "id": "agent_1", "trace_id": "t",
        "parent_id": None, "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:12Z",
        "span_data": {"type": "agent", "name": "Support Agent"}, "error": None,
    }


def _turn_span(tid, t0, t1):
    return {
        "object": "trace.span", "id": tid, "trace_id": "t",
        "parent_id": "agent_1", "started_at": t0, "ended_at": t1,
        "span_data": {"type": "custom", "name": f"turn_{tid}"}, "error": None,
    }


def _fn_span(sid, parent, t0, t1, *, tool_input="{\"id\": 42}", error=True):
    return {
        "object": "trace.span", "id": sid, "trace_id": "t",
        "parent_id": parent, "started_at": t0, "ended_at": t1,
        "span_data": {"type": "function", "name": "crm.lookup", "input": tool_input},
        "error": ({"message": "Error running tool", "data": {"tool_name": "crm.lookup", "error": "redacted"}} if error else None),
    }


def _resp_span(sid, parent, t0, t1):
    return {
        "object": "trace.span", "id": sid, "trace_id": "t",
        "parent_id": parent, "started_at": t0, "ended_at": t1,
        "span_data": {"type": "response", "response_id": "r" + sid, "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}},
        "error": None,
    }


def _cross_turn_trace():
    """3 turns, each: turn_span(parent=agent) -> response + function(parent=turn).
    The 3 crm.lookup spans have 3 DIFFERENT turn parents but the SAME agent ancestor."""
    return {
        "trace": {"trace_id": "t", "workflow_name": "support-agent", "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:12Z"},
        "spans": [
            _agent_span(),
            _turn_span("turn1", "2026-05-31T10:00:00Z", "2026-05-31T10:00:04Z"),
            _resp_span("resp1", "turn1", "2026-05-31T10:00:00Z", "2026-05-31T10:00:01Z"),
            _fn_span("fn1", "turn1", "2026-05-31T10:00:01Z", "2026-05-31T10:00:03Z"),
            _turn_span("turn2", "2026-05-31T10:00:04Z", "2026-05-31T10:00:08Z"),
            _resp_span("resp2", "turn2", "2026-05-31T10:00:04Z", "2026-05-31T10:00:05Z"),
            _fn_span("fn2", "turn2", "2026-05-31T10:00:05Z", "2026-05-31T10:00:07Z"),
            _turn_span("turn3", "2026-05-31T10:00:08Z", "2026-05-31T10:00:12Z"),
            _resp_span("resp3", "turn3", "2026-05-31T10:00:08Z", "2026-05-31T10:00:09Z"),
            _fn_span("fn3", "turn3", "2026-05-31T10:00:09Z", "2026-05-31T10:00:11Z"),
        ],
    }


def test_cross_turn_retry_loop_is_detected_via_agent_ancestor() -> None:
    """C1: the 3 crm.lookup spans across 3 turns (different turn parents, same agent
    ancestor) are recognized as ONE retry loop -> 1 prescription. The fix keys on
    the resolved agent-span ancestor, not the immediate turn parent."""
    bundle = bundle_from_recorded_trace(_cross_turn_trace(), model="gpt-4o-mini")

    prescriptions = analyze_bundle(bundle)
    assert len(prescriptions) == 1
    assert prescriptions[0].severity == "high"


def test_raw_attempt_spans_persist_in_the_base_collapse_is_on_read() -> None:
    """H1: the immutable base stores ONE StepRecord PER SPAN (raw facts retained);
    the retry collapse is computed on READ via derive_retry_steps. The stored
    base must NOT pre-collapse the 3 function attempts into one."""
    bundle = bundle_from_recorded_trace(_cross_turn_trace(), model="gpt-4o-mini")

    # base: every span is its own step (3 function + 3 response + 3 turn + 1 agent = 10)
    assert len(bundle.steps) == 10
    fn_steps = [s for s in bundle.steps if s.span_kind == "function"]
    assert len(fn_steps) == 3
    # raw attempts each carry retry_count 0 — the count is NOT baked in
    assert all(s.retry_count == 0 for s in fn_steps)

    # on READ, the collapse derives the loop
    collapsed = derive_retry_steps(bundle)
    collapsed_fn = [s for s in collapsed if s.span_kind == "function"]
    assert len(collapsed_fn) == 1
    assert collapsed_fn[0].retry_count == 2


def test_retry_scope_and_fingerprint_round_trip_through_storage(tmp_path) -> None:
    """H1: the new bounded facts (retry_scope, input_fingerprint) persist + reload,
    so a future detector fix can re-derive the loop from the stored corpus."""
    bundle = bundle_from_recorded_trace(_cross_turn_trace(), model="gpt-4o-mini")
    db = tmp_path / "ledger.sqlite"
    save_bundle(db, bundle)
    loaded = load_bundle(db, bundle.run.id)

    # detection still works after a store/reload (re-derived on read)
    assert len(analyze_bundle(loaded)) == 1
    fn_steps = [s for s in loaded.steps if s.span_kind == "function"]
    # the retry_scope (agent ancestor) is preserved + identical across attempts
    scopes = {s.retry_scope for s in fn_steps}
    assert scopes == {"agent_1"}
    # the input fingerprint persisted + identical (same input across attempts)
    fps = {s.input_fingerprint for s in fn_steps}
    assert len(fps) == 1 and next(iter(fps))


def test_different_agent_scope_does_not_collapse() -> None:
    """C1 precision: a handoff to a DIFFERENT agent calling the same tool on the
    same input is NOT a retry loop — different agent ancestor -> no collapse."""
    trace = _cross_turn_trace()
    # re-parent turn3 under a different agent (a handoff)
    trace["spans"].append({
        "object": "trace.span", "id": "agent_2", "trace_id": "t", "parent_id": None,
        "started_at": "2026-05-31T10:00:08Z", "ended_at": "2026-05-31T10:00:12Z",
        "span_data": {"type": "agent", "name": "Escalation Agent"}, "error": None,
    })
    for s in trace["spans"]:
        if s["id"] == "turn3":
            s["parent_id"] = "agent_2"

    bundle = bundle_from_recorded_trace(trace, model="gpt-4o-mini")
    collapsed_fn = [s for s in derive_retry_steps(bundle) if s.span_kind == "function"]
    # fn1+fn2 (agent_1) collapse to retry_count=1; fn3 (agent_2) stays separate
    counts = sorted(s.retry_count for s in collapsed_fn)
    assert counts == [0, 1]
