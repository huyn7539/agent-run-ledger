"""Step 3 — the RepairReceipt (a JUDGMENT computed on read) + L2 static verification.

A RepairReceipt is NOT stored in the immutable base — it is computed from the base
FACTS + the prescription. The proof level is graded HONESTLY:

  L2 (static verification): the repair artifact MECHANICALLY removes the failure
      path WITHOUT a re-run. For retry-cap: an APPLYABLE templated diff that bounds
      attempts to a finite cap -> the unbounded-retry path cannot recur.
  L1 (accepted artifact): a relevant but non-runnable artifact (the config_diff
      fallback, no file/line target) -> relevance, not mechanical removal.
  L0 (diagnostic): no accepted fix.

Constraints under test:
  - templated artifact (not free-form LLM output) — auditable, apply-safe.
  - evidence carries ONLY bounded facts (no raw content / sentinels).
  - honesty: an outcome_delta carries a counter-metric, and limits disclose the
    regression-to-the-mean caveat.
"""

from __future__ import annotations

from agent_run_ledger.adapters.openai import bundle_from_recorded_trace
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.receipt import PROOF_LEVELS, build_receipts


# --- a derived real retry loop (function spans) -> L2 with an applyable diff ---


def _function_span(span_id, *, tool_input, started_at, ended_at, patch_target=None, parent_id="agent_root"):
    data = {}
    if patch_target is not None:
        data["retry_budget_patch_target"] = patch_target
    span_data = {"type": "function", "name": "crm.lookup", "input": tool_input}
    if data:
        span_data["data"] = data
    return {
        "object": "trace.span",
        "id": span_id,
        "trace_id": "trace_receipt_0123456789",
        "parent_id": parent_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "span_data": span_data,
        "error": {"message": "Error running tool", "data": {"tool_name": "crm.lookup", "error": "details redacted"}},
    }


def _turn_span(turn_id, *, started_at, ended_at):
    """A per-turn turn span (parent=agent). The REAL SDK opens a fresh one each turn;
    cross-turn tool retries parent to DIFFERENT turns but share the agent scope (B3)."""
    return {
        "object": "trace.span", "id": turn_id, "trace_id": "trace_receipt_0123456789",
        "parent_id": "agent_root", "started_at": started_at, "ended_at": ended_at,
        "span_data": {"type": "custom", "name": turn_id}, "error": None,
    }


_PATCH_TARGET = {
    "path": "settings/retries.py",
    "before": "CRM_LOOKUP_RETRIES = 5",
    "after": "CRM_LOOKUP_RETRIES = 0",
}


def _loop_trace(patch_target=None):
    return {
        "trace": {
            "trace_id": "trace_receipt_0123456789",
            "workflow_name": "retry-loop-agent",
            "started_at": "2026-05-31T10:00:00Z",
            "ended_at": "2026-05-31T10:00:10Z",
        },
        "spans": [
            # the stable agent ancestor that scopes the retried tool spans
            {"object": "trace.span", "id": "agent_root", "trace_id": "trace_receipt_0123456789", "parent_id": None, "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:10Z", "span_data": {"type": "agent", "name": "Support Agent"}, "error": None},
            # REAL SHAPE (B3): each retry is a new turn -> distinct turn parent per attempt.
            _turn_span("turn_1", started_at="2026-05-31T10:00:00Z", ended_at="2026-05-31T10:00:02Z"),
            _function_span("s1", tool_input="lookup 42", parent_id="turn_1", started_at="2026-05-31T10:00:00Z", ended_at="2026-05-31T10:00:02Z", patch_target=patch_target),
            _turn_span("turn_2", started_at="2026-05-31T10:00:02Z", ended_at="2026-05-31T10:00:04Z"),
            _function_span("s2", tool_input="lookup 42", parent_id="turn_2", started_at="2026-05-31T10:00:02Z", ended_at="2026-05-31T10:00:04Z", patch_target=patch_target),
            _turn_span("turn_3", started_at="2026-05-31T10:00:04Z", ended_at="2026-05-31T10:00:06Z"),
            _function_span("s3", tool_input="lookup 42", parent_id="turn_3", started_at="2026-05-31T10:00:04Z", ended_at="2026-05-31T10:00:06Z", patch_target=patch_target),
        ],
    }


def _receipts_for(trace, model="gpt-4o-mini"):
    bundle = bundle_from_recorded_trace(trace, model=model)
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    return bundle, build_receipts(bundle)


def test_receipt_reaches_l2_with_applyable_retry_cap_diff() -> None:
    """The slice's demo artifact: a derived real retry loop + an applyable
    templated retry-cap diff -> proof_level L2, no re-run."""
    bundle, receipts = _receipts_for(_loop_trace(patch_target=_PATCH_TARGET))

    assert len(receipts) == 1
    r = receipts[0]
    assert r.proof_level == "L2"
    assert r.observed_failure == "retry_loop"
    # claim is graded, not a causal promise
    assert "retry" in r.claim.lower()
    # evidence cites the captured loop facts
    assert any("retry_count" in e for e in r.evidence)
    # the artifact is templated + applyable (unified diff)
    assert r.repair_artifact["templated"] is True
    assert r.repair_artifact["patch_type"] == "unified_diff"
    # next_evidence + limits are populated (honest)
    assert r.next_evidence
    assert r.limits


def test_receipt_grades_l1_when_artifact_is_nonrunnable_fallback() -> None:
    """No file/line target in the trace -> only the config_diff fallback -> the
    receipt must NOT overclaim L2. It honestly grades L1."""
    bundle, receipts = _receipts_for(_loop_trace(patch_target=None))

    assert len(receipts) == 1
    r = receipts[0]
    assert r.proof_level == "L1"
    assert r.repair_artifact["patch_type"] == "config_diff"


def test_proof_level_is_from_the_closed_ladder() -> None:
    _, receipts = _receipts_for(_loop_trace(patch_target=_PATCH_TARGET))
    assert receipts[0].proof_level in PROOF_LEVELS


# --- honesty constraints ------------------------------------------------------


def test_outcome_delta_carries_a_counter_metric() -> None:
    """Constraint 5: an expected cost reduction must ship with a guardrail
    counter-metric so the receipt does not overstate a one-sided benefit."""
    _, receipts = _receipts_for(_loop_trace(patch_target=_PATCH_TARGET))
    impact = receipts[0].outcome_delta

    # a guardrail/counter-metric must be present (e.g. success/latency guardrail)
    assert any("guardrail" in k or "counter" in k for k in impact)


def test_no_false_precise_zero_cost_delta(tmp_path) -> None:
    """HIGH (code+arch): retry cost in an agentic loop accrues on the MODEL/response
    turns, not the tool spans — so the tool-span-derived wasted-cost is ~0. The
    receipt must NOT present a confident precise '-0.0' that reads as 'this fix
    saves nothing'. When wasted cost can't be attributed, the figure is omitted/
    flagged unknown and the limits disclose the attribution gap — never a false 0."""
    _, receipts = _receipts_for(_loop_trace(patch_target=_PATCH_TARGET))
    r = receipts[0]
    impact = r.outcome_delta

    # no misleading precise -0.0 / 0.0 cost-saving claim
    cost = impact.get("estimated_cost_delta_usd")
    assert cost in (None, "not attributable") or (isinstance(cost, (int, float)) and cost < 0), (
        f"misleading cost delta: {cost!r}"
    )
    # the attribution gap is disclosed honestly in limits
    limits_text = " ".join(r.limits).lower()
    assert "cost" in limits_text and ("model" in limits_text or "response" in limits_text or "attribut" in limits_text)


def test_limits_disclose_regression_to_the_mean() -> None:
    """Constraint 5: ARL fires on the WORST runs, which partly improve on their
    own; the limits must disclose that before/after deltas are uncorrected."""
    _, receipts = _receipts_for(_loop_trace(patch_target=_PATCH_TARGET))
    limits_text = " ".join(receipts[0].limits).lower()

    assert "regression" in limits_text and "mean" in limits_text


def test_limits_disclose_model_supplied_when_hint_used() -> None:
    """When the model came from the app hint (no span carried it), the receipt
    discloses it — the cost figure depends on a supplied fact."""
    _, receipts = _receipts_for(_loop_trace(patch_target=_PATCH_TARGET))
    limits_text = " ".join(receipts[0].limits).lower()

    assert "model" in limits_text


# --- the receipt is a content-free egress-shaped object -----------------------


def test_receipt_evidence_does_not_leak_raw_content() -> None:
    """The receipt's evidence/claim/limits fields are a NEW channel. They must
    carry only bounded facts — never the raw tool input or error message."""
    sentinel = "SECRET_CUSTOMER_SSN_123456789"
    trace = _loop_trace(patch_target=_PATCH_TARGET)
    # plant the sentinel in the raw tool input of every attempt
    for span in trace["spans"]:
        span["span_data"]["input"] = f"lookup {sentinel}"

    bundle, receipts = _receipts_for(trace)
    r = receipts[0]
    blob = " ".join(
        [r.claim, r.observed_failure, r.proof_level, r.confidence, *r.evidence, *r.limits, *r.next_evidence, str(r.repair_artifact), str(r.outcome_delta)]
    )

    assert sentinel not in blob


def test_no_receipts_on_clean_run() -> None:
    """ZERO receipts when there is no detected failure (the negative gate)."""
    clean = {
        "trace": {"trace_id": "trace_clean_0123456789", "workflow_name": "clean", "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:03Z"},
        "spans": [
            _function_span("s1", tool_input="a", started_at="2026-05-31T10:00:00Z", ended_at="2026-05-31T10:00:01Z"),
        ],
    }
    # remove the error so it's a clean success
    clean["spans"][0]["error"] = None

    _, receipts = _receipts_for(clean)
    assert receipts == []
