"""Task 53 F3 — an EXPLICIT (app-supplied / imported) retry_count must not earn L2.

L2's sufficiency claim ("the new cap statically removes the OBSERVED loop") is
only honest when ARL itself DERIVED the observed count by collapsing raw
attempts on read. An explicit ``retry_count`` is an assertion by the app or the
imported file — a forged neutral import can stamp ``retry_count=99`` on a
single step, write matching evidence, attach a valid cap diff, and (pre-fix)
walk to an L2 accusation-grade receipt. Same spoof class the
``adapter_provenanced`` bit closed for artifact receipts (a71f44c), applied to
the retry lane.

Fail-closed design under test:
  * ``StepRecord.retry_count_source`` defaults to "explicit"; ONLY the on-read
    collapse (``prescriptions._collapse_steps``) sets "derived".
  * The field is judgment-view-only: ``to_dict`` never writes it and
    ``from_dict`` never reads it, so a file cannot claim a derived count.
  * Receipt grading caps explicit-count retry receipts at L1 and disclosed in
    limits; derived counts grade L0-L2 exactly as before.
"""

from __future__ import annotations

from agent_run_ledger.core.models import (
    PrescriptionRecord,
    RunRecord,
    StepRecord,
    TraceBundle,
)
from agent_run_ledger.core.prescriptions import derive_retry_steps
from agent_run_ledger.core.receipt import build_receipts

_CAP_DIFF = (
    "--- a/settings/retries.py\n"
    "+++ b/settings/retries.py\n"
    "@@ -1 +1 @@\n"
    "-CRM_LOOKUP_RETRIES = 5\n"
    "+CRM_LOOKUP_RETRIES = 0\n"
)


def _run(run_id: str) -> RunRecord:
    return RunRecord(
        id=run_id,
        workflow="retry-loop-agent",
        framework="neutral-import",
        provider="openai",
        model="gpt-4o-mini",
        started_at="2026-06-11T00:00:00Z",
        ended_at="2026-06-11T00:00:10Z",
        success_label="failed",
    )


def _attempt(step_id: str, turn: str, *, retry_count: int = 0) -> StepRecord:
    return StepRecord(
        id=step_id,
        run_id="run_f3",
        step_type="function",
        name="crm.lookup",
        started_at=f"2026-06-11T00:00:0{turn[-1]}Z",
        ended_at=f"2026-06-11T00:00:0{turn[-1]}Z",
        parent_step_id=turn,
        span_kind="function",
        retry_scope="agent_root",
        input_fingerprint="fp_lookup_42",
        retry_count=retry_count,
        error="Error running tool",
        error_class="Other",
    )


def _rx(observed: int) -> PrescriptionRecord:
    return PrescriptionRecord(
        id="rx_f3",
        run_id="run_f3",
        severity="high",
        root_cause="retry loop",
        one_line_fix="Set crm.lookup retry budget and fail closed.",
        evidence=["step_id=s1", f"retry_count={observed} additional attempts"],
        patch_type="unified_diff",
        patch=_CAP_DIFF,
        expected_impact={},
        regression_test_template="",
    )


def test_explicit_retry_count_caps_at_l1() -> None:
    """THE F3 PROBE: one step, app-supplied retry_count=5, matching evidence,
    valid cap diff. Pre-fix this earned L2; the count is an import assertion, so
    sufficiency is unverifiable -> L1, with the cap disclosed in limits."""
    bundle = TraceBundle(
        run=_run("run_f3"),
        steps=[_attempt("s1", "turn_1", retry_count=5)],
        prescriptions=[_rx(5)],
    )
    receipts = build_receipts(bundle)
    assert len(receipts) == 1
    assert receipts[0].proof_level == "L1"
    assert any("app-supplied" in lim or "explicit" in lim for lim in receipts[0].limits)


def test_derived_retry_count_still_earns_l2_with_import_disclosure() -> None:
    """The derived lane keeps L2 — but this fixture is an IMPORT-shaped bundle
    (adapter_provenanced=False), and a forged import can fabricate raw attempts
    that ARL's own collapse then derives (Codex Rule 8 re-review F-01). Decision
    recorded in the codex-review file: repair-class grades describe the bundle's
    OWN facts, so derived L2 stands, but the receipt must carry the provenance
    on its face so it cannot be laundered as capture-verified proof."""
    steps = [_attempt(f"s{i}", f"turn_{i}") for i in range(1, 7)]
    bundle = TraceBundle(run=_run("run_f3"), steps=steps, prescriptions=[_rx(5)])
    receipts = build_receipts(bundle)
    assert len(receipts) == 1
    assert receipts[0].proof_level == "L2"
    assert any("imported from a file" in lim for lim in receipts[0].limits)


def test_adapter_provenanced_run_carries_no_import_disclosure() -> None:
    steps = [_attempt(f"s{i}", f"turn_{i}") for i in range(1, 7)]
    bundle = TraceBundle(
        run=_run("run_f3"), steps=steps, prescriptions=[_rx(5)], adapter_provenanced=True
    )
    receipts = build_receipts(bundle)
    assert receipts[0].proof_level == "L2"
    assert not any("imported from a file" in lim for lim in receipts[0].limits)


def test_collapse_marks_derived_and_passthrough_stays_explicit() -> None:
    derived = derive_retry_steps(
        TraceBundle(run=_run("run_f3"), steps=[_attempt(f"s{i}", f"turn_{i}") for i in range(1, 4)])
    )
    assert len(derived) == 1
    assert derived[0].retry_count == 2
    assert derived[0].retry_count_source == "derived"

    passthrough = derive_retry_steps(
        TraceBundle(run=_run("run_f3"), steps=[_attempt("s1", "turn_1", retry_count=4)])
    )
    assert len(passthrough) == 1
    assert passthrough[0].retry_count == 4
    assert passthrough[0].retry_count_source == "explicit"


def test_retry_count_source_is_never_serialized() -> None:
    """The spoof lock (must BITE, not vacuously pass): a file claiming
    retry_count_source="derived" gets "explicit"; to_dict never emits the key."""
    forged = StepRecord.from_dict(
        {"id": "s1", "retry_count": 99, "retry_count_source": "derived"},
        run_id="run_f3",
    )
    assert forged.retry_count == 99
    assert forged.retry_count_source == "explicit"
    assert "retry_count_source" not in forged.to_dict()
    # and a genuinely derived in-memory step does not leak the bit either
    derived = derive_retry_steps(
        TraceBundle(run=_run("run_f3"), steps=[_attempt(f"s{i}", f"turn_{i}") for i in range(1, 4)])
    )[0]
    assert "retry_count_source" not in derived.to_dict()
