"""B2 — the L2 verifier must EARN the grade: a lowered-but-INSUFFICIENT retry cap
must NOT grade L2.

The substring/decrease bug is already fixed (test_patch_safety.py): the grader
requires a real numeric DECREASE on a retry-budget line. But a DECREASE that is
still ABOVE the observed retry count does not mechanically remove the observed
failure path. Observed ``retry_count=2`` is a 3-attempt loop; a cap of 5 (even
lowered from 10) still permits that exact loop, yet the L2 claim says the repair
"statically removes the unbounded-retry failure path."

The honest GENERATED pipeline always uses ``allowed_retries=0`` so it is safe — the
gap is the VERIFIER boundary: a stored/imported prescription whose diff lowers but
does not prevent the observed loop. ``build_receipts`` must compare the new cap
against the observed retry count (recovered from ``rx.evidence``) and require
``new_budget < observed_retry_count`` (STRICT): the cap must drop the loop strictly
below the observed additional-attempt count. A cap EQUAL to the observed count still
permits exactly the observed loop, so it does NOT earn L2. If the observed count
cannot be recovered, grading FAILS CLOSED to L1 (never grant L2 when sufficiency is
unverifiable).
"""

from __future__ import annotations

from agent_run_ledger.core.models import PrescriptionRecord, RunRecord, StepRecord, TraceBundle
from agent_run_ledger.core.receipt import build_receipts


def _unified_cap_diff(before_val: int, after_val: int) -> str:
    return (
        "diff --git a/agent/tools/crm.py b/agent/tools/crm.py\n"
        "--- a/agent/tools/crm.py\n"
        "+++ b/agent/tools/crm.py\n"
        "@@ -1 +1 @@\n"
        f"-CRM_LOOKUP_MAX_RETRIES = {before_val}\n"
        f"+CRM_LOOKUP_MAX_RETRIES = {after_val}\n"
    )


def _bundle_with_prescription(*, observed_retry_count: int, before_val: int, after_val: int) -> TraceBundle:
    """A bundle carrying ONE retry-cap prescription whose evidence cites
    *observed_retry_count* and whose unified diff lowers the cap from *before_val*
    to *after_val*. Models a stored/imported prescription hitting the verifier."""
    run = RunRecord(
        id="run_verifier_boundary",
        workflow="retry-loop-agent",
        framework="openai-agents-python",
        provider="openai",
        model="gpt-4o-mini",
        started_at="2026-05-31T10:00:00Z",
        ended_at="2026-05-31T10:00:10Z",
        success_label="failed",
        total_cost_usd=0.03,
    )
    step = StepRecord(
        id="fn_attempt1",
        run_id=run.id,
        step_type="function",
        name="crm.lookup",
        started_at="2026-05-31T10:00:00Z",
        ended_at="2026-05-31T10:00:06Z",
        span_kind="function",
        retry_count=observed_retry_count,
        cost_usd=0.03,
        error="Error running tool",
        error_class="Other",
    )
    rx = PrescriptionRecord(
        id="rx_imported_0001",
        run_id=run.id,
        severity="high",
        root_cause=f"crm.lookup made {observed_retry_count} additional attempts after the first",
        one_line_fix="Set crm.lookup retry budget and fail closed.",
        evidence=[
            "step_id=fn_attempt1",
            f"retry_count={observed_retry_count} additional attempts",
            f"total_attempts={observed_retry_count + 1}",
            "step_cost_usd=0.030000",
            "step_error_class=Other",
        ],
        patch_type="unified_diff",
        patch=_unified_cap_diff(before_val, after_val),
        expected_impact={"estimated_cost_delta_usd": -0.02},
        regression_test_template="def test_crm_lookup_retry_budget():\n    assert True\n",
    )
    return TraceBundle(run=run, steps=[step], prescriptions=[rx])


def test_lowered_but_insufficient_cap_is_not_graded_l2() -> None:
    """RED-FIRST (B2): observed retry_count=2 (3-attempt loop); the diff lowers the
    cap 10 -> 5, but 5 still permits the observed loop. This must NOT grade L2 — the
    repair does not mechanically remove the observed failure path."""
    bundle = _bundle_with_prescription(observed_retry_count=2, before_val=10, after_val=5)
    receipts = build_receipts(bundle)
    assert len(receipts) == 1
    assert receipts[0].proof_level != "L2", (
        "a cap of 5 does not prevent an observed retry_count=2 (3-attempt) loop; "
        "grading it L2 overclaims 'statically removes the failure path'"
    )
    assert receipts[0].proof_level == "L1"


def test_sufficient_cap_below_observed_count_is_graded_l2() -> None:
    """The strong honest path: capping to 0 (strictly < the observed 2) DOES prevent
    the observed loop -> L2 is earned."""
    bundle = _bundle_with_prescription(observed_retry_count=2, before_val=10, after_val=0)
    receipts = build_receipts(bundle)
    assert len(receipts) == 1
    assert receipts[0].proof_level == "L2"


def test_cap_equal_to_observed_count_does_not_earn_l2() -> None:
    """Boundary: a cap EQUAL to the observed retry_count still permits exactly the
    observed loop (retry_count is additional attempts; new_budget must be strictly
    LESS to drop below it). Cap 2 with observed 2 -> NOT L2."""
    bundle = _bundle_with_prescription(observed_retry_count=2, before_val=5, after_val=2)
    receipts = build_receipts(bundle)
    assert receipts[0].proof_level != "L2"


def test_stale_or_conflicting_retry_count_token_does_not_grade_false_l2() -> None:
    """B2-L1 (fleet code-reviewer): the observed-count recovery must not be fooled
    by a STALE/extra ``retry_count=`` token earlier in evidence. A stored/imported
    prescription whose evidence carries a high stale ``retry_count=99`` token ahead
    of the real ``retry_count=2 additional attempts`` line must NOT let an
    insufficient cap (->5) grade L2. The recognizer anchors on the full evidence
    literal and rejects on conflicting distinct values -> fail closed to L1."""
    bundle = _bundle_with_prescription(observed_retry_count=2, before_val=10, after_val=5)
    rx = bundle.prescriptions[0]
    poisoned = PrescriptionRecord(
        id=rx.id, run_id=rx.run_id, severity=rx.severity, root_cause=rx.root_cause,
        one_line_fix=rx.one_line_fix,
        # a stale/foreign retry_count token ahead of the genuine evidence line
        evidence=["note retry_count=99 (stale)", *rx.evidence],
        patch_type=rx.patch_type, patch=rx.patch, expected_impact=rx.expected_impact,
        regression_test_template=rx.regression_test_template,
    )
    bundle = bundle.with_prescriptions([poisoned])
    receipts = build_receipts(bundle)
    assert receipts[0].proof_level != "L2", (
        "a stale retry_count=99 token must not let a cap of 5 grade L2 over a real "
        "2-retry loop"
    )
    assert receipts[0].proof_level == "L1"


def test_stale_full_phrase_in_a_larger_evidence_string_does_not_supply_a_count() -> None:
    """Codex re-review P1: the observed-count recovery must match only the EXACT
    ARL-authored evidence line ("retry_count=<N> additional attempts"), not the
    phrase embedded in a larger free-text note. A stored/imported prescription whose
    ONLY matching line is `"stale note from old run: retry_count=99 additional
    attempts"` (with NO genuine ARL evidence line) supplies no observed count ->
    sufficiency is unrecoverable -> fail closed to L1, even though the cap (->5)
    lowers from 10."""
    bundle = _bundle_with_prescription(observed_retry_count=2, before_val=10, after_val=5)
    rx = bundle.prescriptions[0]
    poisoned = PrescriptionRecord(
        id=rx.id, run_id=rx.run_id, severity=rx.severity, root_cause=rx.root_cause,
        one_line_fix=rx.one_line_fix,
        # NO genuine ARL evidence line; only a free-text note that embeds the phrase
        evidence=["stale note from old run: retry_count=99 additional attempts"],
        patch_type=rx.patch_type, patch=rx.patch, expected_impact=rx.expected_impact,
        regression_test_template=rx.regression_test_template,
    )
    bundle = bundle.with_prescriptions([poisoned])
    receipts = build_receipts(bundle)
    assert receipts[0].proof_level == "L1", (
        "a stale full phrase inside a free-text note must not supply the observed "
        "count; with no genuine ARL evidence line, sufficiency is unrecoverable -> L1"
    )


def test_comment_only_budget_diff_is_not_graded_l2() -> None:
    """Codex re-review P2: a diff that only changes a COMMENTED-OUT budget line does
    not mechanically remove the failure path — the live assignment is untouched. The
    syntax gate must reject a budget line whose identifier is behind a comment
    marker, so build_receipts cannot bless a comment-only artifact as L2."""
    comment_only = (
        "diff --git a/agent/tools/crm.py b/agent/tools/crm.py\n"
        "--- a/agent/tools/crm.py\n"
        "+++ b/agent/tools/crm.py\n"
        "@@ -1 +1 @@\n"
        "-# CRM_LOOKUP_MAX_RETRIES = 10\n"
        "+# CRM_LOOKUP_MAX_RETRIES = 0\n"
    )
    bundle = _bundle_with_prescription(observed_retry_count=2, before_val=10, after_val=0)
    rx = bundle.prescriptions[0]
    commented = PrescriptionRecord(
        id=rx.id, run_id=rx.run_id, severity=rx.severity, root_cause=rx.root_cause,
        one_line_fix=rx.one_line_fix, evidence=rx.evidence,
        patch_type="unified_diff", patch=comment_only,
        expected_impact=rx.expected_impact, regression_test_template=rx.regression_test_template,
    )
    bundle = bundle.with_prescriptions([commented])
    receipts = build_receipts(bundle)
    assert receipts[0].proof_level != "L2", (
        "a comment-only budget change does not remove the live failure path; "
        "grading it L2 overclaims mechanical removal"
    )


def test_unrecoverable_observed_count_fails_closed_to_l1() -> None:
    """FAIL CLOSED: if the observed retry count cannot be recovered from evidence,
    sufficiency is unverifiable -> never grant L2. The diff is a valid decrease but
    the grader has no observed count to check it against."""
    bundle = _bundle_with_prescription(observed_retry_count=2, before_val=10, after_val=0)
    # strip the retry_count evidence line so the observed count is unrecoverable
    rx = bundle.prescriptions[0]
    stripped = PrescriptionRecord(
        id=rx.id, run_id=rx.run_id, severity=rx.severity, root_cause="crm.lookup looped",
        one_line_fix=rx.one_line_fix,
        evidence=["step_id=fn_attempt1", "step_error_class=Other"],
        patch_type=rx.patch_type, patch=rx.patch, expected_impact=rx.expected_impact,
        regression_test_template=rx.regression_test_template,
    )
    bundle = bundle.with_prescriptions([stripped])
    receipts = build_receipts(bundle)
    assert receipts[0].proof_level == "L1", "unverifiable sufficiency must fail closed to L1"
