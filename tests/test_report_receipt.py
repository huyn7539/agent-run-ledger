"""The report must render the graded RepairReceipt as the PRIMARY block.

Honesty layer being sold: the receipt's proof_level / claim / confidence / limits /
next_evidence — NOT just the raw prescription. The earlier report rendered only
severity/root_cause/patch/expected_impact/regression_test and OMITTED the entire
honesty grade. This guards that the receipt is rendered, at its TRUE level.

The shipped demo path (cli.py: ``load_demo_bundle(variant)`` then
``analyze_bundle``) renders the SAME bundle these tests render, so the asserted
grade is the grade the demo actually shows. The retry-loop demo bundle is the
native-shape ``golden_retry_loop.json`` — its function spans carry NO
``retry_budget_patch_target``, so the artifact is the non-runnable config_diff
fallback and the receipt honestly grades **L1** (relevance, not mechanical
removal). Asserting L1 here — not L2 — is the demo-overfit guard
(test_live_capture_receipt.py:test_real_captured_sdk_run_receipt_is_l1_not_l2):
the demo must show its real grade, never an inflated L2.
"""

from __future__ import annotations

from agent_run_ledger.core.demo import load_demo_bundle
from agent_run_ledger.core.models import RunRecord, StepRecord, TraceBundle
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.receipt import build_receipts
from agent_run_ledger.core.report import render_report


def _demo_retry_bundle():
    """The SHIPPED retry-loop demo bundle — exactly the cli.py demo path."""
    bundle = load_demo_bundle("retry-loop")
    return bundle.with_prescriptions(analyze_bundle(bundle))


def test_report_renders_graded_receipt_as_primary_block() -> None:
    """The receipt's proof_level + confidence + at least one limits line must be
    rendered. These three are the honesty layer; rendering only the raw
    prescription (the prior behavior) omits all three."""
    bundle = _demo_retry_bundle()
    # Sanity: the demo bundle has a detected loop, and the receipt grades L1 here
    # (native shape, no patch target -> config_diff fallback). If this ever flips
    # to L2 without an instrumentation path, the demo-overfit failure has returned.
    receipts = build_receipts(bundle)
    assert len(receipts) == 1
    assert receipts[0].proof_level == "L1"
    assert receipts[0].confidence == "low"

    html = render_report(bundle)

    # (a) proof_level rendered — anchored to its label so a coincidental "L1"
    #     elsewhere cannot pass for it.
    assert "Proof level:</strong> L1" in html
    # (b) confidence value rendered — anchored to its label (the bare word "low"
    #     appears in unrelated markup; the labeled value is the real assertion).
    assert "Confidence:</strong> low" in html
    # (c) at least ONE limits line rendered PROMINENTLY — a distinctive phrase
    #     from receipt.py's _limits (regression-to-the-mean disclosure).
    assert "fires on the worst runs" in html


def test_report_keeps_patch_and_regression_test_as_secondary() -> None:
    """The receipt is ADDED above the raw artifact, not a replacement: the
    prescription's patch + regression-test template + root cause must still
    render (now as a secondary details block)."""
    bundle = _demo_retry_bundle()
    rx = bundle.prescriptions[0]
    html = render_report(bundle)

    assert "Regression Test Template" in html
    assert rx.regression_test_template.split("\n")[0] in html
    # the patch artifact body still renders
    assert rx.patch.splitlines()[0] in html


def test_report_receipt_leads_artifact_follows_in_details() -> None:
    """Item 1 has TWO halves: (a) the graded receipt is PRIMARY, and (b) the raw
    prescription artifact is DEMOTED into a <details> block beneath it. Half (b)
    must be pinned by an assertion, not just the test name — a regression that
    pulls the artifact back to primary, or drops the receipt's primacy, must fail.
    """
    bundle = _demo_retry_bundle()
    html = render_report(bundle)

    assert "<details>" in html
    assert "<summary>" in html
    # The receipt (Proof level) must appear BEFORE the demoted artifact block.
    assert html.index("Proof level") < html.index("<details>")
    # The patch artifact lives INSIDE the details block, not above it.
    assert html.index("<details>") < html.index("Patch Artifact")


def _failed_non_retry_bundle() -> TraceBundle:
    """A run that FAILED for a non-retry reason (a timeout, no retry loop) -> ARL
    emits NO prescription. This is the MODAL concierge-demo input: real user
    sessions overwhelmingly fail for non-retry reasons (context drift, claim-
    evidence mismatch, weak gates), so this empty-prescription path is the DEFAULT
    real-session case, not an edge case."""
    run = RunRecord(
        id="run_failed_timeout",
        workflow="w",
        framework="f",
        provider="openai",
        model="gpt-4o",
        started_at="2026-06-06T00:00:00Z",
        ended_at="2026-06-06T00:00:01Z",
        success_label="failed",
        total_cost_usd=0.01,
        total_input_tokens=1000,
        total_output_tokens=200,
    )
    step = StepRecord(
        id="run_failed_timeout_s1",
        run_id="run_failed_timeout",
        step_type="tool",
        name="fetch",
        started_at=run.started_at,
        ended_at=run.ended_at,
        input_tokens=1000,
        output_tokens=200,
        retry_count=0,
        error="Timeout",
        error_class="Timeout",
    )
    return TraceBundle(run=run, steps=[step])


def test_failed_non_retry_run_does_not_claim_clean() -> None:
    """THE honesty blocker (Codex + code-reviewer fleet): a run that FAILED on a
    non-retry class produces no prescription, so build_receipts is []. The report
    must NOT then claim 'clean run' — that contradicts the metrics row (Errors: 1,
    Outcome: failed) in the SAME document, and it fires on the modal real-session
    input. ARL must never say a failed run was clean."""
    bundle = _failed_non_retry_bundle()
    assert build_receipts(bundle) == []  # no retry loop -> no prescription

    html = render_report(bundle)

    # The metrics row must (still) show the failure honestly.
    assert "Outcome:</strong> failed" in html
    # And the receipt section must NOT call a failed run "clean".
    assert "clean run" not in html.lower()
    # It must say something true: no retry-cap receipt / out of scope for this run.
    assert "Repair Receipt" in html


def test_clean_run_reports_honest_no_fixable_failure() -> None:
    """build_receipts returns [] on a clean run (no prescription). The report must
    say so honestly — never invent a receipt on a clean run. The genuinely-clean
    case (passed, no errors) is the ONLY case allowed to say 'clean run'."""
    clean = load_demo_bundle("clean")
    assert build_receipts(clean) == []

    html = render_report(clean)

    assert "No fixable failure detected" in html
    assert "clean run" in html.lower()
