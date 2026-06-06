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


def test_clean_run_reports_honest_no_fixable_failure() -> None:
    """build_receipts returns [] on a clean run (no prescription). The report must
    say so honestly — never invent a receipt on a clean run."""
    clean = load_demo_bundle("clean")
    assert build_receipts(clean) == []

    html = render_report(clean)

    assert "No fixable failure detected" in html
    assert "clean run" in html.lower()
