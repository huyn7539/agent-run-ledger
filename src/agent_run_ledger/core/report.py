from __future__ import annotations

from html import escape
from pathlib import Path

from agent_run_ledger.core.compare import RunComparison
from agent_run_ledger.core.cost import cost_display
from agent_run_ledger.core.models import PrescriptionRecord, TraceBundle
from agent_run_ledger.core.prescriptions import derive_retry_steps
from agent_run_ledger.core.receipt import RepairReceipt, build_receipts


def render_report(bundle: TraceBundle) -> str:
    # Read the DERIVED retry view (collapse-on-read), the SAME view the detector
    # uses — so the Retries metric + step rows agree with the prescription cards.
    # Reading raw bundle.steps would show retry_count=0 per attempt while a
    # prescription cites retry_count=N (a self-contradicting artifact).
    steps = derive_retry_steps(bundle)
    retry_total = sum(step.retry_count for step in steps)
    error_total = sum(1 for step in steps if step.error)
    # L7/LR2: the displayed run cost is computed on read from the FACTS, never
    # the cached total_cost_usd (which a price-table change can make stale). Use the
    # disclosure form so an unpriced model (real tokens, unknown rate) reads as
    # "unpriced (...)" rather than a misleading $0.00 (A2).
    run_cost_display = cost_display(bundle)
    step_rows = "\n".join(
        "<tr>"
        f"<td>{escape(step.id)}</td>"
        f"<td>{escape(step.step_type)}</td>"
        f"<td>{escape(step.name)}</td>"
        f"<td>{step.input_tokens + step.output_tokens}</td>"
        f"<td>${step.cost_usd:.6f}</td>"
        f"<td>{step.retry_count}</td>"
        f"<td>{escape(step.error or '')}</td>"
        "</tr>"
        for step in steps
    )
    prescription_html = _render_prescriptions(bundle)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Agent Run Ledger - {escape(bundle.run.id)}</title>
  <style>
    body {{ font-family: Inter, Segoe UI, Arial, sans-serif; margin: 32px; color: #161616; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d8d8d8; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f5f7; }}
    code, pre {{ background: #f6f6f6; padding: 2px 4px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    .metric {{ display: inline-block; margin-right: 18px; }}
    .warn {{ color: #9a3412; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Agent Run Ledger</h1>
  <p><strong>Run:</strong> {escape(bundle.run.id)} | <strong>Workflow:</strong> {escape(bundle.run.workflow)}</p>
  <div>
    <span class="metric"><strong>Cost:</strong> {escape(run_cost_display)}</span>
    <span class="metric"><strong>Latency:</strong> {bundle.run.total_latency_ms} ms</span>
    <span class="metric"><strong>Tokens:</strong> {bundle.run.total_input_tokens + bundle.run.total_output_tokens}</span>
    <span class="metric"><strong>Retries:</strong> {retry_total}</span>
    <span class="metric"><strong>Errors:</strong> {error_total}</span>
    <span class="metric"><strong>Outcome:</strong> {escape(bundle.run.success_label)}</span>
  </div>
  <h2>Steps</h2>
  <p style="color:#555;font-size:0.9em">Per-step cost is the adapter/cached
  estimate and may not sum to the headline Cost above, which is computed on read
  from token facts. The headline figure is the authoritative one.</p>
  <table>
    <thead>
      <tr><th>ID</th><th>Type</th><th>Name</th><th>Tokens</th><th>Cost (cached est.)</th><th>Retries</th><th>Error</th></tr>
    </thead>
    <tbody>{step_rows}</tbody>
  </table>
  {prescription_html}
</body>
</html>
"""


def write_report(bundle: TraceBundle, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(bundle), encoding="utf-8")
    return path


def render_comparison(comparison: RunComparison) -> str:
    lines = ["Run comparison"]
    for key, value in comparison.to_rows():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _render_prescriptions(bundle: TraceBundle) -> str:
    # The graded RepairReceipt is the PRODUCT — the honesty layer (proof level,
    # claim, confidence, limits, next evidence) computed on read from the facts.
    # build_receipts returns [] when there is no detected failure (the negative
    # gate: no invented receipts on a clean run), so the empty list and the
    # no-prescription branch are the SAME honest case.
    receipts = build_receipts(bundle)
    if not receipts:
        # No prescription -> no receipt. But "no receipt" is NOT "clean run": this
        # slice only DETECTS the retry-cap class, so a run that FAILED for another
        # reason (timeout, schema error, context drift) also yields zero receipts.
        # Claiming "clean run" on such a run would contradict the metrics row
        # (Errors/Outcome) in this same document AND fire on the modal real-session
        # input (real sessions fail mostly for non-retry reasons). Only call a run
        # clean when it actually passed with no errors; otherwise say, honestly,
        # that this run's failure is outside the one class ARL currently grades.
        derived = derive_retry_steps(bundle)
        has_errors = any(step.error for step in derived)
        if bundle.run.success_label == "passed" and not has_errors:
            return (
                "<h2>Repair Receipt</h2>"
                "<p>No fixable failure detected — clean run. ARL emits a graded "
                "repair receipt only when it detects a fixable failure path; this "
                "run produced none, so there is nothing to grade.</p>"
            )
        return (
            "<h2>Repair Receipt</h2>"
            "<p>No retry-cap repair receipt for this run. ARL's current slice grades "
            "the retry-loop class only; this run did not match that class, so there is "
            "no graded receipt — this is <strong>not</strong> a clean-run claim (see "
            "the run outcome and errors above). Other failure classes are out of scope "
            "for this slice.</p>"
        )
    # Receipts and prescriptions are BOTH built from bundle.prescriptions in the same
    # order, so zip pairs each receipt with the prescription it graded (its patch +
    # regression test render as the SECONDARY details block under the receipt).
    # Guard the invariant explicitly: if a future code path ever filters receipts,
    # a bare zip() would SILENTLY truncate and mis-pair receipts to prescriptions
    # (showing the wrong patch under a grade) — fail loud instead (fleet INFO ×3).
    if len(receipts) != len(bundle.prescriptions):
        raise ValueError(
            f"receipt/prescription count mismatch: {len(receipts)} receipts vs "
            f"{len(bundle.prescriptions)} prescriptions — pairing would be unsafe"
        )
    cards = [
        _render_receipt_card(receipt, item)
        for receipt, item in zip(receipts, bundle.prescriptions)
    ]
    return "\n".join(cards)


def _render_receipt_card(receipt: RepairReceipt, item: PrescriptionRecord) -> str:
    """Render ONE graded receipt as the PRIMARY block, with the raw prescription
    artifact (patch, expected impact, regression test) kept as a SECONDARY details
    block beneath it. The honesty layer leads; the artifact follows."""
    # PRIMARY: the graded receipt. The limits list is shown PROMINENTLY (a visible
    # <ul>, never hidden) — it is the core of what the receipt sells.
    limits = "".join(f"<li>{escape(line)}</li>" for line in receipt.limits)
    next_evidence = "".join(f"<li>{escape(line)}</li>" for line in receipt.next_evidence)
    receipt_evidence = "".join(f"<li>{escape(line)}</li>" for line in receipt.evidence)
    # SECONDARY: the raw prescription artifact (root cause + patch + regression test
    # are kept, not deleted — the receipt is ADDED above them).
    rx_evidence = "".join(f"<li>{escape(line)}</li>" for line in item.evidence)
    return f"""
            <section>
              <h2 class="warn">Repair Receipt: {escape(item.one_line_fix)}</h2>
              <p><strong>Proof level:</strong> {escape(receipt.proof_level)}</p>
              <p><strong>Confidence:</strong> {escape(receipt.confidence)}</p>
              <p><strong>Observed failure:</strong> {escape(receipt.observed_failure)}</p>
              <p><strong>Claim:</strong> {escape(receipt.claim)}</p>
              <h3>Limits</h3>
              <ul>{limits}</ul>
              <h3>Next Evidence</h3>
              <ul>{next_evidence}</ul>
              <h3>Outcome Delta</h3>
              <pre>{escape(str(receipt.outcome_delta))}</pre>
              <h3>Receipt Evidence</h3>
              <ul>{receipt_evidence}</ul>
              <details>
                <summary>Prescription artifact (details)</summary>
                <p><strong>Severity:</strong> {escape(item.severity)}</p>
                <p><strong>Patch type:</strong> {escape(item.patch_type)}</p>
                <p><strong>Root cause:</strong> {escape(item.root_cause)}</p>
                <h3>Evidence</h3>
                <ul>{rx_evidence}</ul>
                <h3>Patch Artifact</h3>
                <pre>{escape(item.patch)}</pre>
                <h3>Expected Impact</h3>
                <pre>{escape(str(item.expected_impact))}</pre>
                <h3>Regression Test Template</h3>
                <pre>{escape(item.regression_test_template)}</pre>
              </details>
            </section>
            """
