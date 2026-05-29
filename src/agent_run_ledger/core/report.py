from __future__ import annotations

from html import escape
from pathlib import Path

from agent_run_ledger.core.compare import RunComparison
from agent_run_ledger.core.cost import cost_on_read
from agent_run_ledger.core.models import TraceBundle


def render_report(bundle: TraceBundle) -> str:
    retry_total = sum(step.retry_count for step in bundle.steps)
    error_total = sum(1 for step in bundle.steps if step.error)
    # L7/LR2: the displayed run cost is computed on read from the FACTS, never
    # the cached total_cost_usd (which a price-table change can make stale).
    run_cost = cost_on_read(bundle)
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
        for step in bundle.steps
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
    <span class="metric"><strong>Cost:</strong> ${run_cost:.6f}</span>
    <span class="metric"><strong>Latency:</strong> {bundle.run.total_latency_ms} ms</span>
    <span class="metric"><strong>Tokens:</strong> {bundle.run.total_input_tokens + bundle.run.total_output_tokens}</span>
    <span class="metric"><strong>Retries:</strong> {retry_total}</span>
    <span class="metric"><strong>Errors:</strong> {error_total}</span>
    <span class="metric"><strong>Outcome:</strong> {escape(bundle.run.success_label)}</span>
  </div>
  <h2>Steps</h2>
  <table>
    <thead>
      <tr><th>ID</th><th>Type</th><th>Name</th><th>Tokens</th><th>Cost</th><th>Retries</th><th>Error</th></tr>
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
    if not bundle.prescriptions:
        return "<h2>Recommended Next Test</h2><p>No prescriptions emitted.</p>"
    cards = []
    for item in bundle.prescriptions:
        evidence = "".join(f"<li>{escape(line)}</li>" for line in item.evidence)
        cards.append(
            f"""
            <section>
              <h2 class="warn">Recommended Next Test: {escape(item.one_line_fix)}</h2>
              <p><strong>Severity:</strong> {escape(item.severity)}</p>
              <p><strong>Patch type:</strong> {escape(item.patch_type)}</p>
              <p><strong>Root cause:</strong> {escape(item.root_cause)}</p>
              <h3>Evidence</h3>
              <ul>{evidence}</ul>
              <h3>Patch Artifact</h3>
              <pre>{escape(item.patch)}</pre>
              <h3>Expected Impact</h3>
              <pre>{escape(str(item.expected_impact))}</pre>
              <h3>Regression Test Template</h3>
              <pre>{escape(item.regression_test_template)}</pre>
            </section>
            """
        )
    return "\n".join(cards)
