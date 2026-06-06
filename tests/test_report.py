from __future__ import annotations

from pathlib import Path

from agent_run_ledger.core.compare import compare_bundles
from agent_run_ledger.core.io import load_trace
from agent_run_ledger.core.models import PrescriptionRecord, RunRecord, StepRecord, TraceBundle
from agent_run_ledger.core.report import render_comparison, render_report


def test_report_html_escapes_user_fields_and_preserves_emoji() -> None:
    raw = '<script>alert("x")</script>&\U0001f600'
    run = RunRecord(
        id=f"run_{raw}",
        workflow=f"workflow_{raw}",
        framework="test",
        provider="test",
        model="test",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
        success_label=f"failed_{raw}",
        total_cost_usd=0.1,
    )
    step = StepRecord(
        id=f"step_{raw}",
        run_id=run.id,
        step_type=f"tool_{raw}",
        name=f"name_{raw}",
        started_at=run.started_at,
        ended_at=run.ended_at,
        cost_usd=0.1,
        error=f"ValueError: {raw}",
    )
    rx = PrescriptionRecord(
        id="rx_escape",
        run_id=run.id,
        severity=f"medium_{raw}",
        root_cause=f"root_{raw}",
        one_line_fix=f"fix_{raw}",
        evidence=[f"evidence_{raw}"],
        patch_type="code_snippet",
        patch="value = 1\n" + ("x" * 64) + raw,
        expected_impact={"note": raw},
        regression_test_template=f"def test_escape():\n    assert {raw!r}\n",
    )

    html = render_report(TraceBundle(run=run, steps=[step], prescriptions=[rx]))

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&quot;x&quot;" in html
    assert "&amp;" in html
    assert "\U0001f600" in html


def test_no_prescriptions_section_renders() -> None:
    html = render_report(load_trace(Path("fixtures/clean_run.json")))

    # A clean run has no detected failure, so build_receipts() returns [] and the
    # report renders the honest no-fixable-failure message instead of a graded
    # repair receipt (the receipt is now the primary block; report.py).
    assert "No fixable failure detected" in html


def test_report_numeric_fields_do_not_render_nan() -> None:
    html = render_report(load_trace(Path("fixtures/clean_run.json")))

    assert "nan" not in html.lower()


def test_compare_deltas_and_formatting() -> None:
    left = load_trace(Path("fixtures/golden_retry_loop.json"))
    right = load_trace(Path("fixtures/clean_run.json"))

    comparison = compare_bundles(left, right)
    rendered = render_comparison(comparison)

    # LR2 (Task 52 item 2): cost_delta is computed on read from token facts, the
    # SAME source report + list use — NOT the cached total_cost_usd. On these
    # fixtures the cached totals (0.1842, 0.071) are stale vs what the tokens
    # actually price to (0.00472, 0.00248), so the cached delta (-0.1132) is ~50x
    # the true on-read delta (-0.00224). Asserting the on-read delta is what keeps
    # compare in agreement with report/list.
    from agent_run_ledger.core.cost import cost_on_read

    assert comparison.cost_delta_usd == cost_on_read(right) - cost_on_read(left)
    assert comparison.latency_delta_ms == right.run.total_latency_ms - left.run.total_latency_ms
    assert comparison.retry_delta == -4
    assert comparison.success_change == "failed -> passed"
    assert "cost_delta_usd: -0.002240" in rendered
    assert "success_change: failed -> passed" in rendered


def test_identical_run_compare_is_unchanged() -> None:
    bundle = load_trace(Path("fixtures/clean_run.json"))
    comparison = compare_bundles(bundle, bundle)

    assert comparison.cost_delta_usd == 0
    assert comparison.latency_delta_ms == 0
    assert comparison.retry_delta == 0
    assert comparison.success_change == "unchanged"


def test_report_retries_metric_matches_derived_prescription() -> None:
    """HIGH regression (advisor): after collapse-on-read, report.py must read the
    DERIVED retry view, not raw bundle.steps — else a real retry loop renders
    'Retries: 0' while the prescription card below cites retry_count=2 (a self-
    contradicting artifact). report + compare + detector must read ONE view."""
    import json

    from agent_run_ledger.adapters.openai import bundle_from_recorded_trace
    from agent_run_ledger.core.prescriptions import analyze_bundle

    bundle = bundle_from_recorded_trace(
        json.loads(Path("fixtures/live_retry_loop_interleaved.json").read_text()),
        model="gpt-4o-mini",
    )
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    assert bundle.prescriptions  # a retry loop was detected

    html = render_report(bundle)
    # the Retries metric must reflect the derived loop (>=2), not the raw 0
    import re

    m = re.search(r"Retries:</strong> (\d+)", html)
    assert m is not None
    assert int(m.group(1)) >= 2, "report Retries metric contradicts the prescription"


def test_compare_retry_delta_uses_derived_view() -> None:
    """compare.py must also read the derived view: a clean run vs a real retry
    loop must show a non-zero retry_delta."""
    import json

    from agent_run_ledger.adapters.openai import bundle_from_recorded_trace

    loop = bundle_from_recorded_trace(
        json.loads(Path("fixtures/live_retry_loop_interleaved.json").read_text()),
        model="gpt-4o-mini",
    )
    clean = load_trace(Path("fixtures/clean_run.json"))

    comparison = compare_bundles(clean, loop)
    assert comparison.retry_delta >= 2
