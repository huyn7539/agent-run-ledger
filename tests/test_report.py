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

    assert "No prescriptions emitted." in html


def test_report_numeric_fields_do_not_render_nan() -> None:
    html = render_report(load_trace(Path("fixtures/clean_run.json")))

    assert "nan" not in html.lower()


def test_compare_deltas_and_formatting() -> None:
    left = load_trace(Path("fixtures/golden_retry_loop.json"))
    right = load_trace(Path("fixtures/clean_run.json"))

    comparison = compare_bundles(left, right)
    rendered = render_comparison(comparison)

    assert comparison.cost_delta_usd == right.run.total_cost_usd - left.run.total_cost_usd
    assert comparison.latency_delta_ms == right.run.total_latency_ms - left.run.total_latency_ms
    assert comparison.retry_delta == -4
    assert comparison.success_change == "failed -> passed"
    assert "cost_delta_usd: -0.113200" in rendered
    assert "success_change: failed -> passed" in rendered


def test_identical_run_compare_is_unchanged() -> None:
    bundle = load_trace(Path("fixtures/clean_run.json"))
    comparison = compare_bundles(bundle, bundle)

    assert comparison.cost_delta_usd == 0
    assert comparison.latency_delta_ms == 0
    assert comparison.retry_delta == 0
    assert comparison.success_change == "unchanged"
