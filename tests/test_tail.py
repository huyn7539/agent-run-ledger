from __future__ import annotations

from pathlib import Path
import math

import pytest

from agent_run_ledger.core.io import load_trace
from agent_run_ledger.core.models import (
    PrescriptionRecord,
    RunRecord,
    StepRecord,
    TraceBundle,
    TraceValidationError,
)
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.report import render_report
from agent_run_ledger.core.storage import load_bundle, save_bundle


def test_empty_bundle_rejected() -> None:
    run = RunRecord(
        id="run_empty",
        workflow="tail",
        framework="test",
        provider="test",
        model="test",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
        success_label="passed",
    )

    with pytest.raises(TraceValidationError, match="at least one step"):
        TraceBundle(run=run, steps=[]).validate()


def test_single_step_bundle_load_save_report_no_rx(tmp_path: Path) -> None:
    bundle = load_trace(Path("fixtures/clean_run.json"))
    db = tmp_path / "ledger.sqlite"

    run_id = save_bundle(db, bundle)
    loaded = load_bundle(db, run_id)

    assert analyze_bundle(loaded) == []
    # A clean run yields no receipt: the report renders the honest no-fixable-
    # failure message in place of a graded repair receipt (report.py now leads with
    # the RepairReceipt; the empty-receipt case is the same honest clean-run case).
    assert "No fixable failure detected" in render_report(loaded)


def test_extreme_token_counts_roundtrip(tmp_path: Path) -> None:
    run = RunRecord(
        id="run_big_tokens",
        workflow="tail",
        framework="test",
        provider="test",
        model="test",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
        success_label="passed",
        total_input_tokens=10**15,
        total_output_tokens=10**15,
    )
    step = StepRecord(
        id="step_big_tokens",
        run_id=run.id,
        step_type="model",
        name="big.tokens",
        started_at=run.started_at,
        ended_at=run.ended_at,
        input_tokens=10**15,
        output_tokens=10**15,
    )
    db = tmp_path / "ledger.sqlite"

    save_bundle(db, TraceBundle(run=run, steps=[step]))
    loaded = load_bundle(db, run.id)

    assert loaded.steps[0].input_tokens == 10**15
    assert loaded.steps[0].output_tokens == 10**15


def test_step_all_optionals_none() -> None:
    run = RunRecord(
        id="run_none",
        workflow="tail",
        framework="test",
        provider="test",
        model="test",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
        success_label="passed",
    )
    step = StepRecord(
        id="step_none",
        run_id=run.id,
        step_type="tool",
        name="none.step",
        started_at=run.started_at,
        ended_at=run.ended_at,
        error=None,
    )

    TraceBundle(run=run, steps=[step]).validate()


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_bad_cost_impossible_end_to_end(bad: float) -> None:
    run = RunRecord(
        id="run_bad_cost",
        workflow="tail",
        framework="test",
        provider="test",
        model="test",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
        success_label="passed",
        total_cost_usd=0.0,
    )
    step = StepRecord(
        id="step_bad_cost",
        run_id=run.id,
        step_type="tool",
        name="bad.cost",
        started_at=run.started_at,
        ended_at=run.ended_at,
        cost_usd=bad,
    )

    with pytest.raises(TraceValidationError):
        TraceBundle(run=run, steps=[step]).validate()


def test_prescription_references_nonexistent_step_currently_allowed() -> None:
    bundle = load_trace(Path("fixtures/clean_run.json"))
    rx = PrescriptionRecord(
        id="rx_missing_step",
        run_id=bundle.run.id,
        severity="medium",
        root_cause="references missing step_id=missing_step",
        one_line_fix="add linkage validation",
        evidence=["step_id=missing_step"],
        patch_type="code_snippet",
        patch="value = 1\n" + ("x" * 64),
        expected_impact={},
        regression_test_template="def test_missing_step(): pass",
    )

    bundle.with_prescriptions([rx]).validate()


def test_prescription_wrong_run_id_currently_allowed() -> None:
    bundle = load_trace(Path("fixtures/clean_run.json"))
    rx = PrescriptionRecord(
        id="rx_wrong_run",
        run_id="other_run",
        severity="medium",
        root_cause="wrong run linkage",
        one_line_fix="add linkage validation",
        evidence=["step_id=step_1"],
        patch_type="code_snippet",
        patch="value = 1\n" + ("x" * 64),
        expected_impact={},
        regression_test_template="def test_wrong_run(): pass",
    )

    bundle.with_prescriptions([rx]).validate()
