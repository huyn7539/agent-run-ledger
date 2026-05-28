from pathlib import Path

from agent_run_ledger.core.io import load_trace, semantic_trace_dict
from agent_run_ledger.core.models import (
    PrescriptionRecord,
    RunRecord,
    StepRecord,
    TraceBundle,
    TraceValidationError,
)


def test_golden_retry_loop_validates() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))

    assert bundle.run.id == "run_retry_loop"
    assert bundle.run.total_input_tokens == 4200
    assert bundle.run.total_output_tokens == 1900
    assert len(bundle.steps) == 3


def test_trace_roundtrip_semantic_dict() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    roundtripped = TraceBundle.from_dict(bundle.to_dict())

    assert semantic_trace_dict(roundtripped) == semantic_trace_dict(bundle)


def test_prescription_requires_runnable_patch() -> None:
    run = RunRecord(
        id="run_validation",
        workflow="validation",
        framework="test",
        provider="test",
        model="test",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
        success_label="failed",
    )
    step = StepRecord(
        id="step_validation",
        run_id=run.id,
        step_type="tool",
        name="demo",
        started_at=run.started_at,
        ended_at=run.ended_at,
    )

    for patch, patch_type in [
        ("x", "code_snippet"),
        ("not a diff but long enough to pass a naive length-only validator" * 2, "unified_diff"),
        ("diff --git a/file b/file\n", "code_snippet"),
    ]:
        bundle = TraceBundle(
            run=run,
            steps=[step],
            prescriptions=[
                PrescriptionRecord(
                    id="rx_bad",
                    run_id=run.id,
                    severity="medium",
                    root_cause="bad patch",
                    one_line_fix="bad patch",
                    evidence=["bad"],
                    patch_type=patch_type,
                    patch=patch,
                    expected_impact={},
                    regression_test_template="def test_bad(): pass",
                )
            ],
        )

        try:
            bundle.validate()
        except TraceValidationError:
            continue
        raise AssertionError(f"invalid patch was accepted: {patch_type} {patch!r}")
