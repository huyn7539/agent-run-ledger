import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agent_run_ledger.core.io import load_trace, semantic_trace_dict
from agent_run_ledger.core.io import write_trace
from agent_run_ledger.core.models import (
    PrescriptionRecord,
    RunRecord,
    StepRecord,
    TraceBundle,
    TraceValidationError,
)


def _minimal_bundle(*, step_cost: float = 0.1, run_total: float = 0.1) -> TraceBundle:
    run = RunRecord(
        id="run_cost_validation",
        workflow="cost-validation",
        framework="test",
        provider="test",
        model="test",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
        success_label="passed",
        total_cost_usd=run_total,
    )
    step = StepRecord(
        id="step_cost_validation",
        run_id=run.id,
        step_type="tool",
        name="cost.step",
        started_at=run.started_at,
        ended_at=run.ended_at,
        cost_usd=step_cost,
    )
    return TraceBundle(run=run, steps=[step])


def test_golden_retry_loop_validates() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))

    assert bundle.run.id == "run_retry_loop"
    assert bundle.run.total_input_tokens == 4200
    assert bundle.run.total_output_tokens == 1900
    assert len(bundle.steps) == 3


def test_negative_step_cost_rejected() -> None:
    bundle = _minimal_bundle(step_cost=-5.0, run_total=0.1)

    with pytest.raises(TraceValidationError, match="negative cost_usd"):
        bundle.validate()


def test_negative_run_total_rejected() -> None:
    bundle = _minimal_bundle(step_cost=0.1, run_total=-1.0)

    with pytest.raises(TraceValidationError, match="negative total_cost_usd"):
        bundle.validate()


@pytest.mark.parametrize("bad_cost", [float("nan"), float("inf")])
def test_nonfinite_step_cost_rejected_blocks_json_constants(
    tmp_path: Path, bad_cost: float
) -> None:
    bundle = _minimal_bundle(step_cost=bad_cost, run_total=0.1)

    with pytest.raises(TraceValidationError, match="non-finite cost_usd"):
        bundle.validate()

    out = tmp_path / "trace.json"
    with pytest.raises(TraceValidationError):
        write_trace(bundle, out)
    if out.exists():
        json.loads(
            out.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                AssertionError(f"bare JSON constant reached output: {value}")
            ),
        )


@pytest.mark.parametrize("bad_total", [float("nan"), float("inf")])
def test_nonfinite_run_total_rejected(bad_total: float) -> None:
    bundle = _minimal_bundle(step_cost=0.1, run_total=bad_total)

    with pytest.raises(TraceValidationError, match="non-finite total_cost_usd"):
        bundle.validate()


def test_retry_count_nan_rejected_at_ingest() -> None:
    data = {
        "schema_version": "0.1",
        "run": {
            "id": "run_retry_nan",
            "workflow": "cost-validation",
            "framework": "test",
            "provider": "test",
            "model": "test",
            "started_at": "2026-05-28T00:00:00Z",
            "ended_at": "2026-05-28T00:00:01Z",
            "success_label": "passed",
            "total_cost_usd": 0.1,
        },
        "steps": [
            {
                "id": "step_retry_nan",
                "type": "tool",
                "name": "cost.step",
                "started_at": "2026-05-28T00:00:00Z",
                "ended_at": "2026-05-28T00:00:01Z",
                "cost_usd": 0.1,
                "retry_count": math.nan,
            }
        ],
    }

    with pytest.raises(ValueError):
        TraceBundle.from_dict(data)


def test_trace_roundtrip_semantic_dict() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    roundtripped = TraceBundle.from_dict(bundle.to_dict())

    assert semantic_trace_dict(roundtripped) == semantic_trace_dict(bundle)


def test_computed_totals_zero_falls_through_to_step_sum() -> None:
    bundle = TraceBundle.from_dict(
        {
            "schema_version": "0.1",
            "run": {
                "id": "run_zero_total",
                "workflow": "cost-validation",
                "framework": "test",
                "provider": "test",
                "model": "test",
                "started_at": "2026-05-28T00:00:00Z",
                "ended_at": "2026-05-28T00:00:01Z",
                "success_label": "passed",
                "total_cost_usd": 0.0,
            },
            "steps": [
                {
                    "id": "step_cost",
                    "type": "tool",
                    "name": "cost.step",
                    "started_at": "2026-05-28T00:00:00Z",
                    "ended_at": "2026-05-28T00:00:01Z",
                    "cost_usd": 0.25,
                    "retry_count": 0,
                }
            ],
        }
    )

    assert bundle.run.total_cost_usd == 0.25


@pytest.mark.parametrize("length", [64, 65])
def test_patch_length_boundary_accepts_valid_snippet(length: int) -> None:
    patch = "value = 1\n" + ("x" * (length - len("value = 1\n")))
    bundle = _minimal_bundle().with_prescriptions(
        [
            PrescriptionRecord(
                id=f"rx_len_{length}",
                run_id="run_cost_validation",
                severity="medium",
                root_cause="length",
                one_line_fix="length",
                evidence=["length"],
                patch_type="code_snippet",
                patch=patch,
                expected_impact={},
                regression_test_template="def test_length(): pass",
            )
        ]
    )

    bundle.validate()


def test_patch_length_63_rejected_with_type_valid_snippet() -> None:
    patch = "value = 1\n" + ("x" * (63 - len("value = 1\n")))
    bundle = _minimal_bundle().with_prescriptions(
        [
            PrescriptionRecord(
                id="rx_len_63",
                run_id="run_cost_validation",
                severity="medium",
                root_cause="length",
                one_line_fix="length",
                evidence=["length"],
                patch_type="code_snippet",
                patch=patch,
                expected_impact={},
                regression_test_template="def test_length(): pass",
            )
        ]
    )

    with pytest.raises(TraceValidationError, match="too short"):
        bundle.validate()


def test_whitespace_only_patch_rejected() -> None:
    bundle = _minimal_bundle().with_prescriptions(
        [
            PrescriptionRecord(
                id="rx_blank",
                run_id="run_cost_validation",
                severity="medium",
                root_cause="blank",
                one_line_fix="blank",
                evidence=["blank"],
                patch_type="code_snippet",
                patch=" " * 80,
                expected_impact={},
                regression_test_template="def test_blank(): pass",
            )
        ]
    )

    with pytest.raises(TraceValidationError):
        bundle.validate()


def test_patch_type_unknown_rejected() -> None:
    with pytest.raises(TraceValidationError, match="unsupported patch_type"):
        PrescriptionRecord.from_dict({"patch_type": "unknown"}, "run")


def test_frozen_record_immutability() -> None:
    bundle = _minimal_bundle()

    with pytest.raises(FrozenInstanceError):
        bundle.run.id = "new"  # type: ignore[misc]


def test_with_prescriptions_does_not_mutate_original() -> None:
    bundle = _minimal_bundle()
    updated = bundle.with_prescriptions([])

    assert updated is not bundle
    assert bundle.prescriptions == []


def test_missing_run_key_rejected() -> None:
    with pytest.raises(TraceValidationError, match="missing 'run'"):
        TraceBundle.from_dict({"schema_version": "0.1", "steps": []})


def test_empty_steps_rejected() -> None:
    with pytest.raises(TraceValidationError, match="at least one step"):
        TraceBundle.from_dict({"schema_version": "0.1", "run": {"id": "run_empty"}, "steps": []})


def test_unsupported_schema_version_rejected() -> None:
    # 9.9 is a higher major than the reader (0.1) → rejected by the L1
    # compatibility gate (message changed from "unsupported" to "incompatible"
    # when the exact-match gate became a major.minor policy).
    with pytest.raises(TraceValidationError, match="incompatible schema_version"):
        TraceBundle.from_dict({"schema_version": "9.9"})


def test_duplicate_step_id_rejected() -> None:
    step = _minimal_bundle().steps[0]
    bundle = TraceBundle(run=_minimal_bundle().run, steps=[step, step])

    with pytest.raises(TraceValidationError, match="duplicate step.id"):
        bundle.validate()


def test_step_run_id_mismatch_rejected() -> None:
    run = _minimal_bundle().run
    step = StepRecord(
        id="step_wrong_run",
        run_id="other",
        step_type="tool",
        name="wrong.run",
        started_at=run.started_at,
        ended_at=run.ended_at,
    )
    bundle = TraceBundle(run=run, steps=[step])

    with pytest.raises(TraceValidationError, match="wrong run"):
        bundle.validate()


def test_negative_retry_count_rejected() -> None:
    run = _minimal_bundle().run
    step = StepRecord(
        id="step_negative_retry",
        run_id=run.id,
        step_type="tool",
        name="negative.retry",
        started_at=run.started_at,
        ended_at=run.ended_at,
        retry_count=-1,
    )

    with pytest.raises(TraceValidationError, match="negative retry_count"):
        TraceBundle(run=run, steps=[step]).validate()


def test_missing_required_fields_get_safe_defaults() -> None:
    bundle = TraceBundle.from_dict({"schema_version": "0.1", "run": {}, "steps": [{}]})

    assert bundle.run.workflow == "unknown-workflow"
    assert bundle.steps[0].name == "unnamed-step"


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
