from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from agent_run_ledger.core.io import load_trace
from agent_run_ledger.core.models import StepRecord, TraceBundle, TraceValidationError
from agent_run_ledger.core.prescriptions import analyze_bundle, detect_retry_cost_loops


def test_retry_loop_emits_patch_artifact() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))

    prescriptions = analyze_bundle(bundle)

    assert len(prescriptions) == 1
    assert prescriptions[0].severity == "high"
    assert prescriptions[0].patch_type == "config_diff"
    assert "Non-runnable config diff" in prescriptions[0].patch
    assert "agent_config.py" not in prescriptions[0].patch
    assert prescriptions[0].regression_test_template.strip()
    assert prescriptions[0].expected_impact["estimated_cost_delta_usd"] == -0.0736


@pytest.mark.parametrize(
    ("cost", "retry_count", "allowed", "expected"),
    [
        (0.10, 2, 0, 0.066667),
        (0.10, 4, 1, 0.06),
        (0.092, 4, 4, 0.0),
        (0.092, 4, 2, 0.0368),
    ],
)
def test_wasted_retry_cost_value_matrix(
    cost: float, retry_count: int, allowed: int, expected: float
) -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    step = next(step for step in bundle.steps if step.name == "demo.flaky_tool")
    step = replace(step, cost_usd=cost, retry_count=retry_count)
    bundle = TraceBundle(run=bundle.run, steps=[step])

    prescription = detect_retry_cost_loops(bundle, allowed_retries=allowed)[0]

    assert prescription.expected_impact["estimated_cost_delta_usd"] == -expected


@pytest.mark.parametrize("retry_count", [0, 1])
def test_retry_below_threshold_emits_none(retry_count: int) -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    step = next(step for step in bundle.steps if step.name == "demo.flaky_tool")

    assert detect_retry_cost_loops(TraceBundle(run=bundle.run, steps=[replace(step, retry_count=retry_count)])) == []


def test_retry_exactly_at_threshold_fires() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    step = next(step for step in bundle.steps if step.name == "demo.flaky_tool")

    assert detect_retry_cost_loops(TraceBundle(run=bundle.run, steps=[replace(step, retry_count=2)]))


def test_allowed_retries_tuning_suppresses_excess_cost_but_still_emits() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    step = next(step for step in bundle.steps if step.name == "demo.flaky_tool")
    prescription = detect_retry_cost_loops(
        TraceBundle(run=bundle.run, steps=[replace(step, retry_count=4)]),
        allowed_retries=5,
    )[0]

    assert prescription.expected_impact["estimated_cost_delta_usd"] == -0.0


def test_zero_cost_still_emits_zero_waste() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    step = next(step for step in bundle.steps if step.name == "demo.flaky_tool")
    prescription = detect_retry_cost_loops(
        TraceBundle(run=bundle.run, steps=[replace(step, cost_usd=0.0)])
    )[0]

    assert prescription.expected_impact["estimated_cost_delta_usd"] == -0.0


def test_severity_high_on_error_medium_otherwise() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    step = next(step for step in bundle.steps if step.name == "demo.flaky_tool")

    high = analyze_bundle(TraceBundle(run=bundle.run, steps=[step]))[0]
    # L8: severity now reads the typed error_class, not the redacted `error`
    # string — clearing the error class downgrades to medium.
    medium = analyze_bundle(
        TraceBundle(run=bundle.run, steps=[replace(step, error=None, error_class=None)])
    )[0]

    assert high.severity == "high"
    assert medium.severity == "medium"


def test_retry_loop_without_target_context_does_not_emit_fake_unified_diff() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))

    prescription = analyze_bundle(bundle)[0]

    assert prescription.patch_type == "config_diff"
    assert not prescription.patch.startswith("diff --git")


def test_retry_loop_with_target_context_emits_applyable_unified_diff(tmp_path: Path) -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    target_step = next(step for step in bundle.steps if step.name == "demo.flaky_tool")
    target_step = replace(
        target_step,
        metadata={
            "retry_budget_patch_target": {
                "path": "settings/retries.py",
                "before": "CRM_LOOKUP_RETRIES = 4",
                "after": "CRM_LOOKUP_RETRIES = 0",
            }
        },
    )
    bundle = TraceBundle(
        run=bundle.run,
        steps=[target_step if step.id == target_step.id else step for step in bundle.steps],
    )
    patch = analyze_bundle(bundle)[0].patch
    target = tmp_path / "target"
    settings = target / "settings"
    settings.mkdir(parents=True)
    (settings / "retries.py").write_bytes(b"CRM_LOOKUP_RETRIES = 4\n")
    subprocess.run(["git", "init"], cwd=target, capture_output=True, check=True)

    result = subprocess.run(
        ["git", "apply", "--check", "-"],
        input=patch.encode(),
        cwd=target,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode()
    assert patch.startswith("diff --git a/settings/retries.py b/settings/retries.py")


@pytest.mark.parametrize(
    ("before", "shape"),
    [
        # AMBIGUOUS lines (more than one integer) must NOT generate a diff — ARL
        # cannot tell which integer is the budget, so it falls back rather than
        # risk corrupting the wrong number. (Security-hardened: the replacement is
        # ARL-generated from the single integer; >1 integer -> refuse.)
        ("RETRIES_4 = 9", "name-collision"),
        ("RETRY_4_LIMIT = 4", "name+value"),
    ],
)
def test_before_replace_digit_collision_does_not_emit_wrong_diff(
    before: str, shape: str
) -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    target_step = next(step for step in bundle.steps if step.name == "demo.flaky_tool")
    target_step = replace(
        target_step,
        metadata={
            "retry_budget_patch_target": {
                "path": "settings/retries.py",
                "before": before,
            }
        },
    )
    bundle = TraceBundle(
        run=bundle.run,
        steps=[target_step if step.id == target_step.id else step for step in bundle.steps],
    )

    prescription = analyze_bundle(bundle)[0]

    assert prescription.patch_type == "config_diff", shape
    assert "Non-runnable config diff" in prescription.patch


def test_single_integer_before_generates_correct_capped_diff() -> None:
    """A `before` line with exactly ONE integer (e.g. RETRIES = 40) is safely
    capped to 0 by ARL — the replacement line is GENERATED, not caller-supplied,
    so it cannot corrupt the wrong number nor carry injected text."""
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    target_step = next(step for step in bundle.steps if step.name == "demo.flaky_tool")
    target_step = replace(
        target_step,
        metadata={"retry_budget_patch_target": {"path": "settings/retries.py", "before": "RETRIES = 40"}},
    )
    bundle = TraceBundle(
        run=bundle.run,
        steps=[target_step if step.id == target_step.id else step for step in bundle.steps],
    )

    prescription = analyze_bundle(bundle)[0]

    assert prescription.patch_type == "unified_diff"
    assert prescription.patch.startswith("diff --git")
    assert "-RETRIES = 40" in prescription.patch
    assert "+RETRIES = 0" in prescription.patch


def test_before_already_at_cap_falls_back_to_config_diff() -> None:
    """If the `before` line is ALREADY at the fail-closed cap (RETRIES = 0), the
    generated `after` equals `before` -> no real change -> config_diff fallback
    (never a no-op diff). Replaces the old caller-supplied before==after test:
    `after` is now ARL-generated, so the only way to hit equality is a budget
    already at the cap."""
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    target_step = next(step for step in bundle.steps if step.name == "demo.flaky_tool")
    target_step = replace(
        target_step,
        metadata={
            "retry_budget_patch_target": {
                "path": "settings/retries.py",
                "before": "CRM_LOOKUP_RETRIES = 0",
            }
        },
    )
    bundle = TraceBundle(
        run=bundle.run,
        steps=[target_step if step.id == target_step.id else step for step in bundle.steps],
    )

    prescription = analyze_bundle(bundle)[0]

    assert prescription.patch_type == "config_diff"
    assert "Non-runnable config diff" in prescription.patch


def test_arl_patch_target_alias_and_backslash_path_normalized() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    target_step = next(step for step in bundle.steps if step.name == "demo.flaky_tool")
    target_step = replace(
        target_step,
        metadata={
            "arl_patch_target": {
                "path": r"settings\retries.py",
                "before": "CRM_LOOKUP_RETRIES = 4",
                "after": "CRM_LOOKUP_RETRIES = 0",
            }
        },
    )

    prescription = analyze_bundle(TraceBundle(run=bundle.run, steps=[target_step]))[0]

    assert prescription.patch_type == "unified_diff"
    assert "settings/retries.py" in prescription.patch


def test_non_demo_fixture_avoids_demo_constants(non_demo_bundle: TraceBundle) -> None:
    prescriptions = analyze_bundle(non_demo_bundle)
    text = "\n".join(
        [
            prescriptions[0].root_cause,
            prescriptions[0].one_line_fix,
            prescriptions[0].patch,
            prescriptions[0].regression_test_template,
            str(prescriptions[0].expected_impact),
        ]
    )

    for demo_identifier in [
        "run_retry_loop",
        "demo.flaky_tool",
        "support-agent-demo",
        "0.092",
        "CRM_LOOKUP_RETRIES",
        "settings/retries.py",
    ]:
        assert demo_identifier not in text


def test_non_demo_target_context_emits_applyable_unified_diff(
    tmp_path: Path, non_demo_target_bundle: TraceBundle
) -> None:
    patch = analyze_bundle(non_demo_target_bundle)[0].patch
    target = tmp_path / "target"
    services = target / "services"
    services.mkdir(parents=True)
    (services / "vendor_retry.py").write_bytes(b"VENDOR_LOOKUP_MAX_RETRIES = 3\n")
    subprocess.run(["git", "init"], cwd=target, capture_output=True, check=True)

    result = subprocess.run(
        ["git", "apply", "--check", "-"],
        input=patch.encode(),
        cwd=target,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode()
    assert "services/vendor_retry.py" in patch


def test_negative_retry_division_guarded_upstream() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    step = StepRecord(
        id="step_negative",
        run_id=bundle.run.id,
        step_type="tool",
        name="negative.retry",
        started_at=bundle.run.started_at,
        ended_at=bundle.run.ended_at,
        retry_count=-1,
        cost_usd=0.1,
    )

    with pytest.raises(TraceValidationError, match="negative retry_count"):
        TraceBundle(run=bundle.run, steps=[step]).validate()


def test_clean_run_emits_no_prescription() -> None:
    bundle = load_trace(Path("fixtures/clean_run.json"))

    assert analyze_bundle(bundle) == []
