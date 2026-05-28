from dataclasses import replace
from pathlib import Path
import subprocess

from agent_run_ledger.core.io import load_trace
from agent_run_ledger.core.models import TraceBundle
from agent_run_ledger.core.prescriptions import analyze_bundle


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


def test_retry_loop_without_target_context_does_not_emit_fake_unified_diff() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))

    prescription = analyze_bundle(bundle)[0]

    assert prescription.patch_type == "config_diff"
    assert not prescription.patch.startswith("diff --git")


def test_retry_loop_with_target_context_emits_applyable_unified_diff(tmp_path: Path) -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    target_step = next(step for step in bundle.steps if step.name == "crm.lookup_customer")
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


def test_clean_run_emits_no_prescription() -> None:
    bundle = load_trace(Path("fixtures/clean_run.json"))

    assert analyze_bundle(bundle) == []