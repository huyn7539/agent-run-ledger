from __future__ import annotations

import difflib
from typing import Any
from uuid import uuid4

from agent_run_ledger.core.models import PrescriptionRecord, StepRecord, TraceBundle


def analyze_bundle(bundle: TraceBundle) -> list[PrescriptionRecord]:
    return detect_retry_cost_loops(bundle)


def detect_retry_cost_loops(
    bundle: TraceBundle,
    retry_threshold: int = 2,
    allowed_retries: int = 0,
) -> list[PrescriptionRecord]:
    prescriptions: list[PrescriptionRecord] = []
    for step in bundle.steps:
        if step.retry_count < retry_threshold:
            continue
        prescriptions.append(_retry_loop_prescription(bundle, step, allowed_retries))
    return prescriptions


def _retry_loop_prescription(
    bundle: TraceBundle,
    step: StepRecord,
    allowed_retries: int,
) -> PrescriptionRecord:
    wasted_cost = _wasted_retry_cost(step, allowed_retries)
    safe_name = _safe_name(step.name)
    patch_type, patch = _retry_budget_artifact(step, allowed_retries)
    regression = f"""def test_{_safe_name(step.name)}_retry_budget():
    result = run_agent_fixture("fixtures/{bundle.run.workflow}.json")
    assert result.step("{step.name}").retry_count <= {allowed_retries}
    assert result.total_cost_usd <= {round(bundle.run.total_cost_usd - wasted_cost, 6)}
"""
    return PrescriptionRecord(
        id=f"rx_{uuid4().hex[:12]}",
        run_id=bundle.run.id,
        # L8: severity reads the typed error_class (the bounded fact the wedge
        # needs), not the always-redacted `error` string. A captured error class
        # means the retry loop ended in a real failure -> high severity.
        severity="high" if step.error_class else "medium",
        root_cause=(
            f"{step.name} made {step.retry_count} additional attempts after the first "
            f"in one run ({1 + step.retry_count} total attempts)"
        ),
        one_line_fix=f"Set {step.name} retry budget to {allowed_retries} and fail closed.",
        evidence=[
            f"step_id={step.id}",
            f"retry_count={step.retry_count} additional attempts",
            f"total_attempts={1 + step.retry_count}",
            f"step_cost_usd={step.cost_usd:.6f}",
            f"step_error_class={step.error_class or 'none'}",
            "cost_estimate=uniform-per-attempt approximation",
        ],
        patch_type=patch_type,
        patch=patch,
        expected_impact={
            "estimated_cost_delta_usd": -wasted_cost,
            "latency_delta_ms": "lower if retry loop was on the critical path",
            "success_delta": "unchanged or higher if caller handles typed failure",
        },
        regression_test_template=regression.replace(_safe_name(step.name), safe_name),
    )


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_") or "step"


def _wasted_retry_cost(step: StepRecord, allowed_retries: int) -> float:
    excess_retries = max(step.retry_count - allowed_retries, 0)
    total_attempts = 1 + step.retry_count
    return round(step.cost_usd * excess_retries / total_attempts, 6)


def _retry_budget_artifact(step: StepRecord, allowed_retries: int) -> tuple[str, str]:
    target = _retry_budget_patch_target(step, allowed_retries)
    if target is None:
        return "config_diff", _retry_budget_config_diff(step.name, step.retry_count, allowed_retries)
    return "unified_diff", _unified_retry_budget_patch(target)


def _retry_budget_patch_target(step: StepRecord, allowed_retries: int) -> dict[str, str] | None:
    raw_target = step.metadata.get("retry_budget_patch_target") or step.metadata.get("arl_patch_target")
    if not isinstance(raw_target, dict):
        return None
    path = _metadata_text(raw_target.get("path"))
    before = _metadata_text(
        raw_target.get("before")
        or raw_target.get("current_line")
        or raw_target.get("current_text")
    )
    after = _metadata_text(
        raw_target.get("after")
        or raw_target.get("replacement_line")
        or raw_target.get("replacement_text")
    )
    if not path or not before or not after or before == after:
        return None
    return {"path": path, "before": before, "after": after}


def _metadata_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _retry_budget_config_diff(step_name: str, current_retry_budget: int, allowed_retries: int) -> str:
    return f"""# Non-runnable config diff: trace did not include a target file and line context.
# To emit an applyable unified diff, instrument the trace step with metadata:
# retry_budget_patch_target.path, retry_budget_patch_target.before, retry_budget_patch_target.after.
retry_budget:
-  {step_name}: {current_retry_budget}
+  {step_name}: {allowed_retries}

Reason:
- {step_name} made {current_retry_budget} additional attempts after the first.
- Recommended fail-closed retry budget is {allowed_retries}.
"""


def _unified_retry_budget_patch(target: dict[str, str]) -> str:
    path = target["path"].replace("\\", "/")
    diff_lines = list(
        difflib.unified_diff(
            _split_patch_text(target["before"]),
            _split_patch_text(target["after"]),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    return f"diff --git a/{path} b/{path}\n" + "\n".join(diff_lines) + "\n"


def _split_patch_text(text: str) -> list[str]:
    if text.endswith("\n"):
        return text.splitlines()
    return f"{text}\n".splitlines()
