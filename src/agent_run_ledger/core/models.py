from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

SCHEMA_VERSION = "0.1"
PatchType = Literal["unified_diff", "code_snippet", "config_diff", "regression_test"]
PATCH_TYPES: tuple[str, ...] = ("unified_diff", "code_snippet", "config_diff", "regression_test")


class TraceValidationError(ValueError):
    """Raised when a trace bundle does not satisfy the V0 schema."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _token_usage(data: dict[str, Any]) -> tuple[int, int]:
    usage = data.get("token_usage") or {}
    return _as_int(usage.get("input")), _as_int(usage.get("output"))


@dataclass(frozen=True)
class RunRecord:
    id: str
    workflow: str
    framework: str
    provider: str
    model: str
    started_at: str
    ended_at: str
    success_label: str
    prompt_hash: str = ""
    config_hash: str = ""
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        return cls(
            id=str(data.get("id") or f"run_{uuid4().hex[:12]}"),
            workflow=str(data.get("workflow") or "unknown-workflow"),
            framework=str(data.get("framework") or "unknown-framework"),
            provider=str(data.get("provider") or "unknown-provider"),
            model=str(data.get("model") or "unknown-model"),
            started_at=str(data.get("started_at") or utc_now_iso()),
            ended_at=str(data.get("ended_at") or data.get("started_at") or utc_now_iso()),
            success_label=str(data.get("success_label") or "unknown"),
            prompt_hash=str(data.get("prompt_hash") or ""),
            config_hash=str(data.get("config_hash") or ""),
            total_cost_usd=_as_float(data.get("total_cost_usd")),
            total_latency_ms=_as_int(data.get("total_latency_ms")),
            total_input_tokens=_as_int(data.get("total_input_tokens")),
            total_output_tokens=_as_int(data.get("total_output_tokens")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow": self.workflow,
            "framework": self.framework,
            "provider": self.provider,
            "model": self.model,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "success_label": self.success_label,
            "prompt_hash": self.prompt_hash,
            "config_hash": self.config_hash,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "total_latency_ms": self.total_latency_ms,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


@dataclass(frozen=True)
class StepRecord:
    """One observed step in a run.

    `retry_count` is the number of additional attempts after the first attempt.
    A step with `retry_count=4` was attempted 5 total times.
    """

    id: str
    run_id: str
    step_type: str
    name: str
    started_at: str
    ended_at: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retry_count: int = 0
    error: str | None = None
    redaction_mode: str = "metadata_only"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], run_id: str) -> StepRecord:
        input_tokens, output_tokens = _token_usage(data)
        return cls(
            id=str(data.get("id") or f"step_{uuid4().hex[:12]}"),
            run_id=run_id,
            step_type=str(data.get("type") or data.get("step_type") or "unknown"),
            name=str(data.get("name") or "unnamed-step"),
            started_at=str(data.get("started_at") or utc_now_iso()),
            ended_at=str(data.get("ended_at") or data.get("started_at") or utc_now_iso()),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_as_float(data.get("cost_usd")),
            retry_count=_as_int(data.get("retry_count")),
            error=str(data["error"]) if data.get("error") is not None else None,
            redaction_mode=str(data.get("redaction_mode") or "metadata_only"),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "type": self.step_type,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "token_usage": {"input": self.input_tokens, "output": self.output_tokens},
            "cost_usd": round(self.cost_usd, 8),
            "retry_count": self.retry_count,
            "redaction_mode": self.redaction_mode,
        }
        if self.error is not None:
            data["error"] = self.error
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass(frozen=True)
class PrescriptionRecord:
    id: str
    run_id: str
    severity: str
    root_cause: str
    one_line_fix: str
    evidence: list[str]
    patch_type: PatchType
    patch: str
    expected_impact: dict[str, Any]
    regression_test_template: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], run_id: str) -> PrescriptionRecord:
        return cls(
            id=str(data.get("id") or f"rx_{uuid4().hex[:12]}"),
            run_id=run_id,
            severity=str(data.get("severity") or "medium"),
            root_cause=str(data.get("root_cause") or "unknown"),
            one_line_fix=str(data.get("one_line_fix") or ""),
            evidence=list(data.get("evidence") or []),
            patch_type=_patch_type(data.get("patch_type") or "code_snippet"),
            patch=str(data.get("patch") or ""),
            expected_impact=dict(data.get("expected_impact") or {}),
            regression_test_template=str(data.get("regression_test_template") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "root_cause": self.root_cause,
            "one_line_fix": self.one_line_fix,
            "evidence": self.evidence,
            "patch_type": self.patch_type,
            "patch": self.patch,
            "expected_impact": self.expected_impact,
            "regression_test_template": self.regression_test_template,
        }


@dataclass(frozen=True)
class TraceBundle:
    run: RunRecord
    steps: list[StepRecord]
    prescriptions: list[PrescriptionRecord] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceBundle:
        schema_version = str(data.get("schema_version") or SCHEMA_VERSION)
        if schema_version != SCHEMA_VERSION:
            raise TraceValidationError(f"unsupported schema_version={schema_version!r}")
        if "run" not in data:
            raise TraceValidationError("trace bundle missing 'run'")
        run = RunRecord.from_dict(dict(data["run"]))
        steps = [StepRecord.from_dict(dict(step), run.id) for step in data.get("steps") or []]
        if not steps:
            raise TraceValidationError("trace bundle must contain at least one step")
        run = _with_computed_totals(run, steps)
        prescriptions = [
            PrescriptionRecord.from_dict(dict(item), run.id)
            for item in data.get("prescriptions") or []
        ]
        bundle = cls(run=run, steps=steps, prescriptions=prescriptions, schema_version=schema_version)
        bundle.validate()
        return bundle

    def validate(self) -> None:
        if not self.run.id:
            raise TraceValidationError("run.id is required")
        step_ids = set()
        for step in self.steps:
            if not step.id:
                raise TraceValidationError("step.id is required")
            if step.id in step_ids:
                raise TraceValidationError(f"duplicate step.id={step.id!r}")
            step_ids.add(step.id)
            if step.run_id != self.run.id:
                raise TraceValidationError(f"step {step.id!r} references wrong run")
            if step.retry_count < 0:
                raise TraceValidationError(f"step {step.id!r} has negative retry_count")
        for prescription in self.prescriptions:
            _validate_patch_artifact(prescription)

    def with_prescriptions(self, prescriptions: list[PrescriptionRecord]) -> TraceBundle:
        return TraceBundle(
            schema_version=self.schema_version,
            run=self.run,
            steps=self.steps,
            prescriptions=prescriptions,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.prescriptions:
            data["prescriptions"] = [item.to_dict() for item in self.prescriptions]
        return data


def _with_computed_totals(run: RunRecord, steps: list[StepRecord]) -> RunRecord:
    total_cost = run.total_cost_usd or sum(step.cost_usd for step in steps)
    input_tokens = run.total_input_tokens or sum(step.input_tokens for step in steps)
    output_tokens = run.total_output_tokens or sum(step.output_tokens for step in steps)
    return RunRecord(
        id=run.id,
        workflow=run.workflow,
        framework=run.framework,
        provider=run.provider,
        model=run.model,
        started_at=run.started_at,
        ended_at=run.ended_at,
        success_label=run.success_label,
        prompt_hash=run.prompt_hash,
        config_hash=run.config_hash,
        total_cost_usd=total_cost,
        total_latency_ms=run.total_latency_ms,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
    )


def _patch_type(value: Any) -> PatchType:
    patch_type = str(value)
    if patch_type not in PATCH_TYPES:
        raise TraceValidationError(f"unsupported patch_type={patch_type!r}")
    return patch_type  # type: ignore[return-value]


def _validate_patch_artifact(prescription: PrescriptionRecord) -> None:
    patch = prescription.patch.strip()
    if len(patch) < 64:
        raise TraceValidationError(
            f"prescription {prescription.id!r} patch artifact is too short"
        )
    validator = {
        "unified_diff": _is_unified_diff,
        "code_snippet": _is_code_snippet,
        "config_diff": _is_config_diff,
        "regression_test": _is_regression_test,
    }[prescription.patch_type]
    if not validator(patch):
        raise TraceValidationError(
            f"prescription {prescription.id!r} patch artifact does not match "
            f"patch_type={prescription.patch_type!r}"
        )


def _is_unified_diff(patch: str) -> bool:
    lines = patch.splitlines()
    return (
        bool(lines)
        and (lines[0].startswith("diff --git ") or lines[0].startswith("--- "))
        and any(line.startswith("--- ") for line in lines)
        and any(line.startswith("+++ ") for line in lines)
        and any(line.startswith("@@") for line in lines)
    )


def _is_code_snippet(patch: str) -> bool:
    return "\n" in patch and any(token in patch for token in ("def ", "class ", "=", "return "))


def _is_config_diff(patch: str) -> bool:
    return (
        _is_unified_diff(patch)
        or ("\n" in patch and any(line.lstrip().startswith(("+", "-")) for line in patch.splitlines()))
    )


def _is_regression_test(patch: str) -> bool:
    return "def test_" in patch or "class Test" in patch
