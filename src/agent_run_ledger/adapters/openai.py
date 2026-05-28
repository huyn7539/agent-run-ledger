from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_run_ledger.core.models import RunRecord, StepRecord, TraceBundle, utc_now_iso
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.storage import save_bundle

_REDACTED = "[redacted]"
_REDACTED_METADATA_KEYS = {"input", "output"}


class NoSpansCapturedError(RuntimeError):
    """Raised when the OpenAI trace processor finishes without any spans."""


class OpenAILedgerTraceProcessor:
    """Best-effort OpenAI Agents SDK trace processor.

    The core package stays provider-neutral. This adapter accepts SDK trace/span
    objects through duck-typed callback methods and stores a local run bundle.
    """

    def __init__(self, db_path: Path, workflow: str = "openai-agent-workflow") -> None:
        self.db_path = db_path
        self.workflow = workflow
        self._trace_id = f"openai_{uuid4().hex[:12]}"
        self._started_at = utc_now_iso()
        self._ended_at = self._started_at
        self._spans: list[dict[str, Any]] = []

    def on_trace_start(self, trace: Any) -> None:
        data = _object_to_dict(trace)
        self._trace_id = str(data.get("trace_id") or data.get("id") or self._trace_id)
        self.workflow = str(data.get("name") or data.get("workflow_name") or self.workflow)
        self._started_at = str(data.get("started_at") or utc_now_iso())

    def on_trace_end(self, trace: Any) -> None:
        data = _object_to_dict(trace)
        self._ended_at = str(data.get("ended_at") or utc_now_iso())
        bundle = self._bundle_from_spans()
        prescriptions = analyze_bundle(bundle)
        save_bundle(self.db_path, bundle.with_prescriptions(prescriptions))

    def on_span_start(self, span: Any) -> None:
        return None

    def on_span_end(self, span: Any) -> None:
        self._spans.append(_object_to_dict(span))

    def force_flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def _bundle_from_spans(self) -> TraceBundle:
        if not self._spans:
            raise NoSpansCapturedError(
                f"OpenAI trace {self._trace_id!r} ended with zero captured spans"
            )
        steps: list[StepRecord] = []
        total_cost = 0.0
        total_input = 0
        total_output = 0
        model = "unknown"
        for idx, span in enumerate(self._spans, start=1):
            span_data = _extract_span_data(span)
            usage = _extract_usage(span)
            cost = _extract_cost(span)
            retry_count = _extract_retry_count(span)
            if model == "unknown":
                model = _extract_model(span)
            total_cost += cost
            total_input += usage[0]
            total_output += usage[1]
            steps.append(
                StepRecord(
                    id=str(span.get("span_id") or span.get("id") or f"span_{idx}"),
                    run_id=self._trace_id,
                    step_type=str(
                        span_data.get("type")
                        or span.get("type")
                        or span.get("span_type")
                        or "span"
                    ),
                    name=str(
                        span_data.get("name")
                        or span.get("name")
                        or span.get("operation")
                        or f"span_{idx}"
                    ),
                    started_at=str(span.get("started_at") or span.get("start_time") or self._started_at),
                    ended_at=str(span.get("ended_at") or span.get("end_time") or utc_now_iso()),
                    input_tokens=usage[0],
                    output_tokens=usage[1],
                    cost_usd=cost,
                    retry_count=retry_count,
                    error=_extract_error(span),
                    redaction_mode="metadata_only",
                    metadata=_safe_metadata(span),
                )
            )
        run = RunRecord(
            id=self._trace_id,
            workflow=self.workflow,
            framework="openai-agents-python",
            provider="openai",
            model=model,
            started_at=self._started_at,
            ended_at=self._ended_at,
            success_label="unknown",
            total_cost_usd=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
        )
        return TraceBundle(run=run, steps=steps)


def make_trace_processor(db_path: Path, workflow: str = "openai-agent-workflow") -> OpenAILedgerTraceProcessor:
    return OpenAILedgerTraceProcessor(db_path=db_path, workflow=workflow)


def bundle_from_recorded_trace(recorded: dict[str, Any]) -> TraceBundle:
    if not recorded.get("spans"):
        raise ValueError("recorded trace contains no spans")
    processor = OpenAILedgerTraceProcessor(Path(":memory:"))
    processor.on_trace_start(recorded["trace"])
    for span in recorded.get("spans") or []:
        processor.on_span_end(span)
    processor._ended_at = str(recorded["trace"].get("ended_at") or utc_now_iso())
    return processor._bundle_from_spans()


def _object_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return {str(k): _clean_value(v) for k, v in obj.items()}
    for method_name in ("export", "to_dict", "model_dump", "dict"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                result = method()
            except TypeError:
                continue
            if isinstance(result, dict):
                return {str(k): _clean_value(v) for k, v in result.items()}
    data = getattr(obj, "__dict__", {})
    return {str(k): _clean_value(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    exported = _object_to_dict(value)
    return exported if exported else str(value)


def _extract_span_data(span: dict[str, Any]) -> dict[str, Any]:
    data = span.get("span_data") or span.get("data") or {}
    return _object_to_dict(data)


def _extract_usage(span: dict[str, Any]) -> tuple[int, int]:
    span_data = _extract_span_data(span)
    custom_data = _extract_custom_data(span)
    usage = (
        span.get("usage")
        or span.get("token_usage")
        or span_data.get("usage")
        or span_data.get("token_usage")
        or custom_data.get("usage")
        or custom_data.get("token_usage")
        or {}
    )
    if not isinstance(usage, dict):
        usage = _object_to_dict(usage)
    return int(usage.get("input") or usage.get("input_tokens") or 0), int(
        usage.get("output") or usage.get("output_tokens") or 0
    )


def _extract_custom_data(span: dict[str, Any]) -> dict[str, Any]:
    span_data = _extract_span_data(span)
    data = span_data.get("data") or span.get("metadata") or {}
    return _object_to_dict(data)


def _extract_cost(span: dict[str, Any]) -> float:
    custom_data = _extract_custom_data(span)
    value = span.get("cost_usd") or custom_data.get("cost_usd") or 0.0
    return float(value)


def _extract_retry_count(span: dict[str, Any]) -> int:
    custom_data = _extract_custom_data(span)
    value = span.get("retry_count") or custom_data.get("retry_count") or 0
    return int(value)


def _extract_model(span: dict[str, Any]) -> str:
    span_data = _extract_span_data(span)
    custom_data = _extract_custom_data(span)
    return str(span.get("model") or span_data.get("model") or custom_data.get("model") or "unknown")


def _extract_error(span: dict[str, Any]) -> str | None:
    span_data = _extract_span_data(span)
    custom_data = _extract_custom_data(span)
    error = span.get("error") or span.get("exception") or span_data.get("error") or custom_data.get("error")
    if error is None:
        return None
    return str(error)


def _safe_metadata(span: dict[str, Any]) -> dict[str, Any]:
    redacted = _redact_sensitive_metadata(span)
    return redacted if isinstance(redacted, dict) else {}


def _redact_sensitive_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _REDACTED
            if str(key).lower() in _REDACTED_METADATA_KEYS
            else _redact_sensitive_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive_metadata(item) for item in value]
    return value
