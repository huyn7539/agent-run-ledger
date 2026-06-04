from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_run_ledger.core.models import (
    RunRecord,
    StepRecord,
    TraceBundle,
    classify_error,
    utc_now_iso,
)
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.provenance import compute_provenance_hash
from agent_run_ledger.core.retries import AttemptFacts, collapse_retry_groups
from agent_run_ledger.core.storage import save_bundle

class NoSpansCapturedError(RuntimeError):
    """Raised when the OpenAI trace processor finishes without any spans."""


class OpenAILedgerTraceProcessor:
    """Best-effort OpenAI Agents SDK trace processor.

    The core package stays provider-neutral. This adapter accepts SDK trace/span
    objects through duck-typed callback methods and stores a local run bundle.
    """

    def __init__(
        self,
        db_path: Path,
        workflow: str = "openai-agent-workflow",
        model: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.workflow = workflow
        # Task 45: the default Responses API emits ONLY response spans, whose
        # export carries NO model field — so a real run has no model discoverable
        # from spans. The instrumenting app DOES know its model when it calls
        # Runner.run; this hint is used ONLY when no span carries a model. Without
        # a model, cost_on_read can't price tokens and falls back to the $0 cache.
        self.model_hint = model
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
        # Build a content-free fact projection per span, sorted deterministically
        # by (started_at, id) — mirrors provenance.py so step identity is stable.
        # on_span_end fires in COMPLETION order, not start order, so this sort is
        # load-bearing for correct retry adjacency.
        raw_facts = [_span_facts(span, idx) for idx, span in enumerate(self._spans, start=1)]
        for f in raw_facts:
            # Preserve prior fallback: a span without its own start uses the trace
            # start (kept here, not in the free helper, which has no trace context).
            if not f["started_at"]:
                f["started_at"] = self._started_at
        facts = sorted(raw_facts, key=lambda f: (f["started_at"], f["id"]))
        model = next((f["model"] for f in facts if f["model"] != "unknown"), "unknown")
        # Task 45: trace-extracted model wins (Chat Completions generation spans
        # carry it). Only when NO span exposed a model do we fall back to the
        # app-supplied hint — the Responses API never serializes the model.
        if model == "unknown" and self.model_hint:
            model = self.model_hint

        # NEW-4: derive retry loops from repeated same-target tool spans. The
        # input fingerprint (a transient digest, never stored) is what lets us
        # tell a genuine loop from legitimate repeated calls. Only function/tool
        # spans WITHOUT an app-supplied retry_count are eligible for derivation;
        # the explicit custom-span retry_count path is preserved unchanged.
        attempts = [
            AttemptFacts(
                index=i,
                name=f["name"],
                span_kind=f["span_kind"],
                parent_id=f["parent_step_id"],
                started_at=f["started_at"],
                ended_at=f["ended_at"],
                has_error=f["raw_error"] is not None,
                error_class=classify_error(f["raw_error"]),
                input_fingerprint=(f["input_fingerprint"] if not f["explicit_retry"] else None),
            )
            for i, f in enumerate(facts)
        ]
        groups = collapse_retry_groups(attempts)

        steps = [self._step_from_group([facts[i] for i in group]) for group in groups]
        total_cost = sum(s.cost_usd for s in steps)
        total_input = sum(s.input_tokens for s in steps)
        total_output = sum(s.output_tokens for s in steps)
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
        bundle = TraceBundle(run=run, steps=steps)
        # L5: stamp the provenance hash LOCALLY at capture — the un-backfillable
        # seed of proof-of-real. Computed over immutable facts only (no cost).
        return replace(bundle, run=replace(run, provenance_hash=compute_provenance_hash(bundle)))

    def _step_from_group(self, group: list[dict[str, Any]]) -> StepRecord:
        """Build one StepRecord from a group of 1+ attempt facts.

        A singleton preserves prior single-span behavior exactly. A multi-attempt
        group is a DERIVED retry loop: retry_count = N-1, identity/timestamps from
        the first/last attempt, tokens + cost SUMMED across attempts (the
        wasted-cost estimate divides cost_usd by the attempt count, so summing is
        required), and error_class from the LAST attempt (the terminal state that
        drives severity)."""
        first = group[0]
        last = group[-1]
        derived_retries = len(group) - 1
        # An explicit app-supplied retry_count (custom-span data) is never derived;
        # such spans are singletons here, so use their own retry_count. A derived
        # group uses N-1.
        retry_count = first["retry_count"] if derived_retries == 0 else derived_retries
        reported_costs = [g["reported_cost"] for g in group if g["reported_cost"] is not None]
        reported_cost = sum(reported_costs) if reported_costs else None
        raw_error = last["raw_error"]
        return StepRecord(
            id=first["id"],
            run_id=self._trace_id,
            # L4: preserve the call-graph edge + OTEL-aligned span kind.
            parent_step_id=first["parent_step_id"],
            span_kind=first["span_kind"],
            step_type=first["step_type"],
            name=first["name"],
            started_at=first["started_at"],
            ended_at=last["ended_at"],
            input_tokens=sum(g["input_tokens"] for g in group),
            output_tokens=sum(g["output_tokens"] for g in group),
            # Task 45 / L7: cached + reasoning token FACTS, summed across attempts.
            cached_input_tokens=sum(g["cached_input"] for g in group),
            reasoning_tokens=sum(g["reasoning"] for g in group),
            # L7: the span-reported cost FACT, summed; None only if no attempt
            # reported one (so cost_on_read can fall back to tokens).
            provider_reported_cost_usd=reported_cost,
            cost_usd=sum(g["cost"] for g in group),
            retry_count=retry_count,
            error=raw_error,
            # L8: bounded error_class at the chokepoint (raw message dropped).
            error_class=classify_error(raw_error),
            redaction_mode="metadata_only",
            metadata=first["metadata"],
        )


def _span_facts(span: dict[str, Any], idx: int) -> dict[str, Any]:
    """Project one raw span into a flat fact dict + a TRANSIENT input fingerprint.

    The fingerprint is a digest of the raw tool input; it is used ONLY for
    retry-loop grouping and is NEVER stored (it does not flow into StepRecord).
    Computed here, while the raw span is still in scope, because input is redacted
    once a StepRecord is constructed (models.sanitize_metadata)."""
    span_data = _extract_span_data(span)
    usage = _extract_usage(span)
    cached_input, reasoning = _extract_token_details(span)
    span_kind = span_data.get("type") or span.get("type") or span.get("span_type")
    parent_id = (
        span.get("parent_id") or span.get("parent_span_id") or span_data.get("parent_id")
    )
    explicit_retry = _has_explicit_retry(span)
    return {
        "id": str(span.get("span_id") or span.get("id") or f"span_{idx}"),
        "parent_step_id": str(parent_id) if parent_id is not None else None,
        "span_kind": str(span_kind) if span_kind is not None else None,
        "step_type": str(span_kind or "span"),
        "name": str(
            span_data.get("name") or span.get("name") or span.get("operation") or f"span_{idx}"
        ),
        "started_at": str(span.get("started_at") or span.get("start_time") or ""),
        "ended_at": str(span.get("ended_at") or span.get("end_time") or utc_now_iso()),
        "input_tokens": usage[0],
        "output_tokens": usage[1],
        "cached_input": cached_input,
        "reasoning": reasoning,
        "cost": _extract_cost(span),
        "reported_cost": _extract_reported_cost(span),
        "retry_count": _extract_retry_count(span),
        "explicit_retry": explicit_retry,
        "model": _extract_model(span),
        "raw_error": _extract_error(span),
        "metadata": _safe_metadata(span),
        "input_fingerprint": _input_fingerprint(span_data),
    }


def _has_explicit_retry(span: dict[str, Any]) -> bool:
    """True if the app instrumented retry_count directly (custom-span data). Such
    spans are NOT eligible for trace-derivation — their count is authoritative."""
    custom_data = _extract_custom_data(span)
    return span.get("retry_count") is not None or custom_data.get("retry_count") is not None


def _input_fingerprint(span_data: dict[str, Any]) -> str | None:
    """Return a stable digest of the span's raw tool input, or None if absent.

    Transient + content-free: the digest is used only to compare same-vs-different
    input for retry grouping; the raw input is discarded and never stored, so this
    leaks nothing (a digest is not reversible to content)."""
    raw_input = span_data.get("input")
    if raw_input is None:
        return None
    try:
        canonical = json.dumps(raw_input, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canonical = str(raw_input)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_trace_processor(
    db_path: Path,
    workflow: str = "openai-agent-workflow",
    model: str | None = None,
) -> OpenAILedgerTraceProcessor:
    return OpenAILedgerTraceProcessor(db_path=db_path, workflow=workflow, model=model)


def bundle_from_recorded_trace(recorded: dict[str, Any], model: str | None = None) -> TraceBundle:
    if not recorded.get("spans"):
        raise ValueError("recorded trace contains no spans")
    processor = OpenAILedgerTraceProcessor(Path(":memory:"), model=model)
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


def _usage_dict(span: dict[str, Any]) -> dict[str, Any]:
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
    return usage


def _extract_usage(span: dict[str, Any]) -> tuple[int, int]:
    usage = _usage_dict(span)
    return int(usage.get("input") or usage.get("input_tokens") or 0), int(
        usage.get("output") or usage.get("output_tokens") or 0
    )


def _extract_token_details(span: dict[str, Any]) -> tuple[int, int]:
    """Return (cached_input_tokens, reasoning_tokens) from the nested usage details
    the live SDK emits (Task 45 / L7). The Responses-API usage nests
    ``input_tokens_details.cached_tokens`` and
    ``output_tokens_details.reasoning_tokens``; both are capture-only facts that
    materially change cost (cached is discounted, reasoning bills at output rate).
    Absent details -> 0, the honest default."""
    usage = _usage_dict(span)
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    if not isinstance(output_details, dict):
        output_details = {}
    cached = int(input_details.get("cached_tokens") or 0)
    reasoning = int(output_details.get("reasoning_tokens") or 0)
    return cached, reasoning


def _extract_custom_data(span: dict[str, Any]) -> dict[str, Any]:
    span_data = _extract_span_data(span)
    data = span_data.get("data") or span.get("metadata") or {}
    return _object_to_dict(data)


def _extract_cost(span: dict[str, Any]) -> float:
    custom_data = _extract_custom_data(span)
    value = span.get("cost_usd") or custom_data.get("cost_usd") or 0.0
    return float(value)


def _extract_reported_cost(span: dict[str, Any]) -> float | None:
    """Return the provider-reported cost FACT, or None if the span did not
    report one (L7). Distinct from ``_extract_cost`` (which coalesces to 0.0):
    None must mean "not reported", so cost_on_read can fall back to tokens."""
    custom_data = _extract_custom_data(span)
    if span.get("cost_usd") is not None:
        return float(span["cost_usd"])
    if custom_data.get("cost_usd") is not None:
        return float(custom_data["cost_usd"])
    return None


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
    metadata = _extract_custom_data(span)
    return metadata if isinstance(metadata, dict) else {}
