from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import math
import re
from typing import Any, Literal
from uuid import uuid4

SCHEMA_VERSION = "0.1"


def _parse_version(value: str) -> tuple[int, int]:
    """Parse ``"major.minor"`` into an ``(int, int)`` tuple.

    Parsing to ints (not string compare) is load-bearing: ``"0.10"`` must sort
    above ``"0.9"``, which a lexical compare gets backwards.
    """
    parts = str(value).split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError) as exc:
        raise TraceValidationError(f"malformed schema_version={value!r}") from exc
    return (major, minor)


def is_version_compatible(record_version: str, reader_version: str = SCHEMA_VERSION) -> bool:
    """Return True if a reader at *reader_version* can read a record written at
    *record_version* (L1 compatibility policy).

    Policy: accept SAME major and SAME-OR-LOWER minor. Lower-minor records
    upcast cleanly (``from_dict`` fills missing fields with defaults). A HIGHER
    minor is rejected (the record may carry fact fields the reader cannot
    interpret); a different MAJOR is rejected.
    """
    r_major, r_minor = _parse_version(record_version)
    s_major, s_minor = _parse_version(reader_version)
    if r_major != s_major:
        return False
    return r_minor <= s_minor


PatchType = Literal["unified_diff", "code_snippet", "config_diff", "regression_test"]
PATCH_TYPES: tuple[str, ...] = ("unified_diff", "code_snippet", "config_diff", "regression_test")

# L7: closed set of billing modes. The FULL enum is locked now — adding a member
# later is a no-op; adding the column later would be a base migration.
BILLING_MODES: tuple[str, ...] = (
    "pay_per_use",
    "subscription",
    "enterprise_contract",
    "local",
    "unknown",
)
_REDACTED = "[redacted]"
_ALLOWED_METADATA_KEYS = {
    "after",
    "arl_patch_target",
    "before",
    "cost_usd",
    "current_line",
    "current_text",
    "input_tokens",
    "max_tokens",
    "model",
    "output_tokens",
    "path",
    "replacement_line",
    "replacement_text",
    "retry_budget_patch_target",
    "retry_count",
    "severity",
    "step_type",
    "total_tokens",
    "usage",
}
_SAFE_ERROR_MESSAGE = "details redacted"

# L6: a hash-or-empty token — empty string, or a lowercase hex string. The
# alphabet is hex-only so a human-readable label (e.g. "prompt_retry_loop_v1")
# is rejected. Length is left flexible (sha256 = 64) to avoid over-constraining
# a future digest algorithm; the alphabet is the load-bearing check.
_HEX_RE = re.compile(r"\A[0-9a-f]*\Z")


def _is_hex_or_empty(value: str) -> bool:
    """Return True if *value* is empty or a lowercase-hex string (L6)."""
    return bool(_HEX_RE.match(value))


# L8 / LR8: a small CLOSED vocabulary of error classes + an "Other" bucket. The
# SHAPE (a bounded typed label) is locked; the MEMBERSHIP is lean — split new
# classes out of "Other" only when real traces show they're worth it. Keys are
# matched against the leading ClassName token of an error (case-insensitive,
# substring), so "TimeoutError" -> "Timeout", "RateLimitError" -> "RateLimit".
ERROR_CLASSES: tuple[str, ...] = (
    "Timeout",
    "RateLimit",
    "Validation",
    "Auth",
    "Network",
    "Other",
)

# Map known Python/SDK exception class-name fragments to a bounded label. Order
# matters (first substring hit wins); everything unmatched falls to "Other".
_ERROR_CLASS_MAP: tuple[tuple[str, str], ...] = (
    ("timeout", "Timeout"),
    ("ratelimit", "RateLimit"),
    ("rate_limit", "RateLimit"),
    ("toomanyrequests", "RateLimit"),
    ("valueerror", "Validation"),
    ("validation", "Validation"),
    ("typeerror", "Validation"),
    ("keyerror", "Validation"),
    ("auth", "Auth"),
    ("permission", "Auth"),
    ("forbidden", "Auth"),
    ("unauthorized", "Auth"),
    ("connection", "Network"),
    ("network", "Network"),
    ("socket", "Network"),
    ("dns", "Network"),
)


def classify_error(error: Any) -> str | None:
    """Map an exception or error string to a bounded ``error_class`` LABEL (L8).

    The raw message is DROPPED here — only the class name is inspected, and only
    a bounded label is returned, so no prompt/content can leak through this path.
    Returns None when there is no error.

    - For an Exception, the class name is ``type(error).__name__``.
    - For a string like ``"TimeoutError: <message>"``, only the leading
      identifier token (before the first ``:`` / whitespace) is used; the
      message after it is never inspected or returned.
    """
    if error is None:
        return None
    if isinstance(error, BaseException):
        name = type(error).__name__
    else:
        # take only the leading class-name-shaped token; drop everything after.
        text = str(error).strip()
        token = re.split(r"[:\s]", text, maxsplit=1)[0]
        name = token
    lowered = name.lower()
    for fragment, label in _ERROR_CLASS_MAP:
        if fragment in lowered:
            return label
    return "Other"


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
    """A single agent run (one workflow invocation) — FACTS only.

    REDACTION CONTRACT (L13): any future column that holds CONTENT (not a
    fact/judgment scalar) MUST register a sensitivity tag and route through the
    same redaction + consent contract as step metadata (ADR-001) and the egress
    chokepoint (ADR-002); no brick may add a raw-content column without a
    redaction projection. The provenance_hash and the outcome slot are
    attach-points, NOT content sinks. ``workflow`` and ``success_label`` are
    content-bearing labels: raw locally, stripped at egress (L9).
    """

    id: str
    workflow: str
    framework: str
    provider: str
    model: str
    started_at: str
    ended_at: str
    # L14: success_label is the adapter's PROVISIONAL SELF-REPORT — a fact about
    # what the run CLAIMED its terminal status was, NOT a verdict. The future
    # outcome_json slot carries the higher-trust external verdict. Detectors MUST
    # NOT treat success_label as ground truth, and the two must never be
    # overloaded onto one field. (Renaming to terminal_status is operator-
    # optional and deliberately NOT done here — §6.)
    success_label: str
    prompt_hash: str = ""
    config_hash: str = ""
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # L1: per-record shape stamp — "what shape was this record written under".
    # Distinct from the file-level PRAGMA user_version (which DDL the *file* has
    # had). A single .db legitimately holds mixed-version rows once upgrade-on-
    # read is in play; this stamp is how a reader knows which.
    schema_version: str = SCHEMA_VERSION
    # L7: cost FACTS. billing_mode is a closed enum (cached/batch/subscription
    # runs make a token-only cost wrong by up to 10x). price_table_version makes
    # any cached cost reproducible + staleness-detectable. total_cost_usd above
    # is now a CACHE only (LR2); the source of truth is cost.cost_on_read().
    billing_mode: str = "unknown"
    price_table_version: str | None = None
    # L5: 'sha256:<hex>' over the canonical immutable facts (EXCLUDING cost +
    # derived). Computed locally at capture; pinned by a golden test. NULL until
    # the capture path stamps it.
    provenance_hash: str | None = None
    # LR1: the B2 attach-point. A single nullable column reserved for immutable
    # ground-truth FACTS (external/human/signed attestations). NULL = unknown
    # (the honest default). NEVER computed-on-read; the inner attestation shape
    # is deliberately NOT frozen. ONE column, not outcome/outcome_source/
    # outcome_at — three columns would start designing the B2 shape (forbidden).
    outcome_json: str | None = None

    def __post_init__(self) -> None:
        if self.billing_mode not in BILLING_MODES:
            raise TraceValidationError(
                f"billing_mode={self.billing_mode!r} not in {BILLING_MODES}"
            )
        # L6: prompt_hash / config_hash are LOCAL dedup/integrity tokens, NOT a
        # provenance proof (that's provenance_hash, L5). They must be a canonical
        # lowercase-hex digest or empty — never a human-readable label. They sit
        # in a backfill-impossible column; a label there is a permanent corpus
        # defect.
        if not _is_hex_or_empty(self.prompt_hash):
            raise TraceValidationError(
                f"prompt_hash must be lowercase-hex or empty, got {self.prompt_hash!r}"
            )
        if not _is_hex_or_empty(self.config_hash):
            raise TraceValidationError(
                f"config_hash must be lowercase-hex or empty, got {self.config_hash!r}"
            )
        # LR1: outcome_json, when present, must be well-formed JSON (facts slot).
        if self.outcome_json is not None:
            try:
                json.loads(self.outcome_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise TraceValidationError(f"outcome_json must be valid JSON: {exc}") from exc

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        price_table_version = data.get("price_table_version")
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
            schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
            billing_mode=str(data.get("billing_mode") or "unknown"),
            price_table_version=(
                str(price_table_version) if price_table_version is not None else None
            ),
            provenance_hash=(
                str(data["provenance_hash"]) if data.get("provenance_hash") is not None else None
            ),
            outcome_json=(
                str(data["outcome_json"]) if data.get("outcome_json") is not None else None
            ),
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
            "schema_version": self.schema_version,
            "billing_mode": self.billing_mode,
            "price_table_version": self.price_table_version,
            "provenance_hash": self.provenance_hash,
            "outcome_json": self.outcome_json,
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
    # L4: the call-graph edge. parent_step_id NULL = root span; span_kind is the
    # OTEL-aligned kind (agent/llm/tool/handoff, LR3). The SDK emits span TREES;
    # this edge is the only structural fact lost when the tree is flattened.
    parent_step_id: str | None = None
    span_kind: str | None = None
    # Trace-derived retry detection FACTS (content-free, computed at capture):
    #   retry_scope     = the stable ancestor that groups retries of one call site
    #                     across turns (a real agentic retry spans multiple turns,
    #                     so the immediate parent differs; the agent-span ancestor
    #                     is stable). NULL when no scope could be resolved.
    #   input_fingerprint = a one-way digest of the raw tool input, used to tell a
    #                     genuine retry (same input) from legitimate repetition.
    #                     A digest carries NO recoverable content (leak-safe). NULL
    #                     when the input was not captured.
    # Both are FACTS stored on the immutable base; the retry_count COLLAPSE is a
    # JUDGMENT computed ON READ (prescriptions.derive_retry_steps), never baked in,
    # so a future detector fix can re-derive from the raw corpus.
    retry_scope: str | None = None
    input_fingerprint: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    # L7: observed cost FACTS the API returns. cached_input/reasoning tokens are
    # un-backfillable (only present at capture). provider_reported_cost_usd is
    # the cost FACT (what the provider said it charged; NULL if not reported).
    # cost_usd below is a cache/derived value, NOT authoritative (LR2).
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    provider_reported_cost_usd: float | None = None
    cost_usd: float = 0.0
    retry_count: int = 0
    error: str | None = None
    # L8: typed, bounded error LABEL (e.g. "Timeout"). Derived from the error
    # CLASS at the chokepoint; the raw message is dropped, so this never leaks
    # content. The existing `error` field stays the redacted constant — this is
    # ADDED alongside, the chokepoint is kept (operator decision 2026-05-29).
    error_class: str | None = None
    redaction_mode: str = "metadata_only"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "error",
            sanitize_error(self.error) if self.error is not None else None,
        )
        # L8 hardening (Codex adversarial finding 2026-05-29): error_class is
        # ALWAYS bounded to the closed vocabulary at the chokepoint, regardless of
        # construction path. A caller (esp. the import path) cannot inject raw
        # content as error_class — classify_error maps anything unrecognized to
        # "Other". Idempotent over the vocabulary, so a stored label re-loads
        # unchanged.
        object.__setattr__(self, "error_class", classify_error(self.error_class))
        object.__setattr__(self, "metadata", sanitize_metadata(self.metadata))

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
            parent_step_id=(
                str(data["parent_step_id"]) if data.get("parent_step_id") is not None else None
            ),
            span_kind=(str(data["span_kind"]) if data.get("span_kind") is not None else None),
            retry_scope=(str(data["retry_scope"]) if data.get("retry_scope") is not None else None),
            input_fingerprint=(
                str(data["input_fingerprint"]) if data.get("input_fingerprint") is not None else None
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=_as_int(data.get("cached_input_tokens")),
            reasoning_tokens=_as_int(data.get("reasoning_tokens")),
            provider_reported_cost_usd=(
                _as_float(data["provider_reported_cost_usd"])
                if data.get("provider_reported_cost_usd") is not None
                else None
            ),
            cost_usd=_as_float(data.get("cost_usd")),
            retry_count=_as_int(data.get("retry_count")),
            error=str(data["error"]) if data.get("error") is not None else None,
            # L8: derive the bounded error_class from the incoming error at this
            # chokepoint (import path). An explicit error_class in the data wins;
            # otherwise classify the error (message dropped). This is the same
            # chokepoint that redacts the raw message via sanitize_error below.
            error_class=(
                str(data["error_class"])
                if data.get("error_class") is not None
                else classify_error(data.get("error"))
            ),
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
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "retry_count": self.retry_count,
            "redaction_mode": self.redaction_mode,
        }
        if self.parent_step_id is not None:
            data["parent_step_id"] = self.parent_step_id
        if self.span_kind is not None:
            data["span_kind"] = self.span_kind
        if self.retry_scope is not None:
            data["retry_scope"] = self.retry_scope
        if self.input_fingerprint is not None:
            data["input_fingerprint"] = self.input_fingerprint
        if self.provider_reported_cost_usd is not None:
            data["provider_reported_cost_usd"] = round(self.provider_reported_cost_usd, 8)
        if self.error is not None:
            data["error"] = self.error
        if self.error_class is not None:
            data["error_class"] = self.error_class
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
    # L2: caller-supplied capture timestamp, recorded on the immutable base row.
    # Caller-supplied (not a buried wall-clock) so capture stays deterministic
    # for golden + idempotency tests; "" means "not stamped".
    ingested_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceBundle:
        schema_version = str(data.get("schema_version") or SCHEMA_VERSION)
        # L1 compatibility policy: accept same-major / same-or-lower-minor
        # (missing fields upcast to defaults via the per-field `.get`s below);
        # reject higher minor (unknown fact fields) and any different major.
        if not is_version_compatible(schema_version):
            raise TraceValidationError(
                f"incompatible schema_version={schema_version!r}; reader is "
                f"{SCHEMA_VERSION!r} (accepts same major, same-or-lower minor)"
            )
        if "run" not in data:
            raise TraceValidationError("trace bundle missing 'run'")
        # The run inherits the bundle-level shape stamp unless it carries its own
        # (a single .db can hold mixed-version rows once upgrade-on-read lands).
        run_data = dict(data["run"])
        run_data.setdefault("schema_version", schema_version)
        run = RunRecord.from_dict(run_data)
        steps = [StepRecord.from_dict(dict(step), run.id) for step in data.get("steps") or []]
        if not steps:
            raise TraceValidationError("trace bundle must contain at least one step")
        run = _with_computed_totals(run, steps)
        prescriptions = [
            PrescriptionRecord.from_dict(dict(item), run.id)
            for item in data.get("prescriptions") or []
        ]
        bundle = cls(
            run=run,
            steps=steps,
            prescriptions=prescriptions,
            schema_version=schema_version,
            ingested_at=str(data.get("ingested_at") or ""),
        )
        bundle.validate()
        return bundle

    def validate(self) -> None:
        if self.run.total_cost_usd < 0:
            raise TraceValidationError("run.total_cost_usd has negative total_cost_usd")
        if not math.isfinite(self.run.total_cost_usd):
            raise TraceValidationError("run.total_cost_usd has non-finite total_cost_usd")
        if not self.run.id:
            raise TraceValidationError("run.id is required")
        if not self.steps:
            raise TraceValidationError("trace bundle must contain at least one step")
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
            if step.cost_usd < 0:
                raise TraceValidationError(f"step {step.id!r} has negative cost_usd")
            if not math.isfinite(step.cost_usd):
                raise TraceValidationError(f"step {step.id!r} has non-finite cost_usd")
        for prescription in self.prescriptions:
            _validate_patch_artifact(prescription)

    def with_prescriptions(self, prescriptions: list[PrescriptionRecord]) -> TraceBundle:
        # replace() carries every other field (ingested_at + any future field)
        # through unchanged — same silent-drop guard as _with_computed_totals.
        return replace(self, prescriptions=prescriptions)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.ingested_at:
            data["ingested_at"] = self.ingested_at
        if self.prescriptions:
            data["prescriptions"] = [item.to_dict() for item in self.prescriptions]
        return data


def _with_computed_totals(run: RunRecord, steps: list[StepRecord]) -> RunRecord:
    total_cost = run.total_cost_usd or sum(step.cost_usd for step in steps)
    input_tokens = run.total_input_tokens or sum(step.input_tokens for step in steps)
    output_tokens = run.total_output_tokens or sum(step.output_tokens for step in steps)
    # Use dataclasses.replace so EVERY other field (schema_version and every
    # future fact column) is carried through unchanged. An explicit-kwarg rebuild
    # silently resets any field it forgets to name — the exact silent-drop trap
    # this task is hardening against.
    return replace(
        run,
        total_cost_usd=total_cost,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
    )


def sanitize_error(error: Any) -> str:
    return _SAFE_ERROR_MESSAGE


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    redacted = _sanitize_metadata_value(metadata)
    return redacted if isinstance(redacted, dict) else {}


def _sanitize_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_allowed_metadata_key(key_text):
                clean[key_text] = _sanitize_metadata_value(item)
            else:
                continue
        return clean
    if isinstance(value, list):
        return [_sanitize_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_metadata_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _is_allowed_metadata_key(key: str) -> bool:
    return key.lower() in _ALLOWED_METADATA_KEYS


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
