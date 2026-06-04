"""The RepairReceipt — a JUDGMENT computed on read from the immutable base FACTS.

A receipt is NEVER stored in the base tables. It is derived from a TraceBundle's
facts + its prescriptions at read time, so the facts/judgments boundary
(proof-ladder doc) stays intact: a price-table or grading change recomputes the
receipt without touching the corpus.

The receipt attaches an HONEST proof grade from the L0–L6 ladder. This slice
implements ONE durable class — retry-cap — and the cheapest strong tier, L2
(static verification): the repair MECHANICALLY removes a deterministic failure
path WITHOUT a re-run. A bounded retry budget cannot loop unboundedly; that is
provable by inspecting the (templated) artifact, no live re-run required.

Grade honesty rules:
  - An APPLYABLE templated retry-cap diff (file/line target present) -> L2.
  - The non-runnable config_diff fallback (no target) -> L1 (relevance only).
  - Never claim causality. The claim is graded directional, with limits shown.
  - Every outcome_delta carries a counter-metric guardrail; limits disclose the
    regression-to-the-mean caveat (ARL fires on the worst runs, which partly
    self-correct) and any model fact supplied by the app.

This module is provider-neutral and content-free: it reads only bounded facts and
the already-redacted prescription evidence. It introduces NO new egress channel
content beyond bounded labels/numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent_run_ledger.core.models import TraceBundle

# Closed proof ladder (the SHAPE is locked; this slice grades only L0–L2).
PROOF_LEVELS: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4", "L5", "L6")

# A receipt's failure label is a bounded vocabulary (proof-ladder doc).
OBSERVED_FAILURES: tuple[str, ...] = (
    "retry_loop",
    "schema_mismatch",
    "context_bloat",
    "model_misroute",
    "missing_contract",
)


@dataclass(frozen=True)
class RepairReceipt:
    """The product's unit of output (proof-ladder doc shape)."""

    run_id: str
    claim: str
    observed_failure: str
    evidence: list[str]
    repair_artifact: dict[str, Any]
    proof_level: str
    confidence: str
    limits: list[str]
    next_evidence: list[str]
    outcome_delta: dict[str, Any] = field(default_factory=dict)


def build_receipts(bundle: TraceBundle) -> list[RepairReceipt]:
    """Compute the receipts for *bundle* from its facts + prescriptions.

    Returns [] when there is no prescription (no detected failure) — the negative
    gate: no invented receipts on a clean run."""
    receipts: list[RepairReceipt] = []
    for rx in bundle.prescriptions:
        # This slice grades the retry-cap class only. A prescription whose
        # one_line_fix sets a retry budget is a retry_loop repair.
        proof_level = _grade_retry_cap(rx.patch_type, rx.patch)
        model_supplied = _model_priced_run(bundle)
        receipts.append(
            RepairReceipt(
                run_id=bundle.run.id,
                claim=_claim(proof_level),
                observed_failure="retry_loop",
                evidence=list(rx.evidence),
                repair_artifact={
                    "patch_type": rx.patch_type,
                    # Templated, NOT free-form LLM output: the retry-cap artifact is
                    # generated from a constrained template (difflib over a target
                    # line, or a fixed config block), so it is auditable + apply-safe.
                    "templated": True,
                    "one_line_fix": rx.one_line_fix,
                    "patch": rx.patch,
                },
                proof_level=proof_level,
                confidence=_confidence(proof_level),
                limits=_limits(proof_level, model_supplied),
                next_evidence=_next_evidence(proof_level),
                outcome_delta=_outcome_delta(rx.expected_impact),
            )
        )
    return receipts


def _grade_retry_cap(patch_type: str, patch: str) -> str:
    """Grade the proof level for a retry-cap repair by STATIC inspection.

    L2 iff the artifact is an APPLYABLE templated retry-cap diff: a unified diff
    whose target line lowers a retry budget to a finite cap. That mechanically
    removes the unbounded-retry path — provable without a re-run. The non-runnable
    config_diff fallback (no file/line target) is relevant but not mechanical ->
    L1. Anything else -> L0 (diagnostic)."""
    if patch_type == "unified_diff" and _is_retry_cap_diff(patch):
        return "L2"
    if patch_type == "config_diff":
        return "L1"
    return "L0"


def _is_retry_cap_diff(patch: str) -> bool:
    """True iff *patch* is a unified diff that VERIFIABLY bounds a retry budget —
    a real numeric DECREASE on a retry-budget line, not a substring match.

    Hardened (fleet HIGH): the old ``"retr" in patch`` check matched the file PATH
    and graded arbitrary or budget-RAISING diffs as L2. L2 now requires a changed
    line whose identifier names a retry budget AND whose integer value strictly
    DECREASES (removed value > added value). That is what "statically removes the
    unbounded-retry path" actually means."""
    lines = patch.splitlines()
    has_diff_markers = (
        any(line.startswith("--- ") for line in lines)
        and any(line.startswith("+++ ") for line in lines)
        and any(line.startswith("@@") for line in lines)
    )
    if not has_diff_markers:
        return False
    # Consider only CONTENT lines (exclude file headers ---/+++).
    removed = [ln[1:] for ln in lines if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:] for ln in lines if ln.startswith("+") and not ln.startswith("+++")]
    old_budget = _retry_budget_value(removed)
    new_budget = _retry_budget_value(added)
    if old_budget is None or new_budget is None:
        return False
    # Strict decrease: the cap is lower than the prior budget.
    return new_budget < old_budget


# A retry-budget assignment line: an identifier mentioning retr/retries/attempts/
# backoff/max_tries set to an integer (e.g. CRM_LOOKUP_MAX_RETRIES = 5,
# retry_budget: 3). The identifier match is what excludes unrelated diffs whose
# PATH merely contains "retr" (e.g. retrieve.py).
_RETRY_BUDGET_LINE = re.compile(
    r"(retr(y|ies)|max[_ ]?tries|attempts|backoff)\w*\s*[:=]\s*(\d+)",
    re.IGNORECASE,
)


def _retry_budget_value(content_lines: list[str]) -> int | None:
    """Return the integer retry budget asserted by exactly one of *content_lines*,
    or None if zero or more-than-one such line is present (ambiguous -> reject)."""
    values = [
        int(m.group(3))
        for line in content_lines
        if (m := _RETRY_BUDGET_LINE.search(line)) is not None
    ]
    return values[0] if len(values) == 1 else None


def _model_priced_run(bundle: TraceBundle) -> bool:
    """True when the run carries a known model. Some adapters cannot recover the
    model from the trace and rely on an app-supplied hint, and a receipt consumer
    cannot tell which — so whenever a model is present, the receipt discloses that
    any cost figure rests on the model fact. Provider-neutral: keyed on the model
    fact, not any framework string."""
    return bundle.run.model != "unknown"


def _claim(proof_level: str) -> str:
    if proof_level == "L2":
        return (
            "This repair statically removes the unbounded-retry failure path "
            "(graded directional evidence; not a causal guarantee)."
        )
    if proof_level == "L1":
        return (
            "This repair is relevant to the observed retry loop and is applyable "
            "(relevance only; mechanical removal not established)."
        )
    return "ARL found a likely retry loop; no accepted fix (diagnostic)."


def _confidence(proof_level: str) -> str:
    return {"L2": "medium", "L1": "low", "L0": "low"}.get(proof_level, "low")


def _limits(proof_level: str, model_supplied: bool) -> list[str]:
    limits = [
        # Constraint 5: regression-to-the-mean disclosure.
        "Before/after deltas are uncorrected for regression to the mean — ARL "
        "fires on the worst runs, which partly improve on their own.",
        # fleet HIGH: retry cost accrues on the repeated MODEL/response turns, not
        # the collapsed tool span — so the per-loop wasted-cost estimate is often
        # not attributable from the tool span alone. The L2 grade is STRUCTURAL
        # (cost-independent); the cost figure is supporting, not the proof.
        "Cost saving is not attributable from the tool span alone — retry waste "
        "accrues on the repeated model/response turns; the L2 grade does not "
        "depend on the cost figure.",
        # honest live-trace classification limit (verified against SDK source).
        "Live tool/response errors classify as 'Other': the SDK span error is "
        "free text, and bounded error-class precision needs app instrumentation.",
        "Retry detection covers tool/function calls only; response-call retries "
        "are not collapsed (no name/input to distinguish genuine vs legitimate).",
    ]
    if model_supplied:
        limits.append(
            "The cost figure depends on the run's model identity, which some "
            "adapters obtain from an app-supplied hint when the trace omits it."
        )
    if proof_level != "L2":
        limits.append(
            "Proof level below L2: the artifact lacks a file/line target, so "
            "mechanical removal of the failure path is not statically established."
        )
    return limits


def _next_evidence(proof_level: str) -> list[str]:
    if proof_level == "L2":
        # Apply-blind guard (fleet HIGH): never tell the user to apply blindly. The
        # diff is shown for REVIEW; the shipped regression test verifies it before
        # merge. ARL advises, the user applies.
        return [
            "review the templated retry-cap diff, then apply it and run the shipped "
            "regression test before merging",
            "observe the next N similar runs for recurrence (L4 evidence)",
        ]
    return [
        "instrument the trace step with a retry_budget_patch_target (path + before "
        "line) so ARL can generate a reviewable applyable diff (L2)",
    ]


def _outcome_delta(expected_impact: dict[str, Any]) -> dict[str, Any]:
    """Carry the prescription's expected impact + a counter-metric guardrail
    (Constraint 5), so a one-sided cost win is never shown without its guardrail.

    Honesty (fleet HIGH): retry cost in an agentic loop accrues on the repeated
    MODEL/response turns, not the tool spans the detector collapses — so the
    tool-derived wasted-cost is ~0. We must NOT present a confident precise
    ``-0.0`` (it reads as 'this fix saves nothing'). When the estimate rounds to
    ~0, replace the number with the honest label 'not attributable' and disclose
    the attribution gap in the receipt's limits."""
    delta = dict(expected_impact)
    cost = delta.get("estimated_cost_delta_usd")
    if isinstance(cost, (int, float)) and round(cost, 6) == 0.0:
        delta["estimated_cost_delta_usd"] = "not attributable"
    delta.setdefault(
        "guardrail_success_rate",
        "must not decrease — verify the shipped regression test before applying; "
        "a capped retry fails closed (typed failure) rather than looping.",
    )
    return delta
