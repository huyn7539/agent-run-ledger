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

This module is provider-neutral. The DETECTION and GRADING facts it computes are
content-free: the claim, proof level, confidence, limits, next_evidence, and the
evidence list read only bounded labels/numbers from the already-redacted
prescription. Those fields introduce NO new egress channel content.

ONE field is NOT content-free: ``repair_artifact.patch`` (and the app-supplied
allowed-metadata ``before`` / ``path`` / ``after`` values it is built from) MAY
carry local file paths and source-line content — the unified_diff embeds the real
``before`` line and the repo-relative ``path``. So a receipt is safe to surface
locally, but ``repair_artifact.patch`` is NOT remote-egress-safe until the Task 46
allowed-metadata value scrub redacts those before/path/after values. Treat the
patch as local-only output, not a bounded label, when forwarding off-box.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent_run_ledger.core.models import PrescriptionRecord, StepRecord, TraceBundle
from agent_run_ledger.core.prescriptions import derive_retry_steps

# Closed proof ladder (the SHAPE is locked; this slice grades only L0–L2).
PROOF_LEVELS: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4", "L5", "L6")

# Codex P1 (2026-06-11, spoof-hardened same day): artifact_failure FACTS
# (deletes_test_path, completion_claim_follows, user_directed_deletion) are only
# trustworthy when an ADAPTER computed them in-process from real session
# content. A neutral JSON import lets a hostile author hand-supply those
# booleans, so a forged file could claim an artifact_failure at L1 against a
# stranger it's accusing. Artifact grading is therefore gated on
# ``TraceBundle.adapter_provenanced`` — an in-process bit set ONLY by the
# claude-code / codex-cli capture adapters and never read from an imported
# file. The first version of this gate keyed off ``run.framework``, which the
# imported file controls: a forger who declared ``"framework": "claude-code"``
# walked straight through it. The framework string is metadata; the trust bit
# is process provenance. This mirrors the retry path's "grade from derived
# facts, never the evidence string" discipline (Task 51), applied to the import
# boundary.

# A receipt's failure label is a bounded vocabulary (proof-ladder doc).
OBSERVED_FAILURES: tuple[str, ...] = (
    "retry_loop",
    "artifact_failure",
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
    # The DERIVED (collapsed) steps carry the AUTHORITATIVE retry_count per step id —
    # a stored/imported prescription's free-form evidence is attacker-controllable, so
    # grading must come from the real facts, not the evidence string (Task 51 forged).
    derived_by_id = {s.id: s for s in derive_retry_steps(bundle)}
    raw_by_id = {s.id: s for s in bundle.steps}
    receipts: list[RepairReceipt] = []
    for rx in bundle.prescriptions:
        # Task 58: branch by the prescription's bounded failure class. The
        # artifact_failure class grades L0-L1 only (success-claim/log divergence:
        # relevance can be proven from the log; intent and correctness cannot).
        if rx.failure_class == "artifact_failure":
            receipts.append(_artifact_failure_receipt(bundle, rx, raw_by_id))
            continue
        # This slice grades the retry-cap class only. A prescription whose
        # one_line_fix sets a retry budget is a retry_loop repair. The observed retry
        # count must be the CITED STEP's REAL retry_count (looked up in the derived
        # facts), NOT the number the evidence string claims — else forged evidence
        # (retry_count=99 over a real 2) upgrades an insufficient cap to L2 (Task 51).
        observed = _authoritative_observed_count(rx, derived_by_id)
        proof_level = _grade_retry_cap(rx.patch_type, rx.patch, observed)
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


# --------------------------------------------------------------------------- #
# Task 58 — artifact_failure receipts (success-claim vs log-evidence divergence)
# --------------------------------------------------------------------------- #
# Grading is FACT-CORROBORATED, never prescription-trusted (Task 51 lessons): a
# stored/imported prescription claiming R1 earns L1 ONLY when the cited step's
# bounded capture-time facts in the bundle corroborate it; anything else fails
# closed to L0. R2 is always L0 — absence-of-evidence (zero mutating calls) is
# structurally weaker than presence-of-deletion, so it never grades above
# diagnostic.

_ARTIFACT_RULE_RE = re.compile(r"rule=(R1|R2)")


def _artifact_rule(evidence: list[str]) -> str | None:
    """The sub-rule label from the EXACT ARL-authored evidence line (fullmatch on
    the stripped line — same anti-forgery discipline as the retry count). None if
    absent or ambiguous (two distinct labels) -> grading fails closed."""
    labels = {
        m.group(1)
        for line in evidence
        if (m := _ARTIFACT_RULE_RE.fullmatch(line.strip())) is not None
    }
    return next(iter(labels)) if len(labels) == 1 else None


def _grade_artifact_failure(
    rx: PrescriptionRecord, raw_by_id: dict[str, StepRecord], adapter_provenanced: bool
) -> str:
    """L1 only for a fact-corroborated R1 from an adapter-provenanced run; else L0.

    Codex P1: the corroborating booleans are only facts when an adapter computed
    them. On a neutral/forged import (``adapter_provenanced`` False) the strongest
    honest grade is L0 — ARL will not L1-accuse on attacker-supplied booleans."""
    if not adapter_provenanced:
        return "L0"
    rule = _artifact_rule(rx.evidence)
    if rule != "R1":
        return "L0"
    step_ids = {
        m.group(1)
        for line in rx.evidence
        if (m := _STEP_ID_RE.fullmatch(line.strip())) is not None
    }
    if len(step_ids) != 1:
        return "L0"
    step = raw_by_id.get(next(iter(step_ids)))
    if step is None:
        return "L0"
    md = step.metadata
    if (
        md.get("deletes_test_path") is True
        and md.get("completion_claim_follows") is True
        and md.get("user_directed_deletion") is not True
    ):
        return "L1"
    return "L0"


def _artifact_failure_receipt(
    bundle: TraceBundle, rx: PrescriptionRecord, raw_by_id: dict[str, StepRecord]
) -> RepairReceipt:
    rule = _artifact_rule(rx.evidence)
    proof_level = _grade_artifact_failure(rx, raw_by_id, bundle.adapter_provenanced)
    return RepairReceipt(
        run_id=bundle.run.id,
        claim=_artifact_claim(rule, proof_level),
        observed_failure="artifact_failure",
        evidence=list(rx.evidence),
        repair_artifact={
            "patch_type": rx.patch_type,
            # NOT a templated applyable patch: the artifact is a text fix
            # direction (ARL stores no file content, so it cannot name the path).
            "templated": False,
            "one_line_fix": rx.one_line_fix,
            "patch": rx.patch,
        },
        proof_level=proof_level,
        confidence="low",
        limits=_artifact_limits(rule),
        next_evidence=_artifact_next_evidence(rule),
        outcome_delta=_artifact_outcome_delta(rx.expected_impact),
    )


def _artifact_claim(rule: str | None, proof_level: str) -> str:
    if rule == "R1" and proof_level == "L1":
        return (
            "The run's success claim diverges from its own session log: a "
            "test-path deletion precedes the completion claim with no intervening "
            "user instruction (relevance proven from the log; intent NOT proven)."
        )
    if rule == "R2":
        return (
            "The assistant claimed completion after a change request, but the "
            "session log records zero mutating tool calls (diagnostic; the claim "
            "is unverified, not proven false)."
        )
    return (
        "ARL flagged a possible success-claim/log divergence, but the cited "
        "step's capture-time facts could not be corroborated (diagnostic only)."
    )


def _artifact_limits(rule: str | None) -> list[str]:
    limits = [
        # The MANDATORY caveat: intent is not provable from the log alone.
        "The deletion or omission may have been user-directed in a way the "
        "detector could not see (an instruction outside this session, or "
        "phrasing outside the bounded directive list); verify intent before "
        "acting.",
        "Completion claims are matched against a bounded marker list computed at "
        "capture; phrasing outside the list is not detected, and a matched "
        "marker proves wording, not intent.",
        "ARL did not run the tests or inspect the working tree; whether the "
        "delivered work is actually correct is not established at this level.",
    ]
    if rule == "R2":
        limits.append(
            "Zero mutating tool calls is evidence of absence in THIS session log "
            "only — work may have landed outside it (another session, manual "
            "edits, MCP side effects the log does not record)."
        )
        limits.append(
            "The change request was inferred from a bounded verb list; the user "
            "may have wanted analysis only."
        )
    return limits


def _artifact_next_evidence(rule: str | None) -> list[str]:
    if rule == "R1":
        return [
            "restore the deleted test path(s) from version control and re-run the "
            "full suite",
            "confirm with the operator whether the deletion was requested",
        ]
    return [
        "diff the working tree / VCS log to verify whether any change actually "
        "landed",
        "re-run the task with the completion claim treated as unverified",
    ]


def _artifact_outcome_delta(expected_impact: dict[str, Any]) -> dict[str, Any]:
    delta = dict(expected_impact)
    delta.setdefault(
        "guardrail_trust",
        "treat in-session success claims as unverified until re-verified against "
        "real command output; ARL advises, you verify.",
    )
    return delta


def _grade_retry_cap(patch_type: str, patch: str, observed_retry_count: int | None) -> str:
    """Grade the proof level for a retry-cap repair by STATIC inspection.

    L2 requires BOTH:
      1. SYNTAX (``_is_retry_cap_diff``): an applyable unified diff whose target
         line lowers a retry budget by a real numeric DECREASE (not a substring/
         path match, not a raise).
      2. SUFFICIENCY (B2): the new cap must actually remove the OBSERVED failure
         path. ``observed_retry_count`` is the number of ADDITIONAL attempts seen
         (a count of 2 is a 3-attempt loop). A new cap that still permits that loop
         (``new_budget >= observed_retry_count``) does NOT mechanically remove the
         path, so it grades L1 — the L2 claim ("statically removes the unbounded-
         retry failure path") would be an overclaim. A new cap STRICTLY below the
         observed count drops the loop below what was seen and earns L2.

    FAIL CLOSED: if the new cap or the observed count cannot be recovered,
    sufficiency is UNVERIFIABLE -> never grant L2 (degrade to L1). The non-runnable
    config_diff fallback (no file/line target) -> L1. Anything else -> L0."""
    if patch_type == "unified_diff" and _is_retry_cap_diff(patch):
        new_budget = _new_cap_value(patch)
        # Fail closed: unrecoverable new cap or observed count -> not L2. But L1
        # (relevance) still requires a CORROBORATED observed loop — a unified diff
        # with no recoverable observed count is L0, not a free L1 (Codex P1).
        if observed_retry_count is None:
            return "L0"
        if new_budget is None:
            return "L1"
        # The new cap must drop the loop BELOW the observed additional-attempt count.
        if new_budget < observed_retry_count:
            return "L2"
        return "L1"
    if patch_type == "config_diff":
        # Codex P1: config_diff is the non-runnable fallback (no file/line target),
        # so it can never reach L2 — but L1 (relevance to the observed loop) is only
        # honest when the observed retry count is CORROBORATED from the real cited
        # step. A forged/uncorroborated prescription (observed_retry_count is None)
        # grades L0 (diagnostic), never a free L1. Was: unconditional L1 — that let a
        # fabricated config_diff prescription claim relevance it hadn't earned.
        return "L1" if observed_retry_count is not None else "L0"
    return "L0"


# The observed additional-attempt count, as the prescription records it in evidence:
# the EXACT ARL-authored line "retry_count=<N> additional attempts" (prescriptions.py).
# We FULLMATCH the whole (stripped) line — not a substring/search — so the phrase
# embedded in a larger free-text note (e.g. "stale note: retry_count=99 additional
# attempts") cannot supply a forged observed count (Codex re-review P1; fleet B2-L1).
# If it is absent OR two DISTINCT counts appear, grading fails closed (no L2).
_OBSERVED_RETRY_RE = re.compile(r"retry_count=(\d+) additional attempts")


def _observed_retry_count(evidence: list[str]) -> int | None:
    """Recover the observed additional-attempt count from a prescription's evidence,
    or None if it is absent OR ambiguous (sufficiency then unverifiable -> fail
    closed). Only the EXACT ARL-authored line counts (fullmatch on the stripped
    line), so a stale/foreign phrase inside a larger note is ignored; if two DISTINCT
    counts appear (a poisoned/inconsistent prescription), we refuse rather than pick
    one."""
    values = {
        int(m.group(1))
        for line in evidence
        if (m := _OBSERVED_RETRY_RE.fullmatch(line.strip())) is not None
    }
    return next(iter(values)) if len(values) == 1 else None


_STEP_ID_RE = re.compile(r"step_id=(\S+)")

# Documentation / prose targets: a retry-budget-looking assignment changed in one of
# these is not a reachable code path, so it cannot statically remove the loop (Task 51).
_NON_CODE_EXTENSIONS = (".md", ".rst", ".txt", ".markdown", ".adoc")


def _authoritative_observed_count(
    rx: PrescriptionRecord, derived_by_id: dict[str, StepRecord]
) -> int | None:
    """The observed additional-attempt count to grade against — recovered from the
    CITED STEP's REAL ``retry_count`` in the derived facts, NOT the evidence string.

    Fail closed (return None -> L1) when sufficiency is unverifiable or evidence looks
    FORGED (Task 51):
      * no ``step_id=`` in evidence, or the cited step is not in the derived facts;
      * the evidence's CLAIMED count (if present) DISAGREES with the cited step's real
        count — a stored/imported prescription claiming ``retry_count=99`` over a real
        2-attempt step is poisoned; we refuse rather than trust the larger number."""
    step_ids = {
        m.group(1) for line in rx.evidence if (m := _STEP_ID_RE.fullmatch(line.strip())) is not None
    }
    if len(step_ids) != 1:
        return None
    step = derived_by_id.get(next(iter(step_ids)))
    if step is None:
        return None
    real = step.retry_count
    claimed = _observed_retry_count(rx.evidence)
    # If evidence claims a count, it must MATCH the real cited-step count; a mismatch
    # (especially a larger forged claim) fails closed. Absent claim -> trust the fact.
    if claimed is not None and claimed != real:
        return None
    return real


def _new_cap_value(patch: str) -> int | None:
    """Return the integer retry budget the diff's ADDED line sets, reusing the same
    budget-line recognizer the syntax gate uses. None if not recoverable."""
    added = [
        ln[1:]
        for ln in patch.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]
    return _retry_budget_value(added)


def _is_retry_cap_diff(patch: str) -> bool:
    """True iff *patch* is a unified diff that VERIFIABLY bounds a retry budget —
    a real numeric DECREASE on a retry-budget line, not a substring match.

    Hardened (fleet HIGH): the old ``"retr" in patch`` check matched the file PATH
    and graded arbitrary or budget-RAISING diffs as L2. L2 now requires a changed
    line whose identifier names a retry budget AND whose integer value strictly
    DECREASES (removed value > added value). That is what "statically removes the
    unbounded-retry path" actually means."""
    lines = patch.splitlines()
    minus_headers = [ln for ln in lines if ln.startswith("--- ")]
    plus_headers = [ln for ln in lines if ln.startswith("+++ ")]
    hunk_headers = [ln for ln in lines if ln.startswith("@@")]
    # EXACTLY one of each header (Task 51 / Codex fleet Finding 1+2): a genuine
    # retry-cap patch is a SINGLE file, SINGLE hunk. Counting (not a distinct-path
    # SET) is load-bearing — a DUPLICATED same-file header (`--- a/crm.py` twice)
    # collapses to one set element and bypassed the old guard, AND a multi-file diff
    # (remove from service_a, add to service_b) is not a live cap decrease. Requiring
    # exactly one `---`, one `+++`, one `@@` rejects both without breaking a real diff.
    if not (len(minus_headers) == 1 and len(plus_headers) == 1 and len(hunk_headers) == 1):
        return False
    # Same single file on both sides.
    def _path(hdr: str) -> str:
        return hdr[4:].strip().split("\t")[0].removeprefix("b/").removeprefix("a/")

    if _path(minus_headers[0]) != _path(plus_headers[0]):
        return False
    # The target must be a CODE path, not documentation/prose (Task 51 wrong-file):
    # a real ``MAX_RETRIES = 0`` assignment living in docs.md changes no reachable
    # code path, so it cannot "statically remove" the loop. (NOTE: closes the doc
    # case; full path-binding to the prescription's cited target remains a follow-up.)
    target = _path(plus_headers[0])
    if any(target.lower().endswith(ext) for ext in _NON_CODE_EXTENSIONS):
        return False
    # Consider only CONTENT lines (exclude file headers ---/+++).
    removed = [ln[1:] for ln in lines if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:] for ln in lines if ln.startswith("+") and not ln.startswith("+++")]
    old = _retry_budget_assignment(removed)
    new = _retry_budget_assignment(added)
    if old is None or new is None:
        return False
    old_ident, old_budget = old
    new_ident, new_budget = new
    # The SAME identifier must be lowered: a diff that removes CRM_MAX_RETRIES and adds
    # PAYMENTS_MAX_RETRIES is not "lowering the observed loop's cap" (Task 51).
    if old_ident != new_ident:
        return False
    # Strict decrease: the cap is lower than the prior budget.
    if new_budget >= old_budget:
        return False
    # NO OTHER executable change: the ONLY changed content line on each side is the
    # one budget assignment. An extra added/removed line (e.g. an injected
    # ``import os; os.system(...)`` alongside the cap decrease) means the patch does
    # more than bound the retry budget, so it is NOT a clean apply-safe retry-cap diff
    # and must not earn L2 (Task 51 — extra-payload). Context/blank lines are fine.
    extra = [ln for ln in (removed + added) if ln.strip() and _RETRY_BUDGET_LINE.match(ln.strip()) is None]
    if extra:
        return False
    return True


# A retry-budget assignment line — ANCHORED at the start of the (stripped) line to a
# real ``IDENTIFIER = INTEGER`` / ``IDENTIFIER: INTEGER`` assignment where the
# identifier names a retry budget. Anchoring (``^``) is the load-bearing change
# (Task 51): the old ``search()`` matched the pattern ANYWHERE in the line, so a
# string literal (``print("MAX_RETRIES = 10")``), a docstring, or any embedded text
# graded L2. A live assignment starts the line; a string/call/comment does not.
_RETRY_BUDGET_LINE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_.]*(?:retr(?:y|ies)|max[_ ]?tries|attempts|backoff)[A-Za-z0-9_]*)"
    # [0-9]+ NOT \d+ : \d matches Unicode fullwidth digits (e.g. '３'), which int()
    # accepts but Python does NOT compile as a live integer literal — that let a
    # non-executable assignment grade L2 (Task 51, Codex). ASCII decimal only.
    r"\s*[:=]\s*([0-9]+)\s*$",
    re.IGNORECASE,
)

# Comment markers (line AND block): a budget assignment behind any of these is not a
# live code path, so changing it removes nothing (Codex re-review P2 + Task 51 block
# comments). Reject a line whose first non-blank chars begin a comment OR a string.
_COMMENT_OR_STRING_PREFIX = re.compile(r'^\s*(#|//|--|;|/\*|\*|"""|\'\'\'|"|\')')


def _retry_budget_assignment(content_lines: list[str]) -> tuple[str, int] | None:
    """Return ``(identifier, value)`` for exactly ONE live retry-budget assignment in
    *content_lines*, or None if zero / more-than-one (ambiguous -> reject). A line
    behind a comment or string marker is NOT a live assignment and is skipped. The
    regex is ANCHORED, so only a real start-of-line assignment matches — a string
    literal / call / embedded text does not."""
    found: list[tuple[str, int]] = []
    for line in content_lines:
        stripped = line.strip()
        if _COMMENT_OR_STRING_PREFIX.match(stripped) is not None:
            continue
        m = _RETRY_BUDGET_LINE.match(stripped)
        if m is not None:
            found.append((m.group(1), int(m.group(2))))
    return found[0] if len(found) == 1 else None


def _retry_budget_value(content_lines: list[str]) -> int | None:
    """Back-compat shim: the integer of the single live retry-budget assignment."""
    a = _retry_budget_assignment(content_lines)
    return a[1] if a is not None else None


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
            "This repair direction is relevant to the observed retry loop, but "
            "mechanical removal is not established (relevance only; the "
            "config_diff fallback carries no file/line target to apply)."
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
