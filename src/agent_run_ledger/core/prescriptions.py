from __future__ import annotations

import difflib
import re
from dataclasses import replace
from typing import Any
from uuid import uuid4

from agent_run_ledger.core.models import PrescriptionRecord, StepRecord, TraceBundle
from agent_run_ledger.core.retries import AttemptFacts, collapse_retry_groups


def analyze_bundle(bundle: TraceBundle) -> list[PrescriptionRecord]:
    # The detector reads the COLLAPSED view computed on read — the immutable base
    # keeps one raw StepRecord per span; the retry collapse is a JUDGMENT, never
    # baked into the corpus (so a future detector fix can re-derive from facts).
    collapsed = replace(bundle, steps=derive_retry_steps(bundle))
    prescriptions = detect_retry_cost_loops(collapsed)
    # Task 58: success-claim vs log-evidence divergence reads the RAW steps' bounded
    # capture-time facts (no collapse involved). Same judgment-on-read doctrine.
    prescriptions.extend(detect_success_lies(bundle))
    return prescriptions


def derive_retry_steps(bundle: TraceBundle) -> list[StepRecord]:
    """Collapse raw per-span steps into a retry-aware view, ON READ.

    A genuine retry loop is N repeated same-scope, same-input tool attempts with
    >=1 failure, ONE attempt per distinct turn (each attempt under its own turn
    parent while retry_scope stays stable). Such a run collapses to ONE StepRecord
    with retry_count=N-1, tokens + cost SUMMED, error_class from the last attempt.
    Everything else (model/response spans, legitimate repetition, app-supplied
    explicit retry_count) passes through unchanged. The base bundle.steps is NOT
    mutated."""
    steps = bundle.steps
    # Deterministic order mirrors provenance.py: (started_at, id).
    indexed = sorted(range(len(steps)), key=lambda i: (steps[i].started_at, steps[i].id))
    attempts = [
        AttemptFacts(
            index=pos,
            name=steps[i].name,
            span_kind=steps[i].span_kind,
            retry_scope=steps[i].retry_scope,
            # B3: the immediate (turn) parent. A real cross-turn retry gives each
            # attempt its OWN distinct turn parent. The grouper requires ONE attempt
            # per DISTINCT turn (_is_one_attempt_per_distinct_turn): it rejects not
            # only pure same-turn fan-out ([t1,t1,t1]) but any group that smuggles a
            # same-turn duplicate alongside cross-turn attempts ([t1,t2,t2]) — a
            # bare >1-turn count is not enough.
            turn_id=steps[i].parent_step_id,
            started_at=steps[i].started_at,
            ended_at=steps[i].ended_at,
            has_error=steps[i].error is not None,
            error_class=steps[i].error_class,
            # An app-supplied explicit retry_count step is NOT eligible for
            # derivation (its count is authoritative); withhold its fingerprint so
            # the grouper leaves it a singleton.
            input_fingerprint=(steps[i].input_fingerprint if steps[i].retry_count == 0 else None),
        )
        for pos, i in enumerate(indexed)
    ]
    groups = collapse_retry_groups(attempts)
    # map sorted-position groups back to original step indices, preserving order
    result: list[StepRecord] = []
    for group in groups:
        original = [indexed[pos] for pos in group]
        result.append(_collapse_steps([steps[i] for i in original]))
    return result


def _collapse_steps(group: list[StepRecord]) -> StepRecord:
    """Collapse 1+ raw attempt steps into one. A singleton returns unchanged."""
    if len(group) == 1:
        return group[0]
    first, last = group[0], group[-1]
    reported = [s.provider_reported_cost_usd for s in group if s.provider_reported_cost_usd is not None]
    return replace(
        first,
        ended_at=last.ended_at,
        retry_count=len(group) - 1,
        # Task 53 F3: this count was DERIVED by ARL from raw attempts — the only
        # path that may mark it so; explicit pass-throughs keep the default.
        retry_count_source="derived",
        input_tokens=sum(s.input_tokens for s in group),
        output_tokens=sum(s.output_tokens for s in group),
        cached_input_tokens=sum(s.cached_input_tokens for s in group),
        reasoning_tokens=sum(s.reasoning_tokens for s in group),
        cost_usd=sum(s.cost_usd for s in group),
        provider_reported_cost_usd=(sum(reported) if reported else None),
        # last attempt's terminal error class drives severity.
        error=last.error,
        error_class=last.error_class,
    )


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


# --------------------------------------------------------------------------- #
# Task 58 — success-claim vs log-evidence divergence (the "success-lie" detector)
# --------------------------------------------------------------------------- #
# Both rules read ONLY bounded boolean FACTS the adapters computed at capture
# time and stored as allowed step metadata (deletes_test_path /
# user_directed_deletion / completion_claim_follows / mutating /
# change_request). Core never sees raw content. ABSTAIN-BY-DEFAULT: a bundle
# whose adapter did not supply a fact can never fire — absence is unknown, not
# evidence. Precision-first: a false positive here is the product's worst bug.


def detect_success_lies(bundle: TraceBundle) -> list[PrescriptionRecord]:
    """R1 (success claim after test deletion) + R2 (completion claim with zero
    mutating tool calls). One prescription per rule per bundle, max."""
    ordered = sorted(bundle.steps, key=lambda s: (s.started_at, s.id))
    prescriptions: list[PrescriptionRecord] = []
    r1 = _detect_test_deletion_lie(bundle, ordered)
    if r1 is not None:
        prescriptions.append(r1)
    r2 = _detect_no_op_completion(bundle, ordered)
    if r2 is not None:
        prescriptions.append(r2)
    return prescriptions


def _detect_test_deletion_lie(
    bundle: TraceBundle, ordered: list[StepRecord]
) -> PrescriptionRecord | None:
    """R1: a step deleted a test-pattern path, a terminal completion claim
    followed with no intervening human instruction, and the deletion was NOT
    user-directed (per the adapter's generous capture-time check)."""
    hits = [
        s
        for s in ordered
        if s.metadata.get("deletes_test_path") is True
        and s.metadata.get("completion_claim_follows") is True
        and s.metadata.get("user_directed_deletion") is not True
    ]
    if not hits:
        return None
    first = hits[0]
    return PrescriptionRecord(
        id=f"rx_{uuid4().hex[:12]}",
        run_id=bundle.run.id,
        failure_class="artifact_failure",
        severity="high",
        root_cause=(
            "a tool call deleted a test-pattern path and a later assistant completion "
            "claim followed with no intervening user instruction "
            f"({len(hits)} deletion step(s))"
        ),
        one_line_fix=(
            "Restore the deleted test path(s) from version control and re-run the "
            "full suite before trusting the success claim."
        ),
        evidence=[
            "rule=R1",
            f"step_id={first.id}",
            "deletes_test_path=true",
            "completion_claim_follows=true",
            f"test_path_deletions={len(hits)}",
        ],
        patch_type="config_diff",
        patch=_R1_FIX_DIRECTION,
        expected_impact={
            "trust_delta": (
                "the run's success claim is unverified until the deleted test "
                "path(s) are restored and the suite re-runs"
            ),
        },
        regression_test_template=(
            "def test_no_test_paths_deleted_before_success_claim():\n"
            "    receipt = arl_verdict(session_log)\n"
            "    assert receipt.observed_failure != 'artifact_failure'\n"
        ),
    )


def _detect_no_op_completion(
    bundle: TraceBundle, ordered: list[StepRecord]
) -> PrescriptionRecord | None:
    """R2: a change was requested, the assistant ended with a completion claim,
    and the session's tool census shows ZERO mutating calls.

    Fires ONLY when the adapter supplied an EXPLICIT boolean ``mutating`` fact on
    EVERY step (the census). A missing fact means unknown -> abstain. A pure Q&A
    session (no change-request instruction) can never fire."""
    if not ordered:
        return None
    census = [s.metadata.get("mutating") for s in ordered]
    if any(not isinstance(value, bool) for value in census):
        return None  # facts unavailable (other adapters / old captures) -> abstain
    if any(census):
        return None  # a (potentially) mutating call exists -> abstain
    if not any(s.metadata.get("change_request") is True for s in ordered):
        return None  # no change request (Q&A session) -> NEVER fire
    last = ordered[-1]
    if last.metadata.get("completion_claim_follows") is not True:
        return None  # the session did not end on a completion claim -> abstain
    return PrescriptionRecord(
        id=f"rx_{uuid4().hex[:12]}",
        run_id=bundle.run.id,
        failure_class="artifact_failure",
        severity="medium",
        root_cause=(
            "the assistant claimed completion after a change request, but the session "
            "log records zero mutating tool calls"
        ),
        one_line_fix=(
            "Verify whether any change actually landed (diff the working tree / VCS "
            "log); treat the completion claim as unverified."
        ),
        evidence=[
            "rule=R2",
            f"step_id={last.id}",
            "mutating_calls=0",
            f"tool_calls={len(ordered)}",
            "completion_claim_follows=true",
            "change_request=true",
        ],
        patch_type="config_diff",
        patch=_R2_FIX_DIRECTION,
        expected_impact={
            "trust_delta": (
                "the completion claim is unverified until a real change is confirmed "
                "outside the agent's self-report"
            ),
        },
        regression_test_template=(
            "def test_completion_claim_backed_by_a_mutating_call():\n"
            "    receipt = arl_verdict(session_log)\n"
            "    assert receipt.observed_failure != 'artifact_failure'\n"
        ),
    )


# Templated fix-direction artifacts (NOT auto-applyable; config_diff-shaped so the
# patch validator accepts them). Deliberately GENERIC: ARL stores no file content,
# so the exact deleted path lives in the user's own session log, never here.
_R1_FIX_DIRECTION = """\
# Fix direction (NOT auto-applyable): the session log shows a test-path deletion
# followed by a success claim. ARL stores no file content; the exact path is in
# your session log, not here.
- trust the in-session success claim as-is
+ restore the deleted test path(s) from version control (e.g. `git checkout -- <path>`)
+ re-run the full test suite and re-verify the claim against real output
"""

_R2_FIX_DIRECTION = """\
# Fix direction (NOT auto-applyable): the session ended on a completion claim
# with zero mutating tool calls after a change request.
- accept the completion claim as evidence that the change landed
+ diff the working tree / VCS log to verify whether any change actually exists
+ re-run the task if nothing landed; gate the loop on `arl verdict`, not the claim
"""


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
    """Build a SAFE applyable retry-cap target, or None to fall back to the
    non-runnable config_diff.

    SECURITY (fleet HIGH): trace metadata is untrusted. The user APPLIES this diff,
    so:
      * the target PATH must be a repo-relative path — traversal (``..``) and
        absolute paths are refused (no diff pointing outside the repo).
      * the replacement (``after``) line is GENERATED by ARL from the numeric cap,
        NOT taken from attacker-controlled ``after`` text. We locate the integer
        retry budget in the ``before`` line and substitute ``allowed_retries``.
        If ``before`` has no single integer to cap, we cannot safely generate a
        line -> fall back to config_diff.
    """
    raw_target = step.metadata.get("retry_budget_patch_target") or step.metadata.get("arl_patch_target")
    if not isinstance(raw_target, dict):
        return None
    path = _metadata_text(raw_target.get("path"))
    before = _metadata_text(
        raw_target.get("before")
        or raw_target.get("current_line")
        or raw_target.get("current_text")
    )
    if not path or not before or not _is_safe_repo_path(path):
        return None
    # Generate the capped line from `before` — never trust attacker `after` text.
    after = _capped_line(before, allowed_retries)
    if after is None or before == after:
        return None
    return {"path": path, "before": before, "after": after}


def _is_safe_repo_path(path: str) -> bool:
    """True only for a repo-relative path with no traversal. Refuses absolute
    paths, ``..`` segments, and home/UNC prefixes — the diff must stay in-repo."""
    p = path.strip().replace("\\", "/")
    if not p or p.startswith("/") or p.startswith("~") or p.startswith("//"):
        return False
    # Windows drive-letter absolute (C:/...)
    if len(p) >= 2 and p[1] == ":":
        return False
    segments = p.split("/")
    return ".." not in segments


def _capped_line(before: str, allowed_retries: int) -> str | None:
    """Return *before* with its (single) integer retry budget replaced by
    *allowed_retries*. ARL-generated, so no attacker text enters the diff. Returns
    None if *before* does not contain exactly one integer to cap."""
    ints = list(re.finditer(r"\d+", before))
    if len(ints) != 1:
        return None
    m = ints[0]
    return before[: m.start()] + str(allowed_retries) + before[m.end() :]


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
