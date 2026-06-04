"""B1 — the HONEST gate: a REAL captured OpenAI Agents SDK run (stub Model, no API
key, no network) produces a receipt at its TRUE level.

This is the demo-overfit guard (06-learning/.../2026-05-28-codex-failure-mode-demo-
overfit.md). The earlier L2 demo passed only because a HAND-AUTHORED fixture injects
``retry_budget_patch_target`` into the function span. The native SDK
``FunctionSpanData.export()`` returns ONLY ``type/name/input/output/mcp_data`` (no
patch target), so a REAL captured retry loop honestly grades **L1**: ARL proves the
retry-loop DETECTION, but the cheap-strong L2 static-repair proof needs app
instrumentation that supplies a safe patch target.

We drive the REAL installed SDK via the shared harness (scripts/live_capture_
harness.py, also used by the demo so the GATE line and these asserts share ONE real
run). The captured span tree is the real ``task -> agent -> turn -> function`` shape;
ARL derives the loop and emits one prescription, and the receipt is L1 — NOT the
fixture's L2. If this ever flips to L2 without a tested instrumentation path, the
demo-overfit failure has returned.
"""

from __future__ import annotations

import pytest

pytest.importorskip("agents", reason="OpenAI Agents SDK not installed")

from agent_run_ledger.core.prescriptions import derive_retry_steps  # noqa: E402
from live_capture_harness import capture_real_retry_run  # noqa: E402


def test_real_captured_sdk_run_has_cross_turn_function_topology(tmp_path) -> None:
    """The REAL shape: task -> agent -> turn -> function. The repeated crm_lookup
    function spans have DIFFERENT immediate (turn) parents but the SAME agent
    scope. (This is what makes the cross-turn detection correct AND what the
    same-turn B3 guard discriminates against.)"""
    bundle, _, _ = capture_real_retry_run(tmp_path)
    fn_steps = [s for s in bundle.steps if s.span_kind == "function"]
    assert len(fn_steps) == 3
    # 3 distinct immediate turn parents, one shared agent scope:
    assert len({s.parent_step_id for s in fn_steps}) == 3
    assert len({s.retry_scope for s in fn_steps}) == 1
    # native function spans carry NO patch-target metadata (the L1 root cause):
    assert all("retry_budget_patch_target" not in s.metadata for s in fn_steps)


def test_real_captured_sdk_run_derives_the_retry_loop(tmp_path) -> None:
    """Detection is SOUND on the real run: the 3 cross-turn attempts collapse to
    one derived step with retry_count=2, and one prescription is emitted."""
    bundle, prescriptions, _ = capture_real_retry_run(tmp_path)
    collapsed_fn = [s for s in derive_retry_steps(bundle) if s.span_kind == "function"]
    assert len(collapsed_fn) == 1
    assert collapsed_fn[0].retry_count == 2
    assert len(prescriptions) == 1


def test_real_captured_sdk_run_receipt_is_l1_not_l2(tmp_path) -> None:
    """THE GATE (B1): a REAL captured run honestly grades L1 — NOT the fixture's
    L2. Native SDK function spans export no patch target, so the artifact is the
    non-runnable config_diff fallback (relevance, not mechanical removal). L2 on a
    real run would require a TESTED app-instrumentation path; without it, claiming
    L2 here is the demo-overfit failure. This test fails if that regresses."""
    _, _, receipts = capture_real_retry_run(tmp_path)
    assert len(receipts) == 1
    r = receipts[0]
    assert r.proof_level == "L1", (
        f"a REAL captured run must honestly grade L1, got {r.proof_level} "
        "(L2 on a native SDK span means fixture metadata leaked into the gate)"
    )
    assert r.observed_failure == "retry_loop"
    assert r.repair_artifact["patch_type"] == "config_diff"
