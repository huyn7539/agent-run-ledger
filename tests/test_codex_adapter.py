"""Codex rollout-log adapter (ONE provider adapter; Codex only).

The Codex CLI writes a JSONL rollout log (one JSON object per line) at
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``. This adapter maps that
provider-specific log into the NEUTRAL ``TraceBundle`` schema. ALL Codex-specific
parsing lives in ``adapters/codex.py``; the core stays provider-neutral.

The adapter records FACTS ONLY — no pricing, no "this was a retry" judgment. The
retry collapse is computed ON READ by ``prescriptions``; these tests assert the
adapter populates exactly the content-free facts the detector keys on
(``span_kind="function"``, a stable ``retry_scope``, a per-turn ``parent_step_id``
synthesized from output-delimited turn boundaries, an ``input_fingerprint``, and a
``has_error`` parsed from the real exit status) so that a real retry FIRES and the
abstain cases ABSTAIN.

False-green guards this file defends (Codex-identified):
  * has_error is PARSED from the exit status ("Process exited with code N" /
    "Exit code: N"), not defaulted to success for every output. Both a failed and
    a succeeded output are asserted.
  * span_kind == "function" for EVERY tool step (exec_command AND apply_patch) —
    anything else and ``retries._is_tool`` never groups it (silent abstain).
  * retry_scope is non-null + STABLE; turn_id (parent_step_id) is per-turn +
    DISTINCT across a real retry, SHARED within a same-turn fan-out. Swapping these
    silently abstains while a one-fixture happy test still passes.
  * tokens are taken from the FINAL cumulative ``token_count`` total, not summed
    over the (cumulative) per-event totals (which over-counts ~N x).
"""

from __future__ import annotations

from pathlib import Path

from agent_run_ledger.adapters.codex import bundle_from_rollout, load_codex_rollout
from agent_run_ledger.core.prescriptions import analyze_bundle, derive_retry_steps

FIXTURES = Path(__file__).parent / "fixtures" / "codex"


def _bundle(name: str):
    return bundle_from_rollout(load_codex_rollout(FIXTURES / name))


# --------------------------------------------------------------------------- #
# Mapping facts (the shape the neutral schema + detector require)
# --------------------------------------------------------------------------- #
def test_rollout_maps_to_a_run_with_function_steps() -> None:
    """The rollout becomes a TraceBundle: a run + one StepRecord per tool call.
    Every tool call (exec_command) is span_kind='function' — the only kind
    ``retries._is_tool`` will group. The model is read from turn_context."""
    bundle = _bundle("fire_no_edit_retry.jsonl")

    assert bundle.run.provider == "codex"
    assert bundle.run.framework == "codex-cli"
    assert bundle.run.model == "gpt-5.5"
    fn_steps = [s for s in bundle.steps if s.span_kind == "function"]
    # 3 exec_command attempts in the fire fixture
    assert len(fn_steps) == 3
    assert all(s.step_type == "function" for s in fn_steps)
    assert all(s.name == "exec_command" for s in fn_steps)


def test_apply_patch_is_a_function_step_too() -> None:
    """apply_patch arrives as a ``custom_tool_call`` (different response-item shape
    than exec_command's ``function_call``) — it MUST still map to a
    span_kind='function' step named 'apply_patch'. Its different NAME is what
    breaks a retry run between two identical commands (fix-then-rerun abstain). If
    it were a non-function kind the run would not break -> false positive."""
    bundle = _bundle("abstain_fix_then_rerun.jsonl")

    patch_steps = [s for s in bundle.steps if s.name == "apply_patch"]
    assert len(patch_steps) == 2
    assert all(s.span_kind == "function" for s in patch_steps)


def test_exit_status_is_parsed_not_defaulted_to_success() -> None:
    """CRITICAL false-green guard: has_error/error is PARSED from the exit status,
    not defaulted to success for every function_call_output. The fire fixture has
    three exit-1 attempts; the fix-then-rerun has a failed (exit 1) then a passed
    (exit 0) exec_command. Both polarities must be represented."""
    fire = _bundle("fire_no_edit_retry.jsonl")
    fire_fn = [s for s in fire.steps if s.span_kind == "function"]
    # every fire attempt failed (exit 1) -> error present on each
    assert all(s.error is not None for s in fire_fn)

    fix = _bundle("abstain_fix_then_rerun.jsonl")
    exec_steps = [s for s in fix.steps if s.name == "exec_command"]
    # exactly one failed (exit 1) and one succeeded (exit 0)
    errored = [s for s in exec_steps if s.error is not None]
    ok = [s for s in exec_steps if s.error is None]
    assert len(errored) == 1
    assert len(ok) == 1


def test_retry_scope_is_stable_and_turn_id_is_per_turn() -> None:
    """The detector keys on retry_scope (must be EQUAL across attempts) and turn_id
    = parent_step_id (must be DISTINCT per attempt in a real cross-turn retry).
    Swapping the two silently abstains. For Codex: retry_scope = the session id
    (coarse-by-design — one session is one agent, the only stable scope); turn_id
    is synthesized per output-delimited turn."""
    bundle = _bundle("fire_no_edit_retry.jsonl")
    fn_steps = sorted(
        (s for s in bundle.steps if s.span_kind == "function"),
        key=lambda s: s.started_at,
    )

    # retry_scope: present, non-null, identical across the 3 attempts
    scopes = {s.retry_scope for s in fn_steps}
    assert None not in scopes
    assert len(scopes) == 1

    # turn_id (parent_step_id): present, non-null, ALL DISTINCT across turns
    turn_ids = [s.parent_step_id for s in fn_steps]
    assert all(t is not None for t in turn_ids)
    assert len(set(turn_ids)) == len(turn_ids)

    # input fingerprint present + identical (same command) across attempts
    fps = {s.input_fingerprint for s in fn_steps}
    assert None not in fps
    assert len(fps) == 1


def test_same_turn_fanout_shares_one_turn_id() -> None:
    """Two identical exec_commands emitted in ONE model turn (no function_call_output
    between them) must share ONE synthesized turn_id — that is what lets the B3
    guard (`_is_one_attempt_per_distinct_turn`) reject the fan-out as a false
    positive."""
    bundle = _bundle("abstain_same_turn_fanout.jsonl")
    fn_steps = [s for s in bundle.steps if s.span_kind == "function"]

    assert len(fn_steps) == 2
    # both calls came before any output -> SAME turn id
    assert fn_steps[0].parent_step_id == fn_steps[1].parent_step_id
    assert fn_steps[0].parent_step_id is not None


def test_run_tokens_from_final_cumulative_not_summed() -> None:
    """token_count events are CUMULATIVE running totals. The run total must equal
    the FINAL total_token_usage, not the sum of every event's total (which would
    over-count ~N x). The fire fixture's final cumulative is 3000 in / 150 out."""
    bundle = _bundle("fire_no_edit_retry.jsonl")

    assert bundle.run.total_input_tokens == 3000
    assert bundle.run.total_output_tokens == 150


def test_per_step_tokens_are_not_double_counted_from_output_size() -> None:
    """Per-step model tokens are 0 (honest): Codex logs do not break model usage
    per call, and the 'Original token count' inside the command output is the
    command's OUTPUT size, not model usage. Summing steps must not exceed the run
    total (no double counting)."""
    bundle = _bundle("fire_no_edit_retry.jsonl")
    step_input = sum(s.input_tokens for s in bundle.steps)
    step_output = sum(s.output_tokens for s in bundle.steps)

    assert step_input == 0
    assert step_output == 0


# --------------------------------------------------------------------------- #
# Detector behaviour on read — fire vs abstain (the load-bearing requirement)
# --------------------------------------------------------------------------- #
def test_genuine_no_edit_retry_fires() -> None:
    """A real retry loop: SAME exec_command, exit 1, re-invoked across distinct
    turns with NO apply_patch between -> exactly ONE retry prescription, severity
    high (terminal failure)."""
    bundle = _bundle("fire_no_edit_retry.jsonl")

    prescriptions = analyze_bundle(bundle)
    assert len(prescriptions) == 1
    assert prescriptions[0].severity == "high"

    collapsed = [s for s in derive_retry_steps(bundle) if s.span_kind == "function"]
    # 3 attempts collapse to one step with retry_count=2
    assert len(collapsed) == 1
    assert collapsed[0].retry_count == 2


def test_fix_then_rerun_abstains() -> None:
    """The SAME pytest command failed, the agent EDITED files (apply_patch), then
    re-ran and it passed. A different tool (apply_patch) between the two attempts
    breaks the run -> NO retry prescription. Dropping apply_patch would make this a
    false positive — this test fails if the adapter filters it out."""
    bundle = _bundle("abstain_fix_then_rerun.jsonl")

    assert analyze_bundle(bundle) == []


def test_same_turn_fanout_abstains() -> None:
    """Two identical exec_commands in ONE turn is a fan-out, not a retry loop. The
    B3 guard (one attempt per distinct turn) rejects it -> no prescription."""
    bundle = _bundle("abstain_same_turn_fanout.jsonl")

    assert analyze_bundle(bundle) == []


def test_clean_session_yields_zero_prescriptions() -> None:
    """A normal session with no repeated-same-input failures produces ZERO
    prescriptions (no false positives). The fix-then-rerun + fan-out fixtures both
    represent legitimate work; neither may emit a prescription."""
    assert analyze_bundle(_bundle("abstain_fix_then_rerun.jsonl")) == []
    assert analyze_bundle(_bundle("abstain_same_turn_fanout.jsonl")) == []


# --- A1: user-message boundary must break the retry continuation (vault-CC, 2026-06-05) ---
def _rollout_user_directed_reruns() -> list[dict]:
    """Same exec_command fails, the USER says 'please rerun', it fails again, user
    again, then it succeeds. A user DIRECTING the reruns is NOT a blind retry loop."""
    return [
        {"type": "session_meta", "payload": {"id": "sessUB"}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"pytest\"}", "call_id": "c1"}, "timestamp": "2026-05-29T12:00:01Z"},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1", "output": "Exit code: 1"}, "timestamp": "2026-05-29T12:00:02Z"},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "please rerun"}, "timestamp": "2026-05-29T12:00:03Z"},
        {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"pytest\"}", "call_id": "c2"}, "timestamp": "2026-05-29T12:00:04Z"},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c2", "output": "Exit code: 1"}, "timestamp": "2026-05-29T12:00:05Z"},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "again"}, "timestamp": "2026-05-29T12:00:06Z"},
        {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"pytest\"}", "call_id": "c3"}, "timestamp": "2026-05-29T12:00:07Z"},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c3", "output": "ok"}, "timestamp": "2026-05-29T12:00:08Z"},
    ]


def test_user_directed_reruns_do_not_false_fire_retry() -> None:
    from agent_run_ledger.adapters.codex import bundle_from_rollout
    from agent_run_ledger.core.prescriptions import analyze_bundle
    bundle = bundle_from_rollout(_rollout_user_directed_reruns())
    assert analyze_bundle(bundle) == []  # user directed the reruns -> NOT a blind retry loop


def test_unrelated_user_message_does_not_suppress_a_genuine_retry() -> None:
    """Over-abstain guard: a user message that is NOT between the repeated calls must
    NOT suppress a genuine no-boundary retry."""
    from agent_run_ledger.adapters.codex import bundle_from_rollout
    from agent_run_ledger.core.prescriptions import analyze_bundle
    recs = [
        {"type": "session_meta", "payload": {"id": "sessG"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "fix the crm bug"}, "timestamp": "2026-05-29T12:00:00Z"},
        {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"crm\"}", "call_id": "g1"}, "timestamp": "2026-05-29T12:00:01Z"},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "g1", "output": "Exit code: 1"}, "timestamp": "2026-05-29T12:00:02Z"},
        {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"crm\"}", "call_id": "g2"}, "timestamp": "2026-05-29T12:00:03Z"},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "g2", "output": "Exit code: 1"}, "timestamp": "2026-05-29T12:00:04Z"},
        {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"crm\"}", "call_id": "g3"}, "timestamp": "2026-05-29T12:00:05Z"},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "g3", "output": "ok"}, "timestamp": "2026-05-29T12:00:06Z"},
    ]
    bundle = bundle_from_rollout(recs)
    assert len(analyze_bundle(bundle)) == 1  # genuine autonomous retry STILL fires
