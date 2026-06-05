# Codex rollout fixtures

These JSONL fixtures mirror the REAL Codex rollout shape verified on disk at
`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` (record types `session_meta`,
`event_msg`, `response_item`, `turn_context`; payloads `function_call`,
`function_call_output`, `custom_tool_call`, `custom_tool_call_output`,
`token_count`, `turn_context`).

They are SANITIZED: no command output bodies, no prompt text, no real
filesystem paths, no `base_instructions`. Only the structural shape is kept
(record/payload types, tool names, `call_id` linkage, exit-status lines,
timestamps) — that is the load-bearing signal the adapter parses.

The record SEQUENCES are taken from real 2026-05-29 sessions:

- `fire_no_edit_retry.jsonl` — a SYNTHETIC no-edit blind retry: the SAME
  `exec_command` runs, fails (exit 1), the model is re-invoked (a reasoning +
  output turn boundary), the SAME command runs again and fails, then a third
  attempt — with NO `apply_patch` between attempts. This is the genuine retry
  loop the detector must FIRE on. (No real session contained a no-edit blind
  retry; see results — real failures are followed by an edit.)

- `abstain_fix_then_rerun.jsonl` — mirrors real session
  `rollout-2026-05-29T08-29-40` indices 87..114: the SAME `pytest` command fails
  (exit 1), the agent EDITS files (`apply_patch`), then re-runs and it passes
  (exit 0). A different tool (`apply_patch`) between the two attempts breaks the
  run -> the detector MUST ABSTAIN (fix-then-rerun is not a blind retry loop).

- `abstain_same_turn_fanout.jsonl` — two IDENTICAL `exec_command` calls emitted
  back-to-back in ONE model turn (no `function_call_output` between them), both
  failing. A same-turn fan-out shares one synthesized `turn_id` -> the B3 guard
  (`_is_one_attempt_per_distinct_turn`) rejects it -> ABSTAIN.
