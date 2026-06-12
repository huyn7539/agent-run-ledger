# Prescription Taxonomy - Operator Gate

Status: operator approval required.

This file is the canonical taxonomy location, but this pass is still a Codex
implementation draft. It should not be treated as closing the operator-owned
taxonomy gate until Hung reviews and rewrites or explicitly approves it.

## Shared Semantics

`retry_count` means additional attempts after the first attempt.

Example: `retry_count = 4` means 5 total attempts.

## V0 Class: Retry/Cost Loop

Detect when a single step keeps retrying and materially increases cost,
latency, or failure ambiguity.

Default trigger:

- `retry_count >= 2`
- step is a model, tool, function, or custom-instrumented app step
- false-positive guards do not suppress the finding

Default allowed retries:

- `allowed_retries = 0`
- V0 favors fail-fast over hidden retry loops because the product's core promise is
  measurable budget control, not silent recovery.

Cost calculation:

```text
total_attempts = 1 + retry_count
excess_retries = max(retry_count - allowed_retries, 0)
wasted_cost = step.cost_usd * excess_retries / total_attempts
```

For `cost_usd = 0.092`, `retry_count = 4`, and `allowed_retries = 0`,
the expected wasted cost is `0.0736`.

## Artifact Rules

Retry/cost-loop prescriptions default to `patch_type = "config_diff"` unless
the trace includes explicit target-file context. Do not emit an applyable
`unified_diff` unless the trace or repo analysis identifies the target file,
current text, and replacement text.

Allowed patch types:

- `unified_diff`: starts with `diff --git` or standard `---`/`+++` diff
  headers, includes at least one hunk, and is backed by target context.
- `code_snippet`: runnable code snippet, not prose.
- `config_diff`: config-oriented diff or key/value change.
- `regression_test`: test body proving the expected budget/failure behavior.

When a unified diff is emitted, it must apply cleanly with:

```powershell
git apply --check -
```

## Severity

- `high`: retry loop ended in an error or exceeded a budget on a critical path.
- `medium`: retry loop consumed measurable cost/latency but did not error.
- `low`: informational only; V0 should avoid emitting low-severity prescriptions.

## False-Positive Guards

Suppress or downgrade when:

- the step explicitly declares a higher retry budget in metadata
- retries are spread across separate user-requested runs
- retry count is unknown or inferred only from natural-language logs
- cost and latency are both unavailable

## Non-Goals

- automatically applying patches
- claiming business correctness
- exposing raw prompts or tool payloads
- pretending SDK token usage is the same as dollar cost
