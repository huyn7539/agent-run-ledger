# ADR-001 — Redaction is a single chokepoint at StepRecord construction (allowlist)

Date: 2026-05-29
Decider: maintainer
Status: ACCEPTED — implement before Phase A code lands.

## Context

ARL is an accountability layer; **"nothing leaks" is the core trust claim, not a feature**.
A dedicated security review (2026-05-29) found the prompt-leak surface is wider than the
original design assumed:

- **Two** StepRecord construction paths exist, not one:
  1. `adapters/openai.py:80` — live/recorded-trace path. `_extract_error` puts raw `str(error)` (echoes prompts) in.
  2. `models.py:116 StepRecord.from_dict` line 129 — the **user-import path**. `error=str(data["error"])` stores
     a leaked prompt VERBATIM. This is the PRIMARY user entry point ("paste trace → get prescription") and the
     original brief's adapter-only redaction left it fully unprotected.
- **Three** egress channels read `step.error` / metadata: SQLite (`storage.py:132`), HTML report
  (`report.py:21` — `escape()` neutralizes markup but does NOT remove a readable prompt), exported JSON (`io`).

A key-name blocklist patch in the adapter (the cheap fix) leaves the import path and the report/JSON egress
leaking. That is whack-a-mole on a trust-critical surface.

## Decision

**Redact ONCE, at StepRecord construction, with a two-category model.** A shared sanitizer runs in StepRecord's
construction (`__post_init__` or a single `_sanitize` called by BOTH `from_dict` and the adapter builder), so both
construction paths are covered (adapter AND user-import).

Category 1 is AUTO-CAPTURED content: `error`, arbitrary metadata values, and arbitrary metadata key names. This
category is hard-redacted by construction before SQLite, HTML, JSON, or future egress channels can observe it.
`error` is reduced to a safe static message — never the raw `str(e)` that may carry prompt text. Unlisted metadata
keys are omitted entirely, so key names cannot become a side channel; unlisted metadata values are omitted with
their keys.

Category 2 is USER-CONTROLLED fields. Labels (`name`, `workflow`, `step_type`, `id`, `success_label`) and
diff-instrumentation metadata (`before`, `after`, `path`, `current_text`, `replacement_text`, `current_line`,
`replacement_line`) pass through by design. They are documented as not-for-sensitive-content. They are not scrubbed:
scrubbing diff content would destroy the product's diffs and diagnostics.
RunRecord fields (`prompt_hash`, `config_hash`, `framework`, `provider`, `model`, `workflow`, `success_label`) are
Category 2: user-/SDK-supplied labels and hashes, pass-through by design, not auto-captured Category-1 content.

The guarantee is: auto-captured content never leaks. User-placed content in Category 2 fields is the
user's responsibility, documented.

## Why this is the end-picture decision, not a bit

This is the design working. If redaction is sound-by-design at the chokepoint, the entire leak CLASS dies in one
place. If it's per-channel, every new output (a future CSV export, an API response, a webhook) re-opens the hole.

## Consequences

- The leak test (sentinel absent from SQLite bytes AND `step.error` column) extends to assert the sentinel is
  also absent from `render_report()` output and `write_trace()` JSON bytes (test-engineer H2), AND that the
  USER-IMPORT path (`from_dict` with a sentinel in `error`) is redacted (test-engineer's missed path).
- Allowlist may drop fields a future feature wants; adding a field to Category 2 is a deliberate, reviewed change
  and must document why verbatim egress is required — which is correct for a trust surface.
- `_extract_error` no longer returns raw `str(e)`; it returns a sanitized class+message. Live-path error
  diagnostics get coarser — acceptable; accountability > debuggability on the leak surface.

## Test-engineer findings folded into the implementation brief (C1–C5, H1–H4)
- C1: BUG-4 tests must NOT assert `result.exception is None` (it's `SystemExit(1)` after `typer.Exit`). Use
  `exit_code==1` + `"error:" in output` + `not isinstance(result.exception, (JSONDecodeError, FileNotFoundError, TraceValidationError))`.
- C2: BUG-4 RED proof must show empty output + raw domain exception pre-fix (exit-code-only is born-green).
- C3: `test_composite_pk_collision` + `init_db` idempotency are BORN-GREEN characterization, NOT BUG-1 red proofs.
  Genuine BUG-1 red proofs: `test_cascade_delete_removes_children` + `test_fk_rejects_orphan_step_insert` +
  `test_connect_enables_foreign_keys` (the tightest — pins pragma placement).
- C4: `PRAGMA foreign_keys=ON` is a silent no-op if issued inside an open transaction. Must be immediately after
  `sqlite3.connect()`, before any DML. Cascade test MUST delete via `storage.connect()` (carries pragma), not raw
  `sqlite3.connect()`. Add `test_connect_enables_foreign_keys` asserting `PRAGMA foreign_keys` returns 1.
- C5: split negative/non-finite cost tests into step-cost vs run-total (validate() raises on step first, so a
  combined fixture lets the run-total guard ship unverified).
- H1: keep BOTH leak assertions (raw-bytes forces metadata fix; error-column forces _extract_error fix) —
  non-redundant. Mandate RED capture showing sentinel PRESENT today in both channels.
- H2: extend leak test to HTML report + exported JSON egress (THIS ADR's chokepoint makes them pass by design).
- H3: pin that no bare NaN/Infinity can reach output JSON (`json.loads(..., parse_constant=raises)`); guard lives
  in validate(), not ingest (`cost_usd=NaN` passes `_as_float` silently).
- H4: parametrize BUG-3 digit-collision over 3 adversarial shapes (name-collision, value-substring, name+value).
- Count: FIVE red→green pairs expected at the gate (BUG-1..4 + leak), not four.

## H3 resolution — labels vs content

`name`, `workflow`, `step_type`, `id`, and `success_label` are operator-controlled labels, not auto-captured
content. They are intentionally not redacted because prescriptions and comparisons rely on them for useful
diagnostics. Operators must not put prompt text, tool arguments, raw responses, or other sensitive content in label
fields.

Diff instrumentation (`before`, `after`, `path`, `current_text`, `replacement_text`, `current_line`,
`replacement_line`) is also Category 2. These values pass through verbatim because the emitted patch and
diagnostic evidence depend on the exact text. Operators must not place sensitive content in these fields.

The hard redaction surface is Category 1 auto-captured content: `error`, arbitrary metadata values, arbitrary
metadata key names, and tool arguments captured as arbitrary metadata. Prescriptions consume only the already
sanitized `step.error` plus Category 2 fields; they must not consume raw auto-captured content fields.
