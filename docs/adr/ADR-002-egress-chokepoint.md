# ADR-002 — All network egress routes through one fail-closed chokepoint

Date: 2026-05-29
Status: ACCEPTED. The *contract* is locked
now; the `emit()` body, consent reader, and ShapeEvent build are DEFERRED (D2).

## Context

ARL is local-first / zero-egress: content (prompts, outputs, error messages,
arbitrary metadata) never leaves the user's machine without explicit per-export
opt-in. Today there is ZERO network egress anywhere under `src/agent_run_ledger`
(enforced by the L10 import guard: no module imports requests/httpx/urllib/
urllib3/aiohttp/socket/http.client). That is the safest possible state. The risk
is a future "ship aggregate shape telemetry" feature added under deadline that
opens a socket directly, or redacts-then-ships a `TraceBundle` and inherits every
field — a leak-by-default.

## Decision (LOCKED NOW)

1. **Single chokepoint.** ALL future network egress (telemetry, shared-export,
   crash reporting) MUST route through one function: `core/telemetry.emit()`.
   Nothing else under `src/` may open a socket. The L10 import guard enforces the
   negative; `emit()` will be the only sanctioned exception when it is built.

2. **Fail closed on consent.** `emit()` sends NOTHING unless consent is
   explicitly present. Unknown/absent/unreadable consent ⇒ send nothing. There is
   no fail-open path.

3. **ShapeEvent is a hand-built closed type (L12).** When telemetry is built, it
   MUST construct a separate narrow `ShapeEvent` dataclass FROM SCRATCH —
   enumerated scalars only (`run_count: int`, `workflow_type: Enum`,
   `prescription_class: Enum`, `fired: bool`, `applied: bool`,
   `anon_install_id: hash`) — with NO reference to `TraceBundle` and mapped
   through an explicit named projection. **Never redact-then-ship a bundle.** A
   redacted bundle inherits every field, so a new base column becomes a silent
   new egress field (leak-by-default). The closed type makes a forgotten new
   column impossible to leak.

## Deferred (D2 — do NOT build now)

The `emit()` body, the consent-file reader, the `ShapeEvent` build, and the
fail-closed RED test are brick-2-era work, gated on the first real telemetry
feature with consent semantics actually decided. The construction-and-
reversibility review for the provenance digest (LR7) binds to THAT PR, not v1.

## What protects us in the interim

- **L10 import guard** (`tests/test_egress_guards.py`) — a naive `requests.post()`
  cannot merge.
- **This ADR** — the one-sentence contract above is the design lock.
- **L13 redaction contract** in the schema docstring — any new content column
  must register a sensitivity tag and route through the same redaction + consent
  contract before it can ship.
