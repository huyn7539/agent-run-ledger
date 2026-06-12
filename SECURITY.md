# Security Policy

## What "secure" means for this project

ARL's core security claim is **zero egress**: the package contains no outbound
network code, no telemetry, and no update checks. This is enforced by AST-level
tests that fail the build if outbound socket primitives or dynamic imports are
introduced (`tests/` — the egress guard). The optional `arl serve` dashboard
binds `127.0.0.1` only.

Because of that, these are treated as **security vulnerabilities**, not bugs:

1. **Any code path that sends data off the machine** — highest severity.
2. **Content leaks into "content-free" outputs** — receipts, verdicts, exports,
   and sweep output must never contain prompts, code, command text, file
   contents, or usernames (enforced by the leak-matrix tests). A leak into any
   shareable artifact is a vulnerability.
3. **Unsafe parsing of session logs** — ARL treats all input as hostile
   (size/depth-bounded parsing, typed errors, nothing evaluated). A crafted
   `.jsonl` that escapes those bounds (code execution, unbounded memory,
   path traversal via embedded paths) is a vulnerability.
4. **`arl serve` binding beyond loopback** or serving content it shouldn't.

## Reporting

Use **GitHub private vulnerability reporting** (Security tab → "Report a
vulnerability") on this repository. If that's unavailable, open a plain issue
saying only "security report, need a private channel" — do not include details
— and a channel will be arranged.

You can expect an acknowledgment within 72 hours. This is a solo-maintained
project; fixes for egress/leak-class reports take priority over everything else.

## Supported versions

Only the latest release line (currently 0.x latest) receives security fixes.

## Verifying the claims yourself

Don't take the README's word for it: the egress-guard and leak-matrix tests run
in CI on every commit, and `arl export` scrubbing is testable locally. Auditing
those tests is the recommended first step before trusting any verification tool —
this one included.
