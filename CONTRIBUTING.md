# Contributing

Thanks for considering it. This project is early and small; the bar for a good
contribution is clarity, not size.

## What's most useful right now

1. **Field reports beat code.** A receipt that was wrong, a failure ARL graded
   clean, or a fix you actually applied — filed as an issue with the four
   artifacts from the README's "Reporting issues" section — is worth more than
   any feature PR today.
2. **Adapters.** If your harness writes session logs, the neutral `TraceBundle`
   JSON format is the integration point (`src/agent_run_ledger/core/models.py`;
   `arl export` shows the shape). An adapter PR should come with real
   (scrubbed) fixture files.
3. **Detector fixtures.** Known-bad session shapes (retry loops, false success
   claims) as test fixtures — especially ones where ARL currently misses or
   misfires.
4. **Bug fixes** with a failing test first.

## Hard rules (PRs violating these will be declined regardless of quality)

- **Zero egress is structural.** No outbound network code, no telemetry, no
  update checks, no "just one optional ping." The AST egress-guard tests will
  fail your build; do not weaken them.
- **Content-free outputs stay content-free.** Nothing a user might share
  (receipts, verdicts, exports, sweep output) may carry prompts, code, command
  text, or usernames. The leak-matrix tests are the contract.
- **Honest grading only.** No detector may claim more than its proof level
  supports. "Clean" must always be scoped to checked classes. A detector that
  can't state its limits doesn't ship.
- **Tests first** for behavior changes. Run `uv run pytest` — the suite must
  be green on your machine before you open the PR.

## Developer setup

```bash
git clone https://github.com/huyn7539/agent-run-ledger
cd agent-run-ledger
uv sync --extra dev
uv run pytest        # full suite; ~30s
uv run arl selftest  # end-to-end pipeline check
```

## License terms for contributions

This project is licensed under FSL-1.1-ALv2 (see [LICENSE.md](LICENSE.md) and the
plain-English summary in the README).

By submitting a contribution you agree that:

1. Your contribution is licensed to the project under **FSL-1.1-ALv2**,
   including its automatic conversion to Apache-2.0 two years after each
   release; and
2. You grant the maintainer (Hung Huynh) a perpetual, worldwide, irrevocable
   right to **relicense your contribution** as part of the project under other
   license terms (for example, a future move to a more permissive license).
3. You certify the [Developer Certificate of Origin](https://developercertificate.org/)
   — sign your commits with `git commit -s` (`Signed-off-by:` line).

Clause 2 exists so the project can ever become *more* open (or adjust terms)
without hunting down every past contributor. It is a one-way valve toward
flexibility, not a rights grab — your contribution can never be made *less*
available to you than the license you submitted it under.

## Process

- Open an issue before large PRs; small fixes can go straight to a PR.
- One change per PR. Match the existing code style (ruff, line length 100).
- CI must be green (ubuntu + windows lanes).
