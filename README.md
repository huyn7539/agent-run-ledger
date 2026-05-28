# Agent Run Ledger

Agent Run Ledger is a local-first CLI for AI agent workflows.

Wedge:

> Every agent run gets a receipt, a diff, and a fix artifact.

V0 records agent runs, stores them in SQLite, renders static HTML reports, compares runs, and emits a retry/cost-loop prescription with a concrete patch artifact.

## Quick Start

Install the checkout in editable mode:

```powershell
uv pip install -e .
```

Then run:

```powershell
uv run arl init
uv run arl run-demo --variant retry-loop
uv run arl list-runs
uv run arl report --run <RUN_ID>
uv run arl run-demo --variant clean
uv run arl compare --left <RUN_ID_A> --right <RUN_ID_B>
```

Without installing the console script, use `uv run` from the repo:

```powershell
uv run python -m agent_run_ledger.cli init
uv run python -m agent_run_ledger.cli run-demo --variant retry-loop
```

## V0 Scope

Included:

- Provider-neutral trace schema.
- Local SQLite storage.
- JSON import/export.
- Static HTML report.
- Run comparison.
- Retry/cost-loop prescription with patch artifact.
- OpenAI adapter isolated outside the core package.
- Read-only Claude/Codex review bus under `.agentbus/`.

Not included:

- Hosted SaaS.
- Auth or billing.
- Public dashboard.
- Memory graph.
- Autonomous patch application.
- Multi-provider adapters.

## Current Gates

Still open before builder demos:

- 5 customer-discovery interviews, with at least 3 validating the patch-artifact pain.
- Operator approval of `docs/prescription-taxonomy.md`.
- One live OpenAI Agents SDK run with an API key. The recorded fixture proves adapter
  normalization only; it is not a live-run proof.

## Private Alpha Definition

An activated install means a builder produces at least one trace report on their own machine. Package downloads, stars, and README views do not count.
