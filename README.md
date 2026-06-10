# Agent Run Ledger

Agent Run Ledger (ARL) is a local-first CLI that turns AI coding-agent runs into
**graded repair receipts** — and, in verdict mode, into a machine-consumable exit
code your loop can gate on.

Wedge:

> Every agent run gets a ledger record. Every DETECTED failure gets a graded repair
> receipt — and, when the evidence supports it, a concrete fix artifact.

(A clean run with no detected failure emits no prescription and no receipt — by
design: ARL does not invent receipts where there is nothing to repair. "The run
finished" and "the run is verified clean" are different claims; ARL exists to keep
them different.)

Everything runs on your machine. Prompt/output content never leaves it. There is no
server, no telemetry, no network code in the package (enforced by tests).

## Quick Start

Install the `arl` command (no account, no config, nothing leaves your machine):

```powershell
uv tool install .
```

See the alarm work first (a bundled known-bad run through the real pipeline),
then grade your newest local agent session:

```powershell
arl selftest                  # proves a receipt fires — so 'clean' means something
arl verdict --latest          # newest Codex CLI session
arl verdict --latest-claude   # newest Claude Code session
```

ARL's value concentrates on the runs you are NOT watching — unattended loops,
scheduled jobs, CI lanes, the overnight batch. If you read every diff
interactively, expect clean verdicts; that is the detector abstaining, and it is
the honest answer.

Or the classic ledger flow:

```powershell
arl run-demo --variant retry-loop    # stores run "run_retry_loop"
arl run-demo --variant clean         # stores run "run_clean_demo"
arl list-runs
arl report --run run_retry_loop
arl compare --left run_retry_loop --right run_clean_demo
```

## Verdict mode — the loop contract

Autonomous loops today exit on tests-pass, string matching, or the agent's own
say-so. `arl verdict` gives a loop an independent, graded exit:

| Exit | Meaning |
|---|---|
| `0` | clean — no structural failure detected (the honest negative) |
| `3` | one or more repair receipts fired — attention |
| `1` | error — unreadable/invalid input **fails closed**: an unparseable run is never silently clean |

`--json` prints a stable machine schema (`arl.verdict/v1`) on stdout:

```json
{
  "schema": "arl.verdict/v1",
  "run_id": "codex_0123…",
  "verdict": "receipts",
  "receipt_count": 1,
  "max_proof_level": "L2",
  "receipts": [ { "claim": "…", "observed_failure": "retry_loop",
                  "proof_level": "L2", "confidence": "medium",
                  "repair_artifact": { "…": "…" }, "limits": ["…"],
                  "next_evidence": ["…"] } ]
}
```

Proof levels are the L0–L6 ladder. L2 means the fix *mechanically removes the
deterministic failure path, verifiable without a re-run* — graded by static
inspection of a templated artifact, never by the model's self-report. Every receipt
carries `limits` (what is NOT proven). Receipts are advisory: ARL never applies a
patch.

Every verdict — including clean — states its **detector coverage** (`coverage` in
the JSON): what was checked and what was NOT. `clean` means "clean for the checked
classes," never "verified correct." Run `arl selftest` once to watch a receipt fire
through the real pipeline; after that, silence is information.

### Recipes

**A bash loop that stops on a dirty run (Ralph-style):**

```bash
while :; do
  run-my-agent
  arl verdict --latest || break   # exit 3 (receipt) or 1 (unreadable) stops the loop
done
```

**A CI step (receipt as job evidence):**

```yaml
- name: ARL verdict on the agent session
  run: |
    arl verdict path/to/session.jsonl --json > arl-verdict.json
    # exit 3 fails the step when a receipt fires; artifact carries the receipt
  continue-on-error: true
- uses: actions/upload-artifact@v4
  with: { name: arl-verdict, path: arl-verdict.json }
```

**A scheduled check of your latest unattended session:**

```powershell
arl verdict --latest --json | Out-File verdict.json   # 0 clean / 3 receipts / 1 error
```

**Sweep an existing archive (months of evidence you already own):**

```powershell
arl sweep ~/.claude/projects --json    # batch-verdict every session log under a root
arl sweep ~/.codex/sessions
```

`sweep` is read-only by default (`--save` to record), caps at `--limit 200`
newest-first, and exits 0 (no receipts anywhere) / 3 (any file fired) / 1
(total failure: nothing readable). Per-file errors are counted and shown, never
silently skipped.

**A Claude Code Stop hook (receipt on every finished session, zero remembered steps):**

```jsonc
// .claude/settings.json
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command",
          "command": "arl verdict --latest-claude --json >> .arl/verdicts.jsonl" } ] }
    ]
  }
}
```

> Mechanics note: Claude Code hooks treat only **exit code 2** as blocking — ARL's
> exit 3 logs the receipt but will not block the session. If you want a fired
> receipt to BLOCK, map it in the hook command:
> `arl verdict --latest-claude --json >> .arl/verdicts.jsonl || exit 2`
> (any non-zero ARL exit — receipt or unreadable — becomes a blocking 2).

## What ARL reads today (honest scope)

- **Codex CLI** session rollouts (`~/.codex/sessions/**/rollout-*.jsonl`) — live, today.
- **Claude Code** session logs (`~/.claude/projects/**/*.jsonl`, including subagent
  sessions) — live, today. A chat-only session with no tool calls errors honestly
  ("no run to record") rather than grading something it cannot see.
- **OpenAI Agents SDK** recorded trace exports (JSON) — live, today.
- **Neutral TraceBundle JSON** (ARL's own schema) — live, today.

All input is treated as hostile: size/depth-bounded parsing, typed errors, nothing
evaluated. Trace content never leaves the machine.

## V0 Scope

Included: provider-neutral trace schema · local SQLite storage · JSON import/export ·
static HTML report · run comparison · retry/cost-loop prescription with patch
artifact · success-lie detector (R1 success claim after test deletion, R2
completion claim with zero mutating calls; graded L0–L1, abstain-by-default) ·
graded RepairReceipts (L0–L2 implemented honestly) · verdict mode with
the loop exit contract · archive sweep (`arl sweep`) · OpenAI + Codex adapters
isolated outside the core package · read-only Claude/Codex review bus under
`.agentbus/`.

Not included: hosted SaaS · auth/billing · public dashboard · memory graph ·
autonomous patch application · telemetry of any kind.

## Status

The detector and receipts run on real Codex sessions today; most well-run
interactive sessions grade **clean**, which is the honest expected result — the
target population is unattended/scheduled/CI runs, where waste hides. The current
validation bar (2026-06-10): graded receipts produced on real users' real failures,
measured by whether they **apply** the fix — not stars, not installs.

## Private Alpha Definition

An activated install means a builder produces at least one verdict or trace report
on their own machine. Package downloads, stars, and README views do not count.
