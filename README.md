<h1 align="center">Agent Run Ledger</h1>

<p align="center"><strong>Did your AI coding agent actually do what it claimed?</strong></p>

<p align="center"><strong>graded receipts · honest abstain · exit codes for loops · 100% local · zero egress</strong></p>

<p align="center">
  <a href="https://github.com/huyn7539/agent-run-ledger/actions/workflows/ci.yml"><img src="https://github.com/huyn7539/agent-run-ledger/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/huyn7539/agent-run-ledger/releases"><img src="https://img.shields.io/github/v/release/huyn7539/agent-run-ledger" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-FSL--1.1--ALv2-blue.svg" alt="License: FSL-1.1-ALv2"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python 3.12+">
</p>

<p align="center">
  <a href="#get-started-60-seconds">Get started</a> ·
  <a href="#how-it-works-30-seconds">How it works</a> ·
  <a href="#gate-your-loop-on-it">Gate your loop</a> ·
  <a href="#honest-scope">Honest scope</a> ·
  <a href="#paid-pilots--audits">Paid pilots</a>
</p>

---

ARL reads the session logs your coding agent already writes — Claude Code, Codex
CLI, OpenAI Agents SDK — and tells you when a run *lied*: retry loops that burned
your quota, success claims with nothing behind them. Every detected failure
becomes a **graded repair receipt** with a proof level, a fix direction, and a
stated list of what is *not* proven. Every clean verdict names exactly what was
checked. Nothing ever leaves your machine — there is no outbound network code in
the package, and a build-failing test keeps it that way.

## Get started (60 seconds)

**1. Install** (Python 3.12+; no account, no config):

```bash
uv tool install git+https://github.com/huyn7539/agent-run-ledger
```

<details><summary>pip / pipx / from a checkout</summary>

```bash
pip install git+https://github.com/huyn7539/agent-run-ledger
pipx install git+https://github.com/huyn7539/agent-run-ledger

# if `arl` isn't on PATH afterwards (machines with several Pythons):
python -m agent_run_ledger --version    # always works
```
</details>

**2. Prove the alarm works** — selftest runs a bundled known-bad session through
the real pipeline:

```text
$ arl selftest
selftest: running a bundled known-bad run through the real pipeline
  receipt fired: retry_loop at L1 (confidence low)
  fix direction: Set demo.flaky_tool retry budget to 0 and fail closed.
selftest: PASS — the alarm fires; 'clean' on your runs means the detector
abstained, not that it is deaf
```

**3. Grade the sessions you already have** — months of evidence are sitting on
your disk right now:

```bash
arl sweep ~/.claude/projects     # Claude Code archive
arl sweep ~/.codex/sessions      # Codex CLI archive
```

Each file gets a verdict: `clean`, `fired` (with receipts), or `error` — counted
separately, never hidden. Start with the archive, not today's session: failures
hide in the runs you *weren't* watching — unattended loops, overnight batches,
CI lanes. If you babysat every diff, expect clean verdicts; that's the detector
abstaining, and it's the honest answer.

**4. Grade every future session automatically:**

```bash
arl init --hooks     # installs a Claude Code Stop hook (idempotent merge)
```

When a receipt's fix actually helps you, say so — it's the one metric this
project measures itself by:

```bash
arl mark-applied <run-id>
```

## How it works (30 seconds)

```
 your agent's own session logs (already on disk — ARL changes nothing)
   ~/.claude/projects/**/*.jsonl · ~/.codex/sessions/** · SDK trace exports
        │
        ▼
 ┌──────────────────────────────────────────────────────┐
 │  ARL  (runs locally — nothing leaves your machine)   │
 │  adapters (hostile-input parsing, size/depth bounds) │
 │     → detectors: retry_loop · artifact_failure       │
 │     → graded receipt: proof level L0–L6 · fix        │
 │       direction · limits (what is NOT proven) ·      │
 │       coverage (what was and wasn't checked)         │
 └──────────────────────────────────────────────────────┘
        │
        ▼
 exit code (0/3/1) · JSON (arl.verdict/v1) · local SQLite ledger · HTML report
```

Receipts are advisory — ARL never applies a patch. Proof levels are graded by
static inspection of the artifact, never by the model's say-so: L2 means *the
fix mechanically removes the deterministic failure path, verifiable without a
re-run*.

## Gate your loop on it

Autonomous loops today exit on tests-pass, string-matching, or the agent's own
say-so. `arl verdict` is an independent, graded exit:

| Exit | Meaning |
|---|---|
| `0` | clean — no structural failure detected (the honest negative) |
| `3` | receipts fired — attention |
| `1` | unreadable input — **fails closed**: an unparseable run is never silently clean |

```bash
# a loop that stops on a dirty run
while :; do
  run-my-agent
  arl verdict --latest || break      # exit 3 or 1 stops the loop
done
```

```yaml
# a CI step — the receipt becomes job evidence
- run: arl verdict path/to/session.jsonl --json > arl-verdict.json
  continue-on-error: true
- uses: actions/upload-artifact@v4
  with: { name: arl-verdict, path: arl-verdict.json }
```

> Claude Code Stop hooks treat only exit code 2 as blocking. To make a fired
> receipt block the session:
> `arl verdict --latest-claude --json >> .arl/verdicts.jsonl || exit 2`

`--json` emits a stable schema (`arl.verdict/v1`) carrying the run id, detector
version, receipts, and an explicit checked/not-checked coverage block.

## Honest scope

This section is the product. Read it before trusting any verdict — including ours.

- **`clean` never means "verified correct."** It means: none of the checked
  failure classes fired. The not-checked list prints next to every verdict.
- **Two detector classes ship today:** `retry_loop` (graded L0–L2) and
  `artifact_failure` (success claims with deleted tests or zero mutating calls;
  R1/R2). Interactive, well-attended sessions mostly grade clean — the target
  population is unattended runs.
- **Reads today:** Claude Code session logs (incl. subagents) · Codex CLI
  rollouts · OpenAI Agents SDK recorded traces · ARL's own neutral JSON. A
  chat-only session errors honestly ("no run to record") instead of grading
  something it can't see.
- **Deliberately absent:** hosted anything, auth, telemetry of any kind,
  auto-update (those need network calls; ARL has none, structurally),
  autonomous patch application.
- All input is treated as hostile: bounded parsing, typed errors, nothing
  evaluated.

## Reporting issues & sharing receipts

ARL sends nothing home, so the only way we learn it misfired — or fired well —
is if you tell us. Everything ARL emits is content-free by construction
(bounded labels, booleans, hashes, counts — never your prompts, code, or
command text), so verdict/sweep JSON is safe to paste. The
[issue templates](.github/ISSUE_TEMPLATE) pre-fill what we need.

What we most want to hear, in order: a receipt that was **wrong** (false
accusation is the worst bug this tool can have) · a failure you know happened
that graded **clean** · a fix you actually **applied**.

Never paste raw session `.jsonl` files — they contain your real prompts. For
parser bugs, `arl export --run <id> --out trace.json` produces the content-free
neutral form.

## Paid pilots & audits

The tool is free. If you want me in the loop, that's the paid part: I run
**design-partner pilots** ($500–2,000/mo — weekly sweeps of your team's agent
sessions, graded receipts of what your agents actually did versus what they
claimed, detectors built from your failure fixtures) and **fixed-price agent
audits**. Everything stays local to your machines — I never need your session
content.

Email **kibahung19@gmail.com**, or open an issue titled `pilot`.

## License

[FSL-1.1-ALv2](LICENSE) — fair source. Free for everyone, individuals and
companies, including all internal commercial use. The one restriction: don't
offer ARL itself as a competing commercial product or service. Every release
converts to plain Apache-2.0 two years after it ships, irrevocably. Same
license family as Sentry, Codecov, and GitButler ([fsl.software](https://fsl.software/)).

The zero-egress claim is enforced by a build-failing test, not a promise —
[audit it](tests/test_egress_guards.py).
