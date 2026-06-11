from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from agent_run_ledger import __version__
from agent_run_ledger.adapters.claude_code import (
    ClaudeCodeSessionError,
    bundle_from_session,
    find_recent_sessions,
    load_claude_session,
    looks_like_claude_session_file,
)
from agent_run_ledger.adapters.codex import (
    CodexRolloutError,
    bundle_from_rollout,
    find_recent_rollouts,
    load_codex_rollout,
    looks_like_jsonl,
)
from agent_run_ledger.adapters.openai import NoSpansCapturedError, bundle_from_recorded_trace
from agent_run_ledger.core.compare import compare_bundles
from agent_run_ledger.core.cost import cost_display
from agent_run_ledger.core.demo import load_demo_bundle
from agent_run_ledger.core.io import TraceParseError, load_json_object, write_trace
from agent_run_ledger.core.models import TraceBundle, TraceValidationError
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.receipt import PROOF_LEVELS, build_receipts
from agent_run_ledger.core.report import render_comparison, write_report
from agent_run_ledger.core.storage import (
    RunAlreadyRecorded,
    cloud_sync_warning,
    init_db,
    list_runs,
    load_bundle,
    save_bundle,
)

app = typer.Typer(
    help=(
        "Agent Run Ledger — a local-first, honest, graded verdict layer for AI "
        "coding-agent runs. Everything stays on your machine.\n\n"
        "New here? Start with these three:\n"
        "  arl selftest                 see a real receipt fire (proves the alarm works)\n"
        "  arl verdict --latest-claude  grade your newest Claude Code session\n"
        "  arl sweep ~/.claude/projects scan your whole session history\n\n"
        "verdict exit codes: 0 = clean (for the checked classes) · 3 = a repair "
        "receipt fired · 1 = unreadable (fails closed). 'clean' never means "
        "'verified correct'."
    ),
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        # Plain print, not the rich Console: must be safe before _safe_console()
        # runs and trivially parseable in a bug report.
        print(f"agent-run-ledger {__version__}")
        raise typer.Exit(0)


@app.callback()
def _app_options(
    version: bool = typer.Option(
        False,
        "--version",
        help="Print the ARL version and exit (include this in bug reports).",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    pass


def _safe_console() -> Console:
    """A Console that can NEVER crash human output on a legacy console encoding.

    P0 (full-suite audit 2026-06-11): on a default Windows console (cp1252/cp437,
    PYTHONUTF8 unset — cmd.exe, most PowerShell, windows-latest CI) a single
    non-encodable glyph in a status line raised UnicodeEncodeError, dumped a
    traceback, and exited 1 — and it fired on the CLEAN verdict path, the product's
    most-common outcome, mis-reported to a gating loop as 'unreadable'. The test
    suite never saw it because CliRunner captures through a UTF-8 buffer. Fix at the
    source: reconfigure stdout to UTF-8 with errors='replace' where supported so no
    human glyph can crash AND mojibake (the `?`/box-char garbling) stops too. If the
    stream can't be reconfigured (already wrapped, e.g. under CliRunner), fall back
    to a default Console — that path is UTF-8 anyway."""
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    return Console()


console = _safe_console()
T = TypeVar("T")


def default_db() -> Path:
    return Path(os.environ.get("ARL_DB", ".arl/ledger.sqlite"))


def _warn_cloud_sync(db: Path) -> None:
    """L11: print a one-time warning if the ledger lives in a cloud-sync dir."""
    warning = cloud_sync_warning(db)
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")


# The exact Stop-hook line from the README recipe — one source of truth so the
# docs, the installer, and the tests can never drift apart.
HOOK_COMMAND = "arl verdict --latest-claude --json >> .arl/verdicts.jsonl"


def _install_stop_hook(target_dir: Path) -> str:
    """Merge the ARL Stop hook into <dir>/.claude/settings.json.

    Non-destructive: existing keys and existing hooks survive. Idempotent
    (Rule 5): if the hook command is already present, NOTHING is written —
    repeat calls are byte-level no-ops. Fail closed: unparseable settings.json
    raises (the caller maps it to exit 1) and the file is never touched.
    Returns "installed" or "already".
    """
    settings_path = target_dir / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"existing {settings_path} is not valid JSON ({exc}); fix it by hand — "
                "arl will not overwrite a file it cannot parse"
            ) from exc
        if not isinstance(settings, dict):
            raise ValueError(f"existing {settings_path} is not a JSON object; not touching it")
    else:
        settings = {}
    stop_entries = settings.setdefault("hooks", {}).setdefault("Stop", [])
    existing = [
        h.get("command")
        for entry in stop_entries
        if isinstance(entry, dict)
        for h in entry.get("hooks", [])
        if isinstance(h, dict)
    ]
    if HOOK_COMMAND in existing:
        return "already"
    stop_entries.append({"hooks": [{"type": "command", "command": HOOK_COMMAND}]})
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return "installed"


@app.command("init")
def init(
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
    hooks: bool = typer.Option(
        False,
        "--hooks",
        help="Also install the Claude Code Stop hook (project-local .claude/settings.json) "
        "so every finished session gets a verdict automatically.",
    ),
    target_dir: Path = typer.Option(
        Path("."), "--dir", help="Project directory for --hooks (default: current)."
    ),
) -> None:
    """Create the local ledger database (optional — verdict/sweep create it on demand)."""
    _warn_cloud_sync(db)
    init_db(db)
    console.print(f"initialized ledger: {db}")
    if hooks:
        try:
            result = _install_stop_hook(target_dir)
        except ValueError as exc:
            console.print(f"error: {exc}")
            raise typer.Exit(1) from exc
        if result == "already":
            console.print("Stop hook already installed — nothing changed.")
        else:
            console.print(
                f"Stop hook installed in {target_dir / '.claude' / 'settings.json'}: "
                f"every finished Claude Code session in this project now appends a "
                f"verdict to .arl/verdicts.jsonl"
            )


@app.command("run-demo")
def run_demo(
    variant: str = typer.Option("retry-loop", help="Demo variant: retry-loop or clean."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    """Store a built-in demo run (retry-loop or clean) to explore report/compare."""
    bundle = load_demo_bundle(variant)
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    run_id = save_bundle(db, bundle)
    console.print(f"stored demo run: {run_id}")


def _load_any_trace(path: Path) -> TraceBundle:
    """Route an import by FORMAT (provider-neutral detection):

    * a line-delimited JSON log whose lines carry the Claude Code session shape
      (top-level ``sessionId`` + ``uuid``) -> the Claude Code adapter;
    * any other line-delimited JSON log -> the Codex rollout adapter;
    * a single JSON object carrying ``trace`` + ``spans`` (and no ``run``) -> a
      recorded OpenAI-SDK trace export -> the OpenAI adapter's
      ``bundle_from_recorded_trace`` (pure parsing; no SDK dependency);
    * any other single JSON object -> the neutral TraceBundle path.

    The detection names no provider field beyond the adapters' own probes — only
    file SHAPE — so the core stays provider-neutral; provider-specific parsing
    lives entirely in the adapters. All single-object reads share
    ``load_json_object``'s defensive size/depth/encoding bounds."""
    if looks_like_jsonl(path):
        if looks_like_claude_session_file(path):
            return bundle_from_session(load_claude_session(path))
        return bundle_from_rollout(load_codex_rollout(path))
    data = load_json_object(path)
    if "run" not in data and "trace" in data and "spans" in data:
        return bundle_from_recorded_trace(data)
    try:
        return TraceBundle.from_dict(data)
    except TraceValidationError as exc:
        raise TraceParseError(f"invalid trace bundle: {exc}") from exc


@app.command("import")
def import_trace(
    path: Path = typer.Argument(..., help="Trace file: a single-object trace JSON, or a Codex .jsonl rollout log."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    """Record a run in the local ledger (for report/compare/export). For a quick
    pass/fail use `arl verdict` instead."""
    bundle = _friendly_or_exit(lambda: _load_any_trace(path))
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    # Rule 5: importing the same run twice is safe and quiet. The fact tables are
    # append-only (L2), so a re-import is a no-op, not a crash — print the existing
    # run id and the next command instead of a stack trace.
    try:
        run_id = save_bundle(db, bundle)
    except RunAlreadyRecorded:
        run_id = bundle.run.id
        console.print(f"already imported: {run_id}")
        console.print(f"  view it:  arl report --run {run_id}")
        return
    console.print(f"imported run: {run_id}")


@app.command("export")
def export_trace(
    run: str = typer.Option(..., "--run", help="Run id."),
    out: Path = typer.Option(..., "--out", help="Output JSON path."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
    raw_local: bool = typer.Option(
        False,
        "--raw-local",
        help=(
            "Keep raw-content metadata values and patch bodies (full-fidelity "
            "LOCAL form - do not share). Default export is the share-safe "
            "scrubbed form (Task 46)."
        ),
    ),
) -> None:
    """Export a recorded run to neutral trace JSON (share-safe scrubbed by default)."""
    bundle = _load_bundle_or_exit(db, run)
    _friendly_or_exit(lambda: write_trace(bundle, out, raw_local=raw_local))
    console.print(f"wrote trace: {out}")


@app.command("serve")
def serve_cmd(
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
    port: int = typer.Option(0, "--port", help="Port (default 0 = ephemeral, printed)."),
) -> None:
    """Read-only localhost dashboard over the ledger (binds 127.0.0.1 only)."""
    from agent_run_ledger.core.serve import make_server

    server = make_server(db, port=port)
    bound = server.server_address[1]
    # ASCII-only output (cp1252 console class): plain prints, no glyphs.
    print(f"arl serve: http://127.0.0.1:{bound}/")
    print("read-only dashboard; Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("arl serve: stopped")


@app.command("propose")
def propose_cmd(
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    """Mine the ledger for templated CLAUDE.md correction candidates (Task 60).

    Templates only, no model calls; a tool name failing the closed slot charset
    is an ABSTENTION, never a proposal. Apply with: arl apply <proposal_id>."""
    from agent_run_ledger.core.propose import MIN_RECEIPTS, mine_proposals

    proposals, abstentions = mine_proposals(db)
    payload = {
        "proposals": [p.display() for p in proposals],
        "abstentions": abstentions,
        "note": (
            f"templated proposals from repeated retry_loop receipts (>= {MIN_RECEIPTS} "
            "runs per tool); ids re-derive deterministically from the ledger facts"
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.command("apply")
def apply_cmd(
    proposal_id: str = typer.Argument(..., help="A proposal id from `arl propose`."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
    claudemd: Path = typer.Option(Path("CLAUDE.md"), "--claudemd", help="Target CLAUDE.md."),
    root: Path = typer.Option(Path("."), "--root", help="Project root the target must stay inside."),
    create: bool = typer.Option(False, "--create", help="Create CLAUDE.md if absent."),
    auto: bool = typer.Option(
        False,
        "--auto",
        help="Only proceed if this class EARNED autonomy from this ledger's own history.",
    ),
) -> None:
    """Apply a mined proposal into the ARL-managed CLAUDE.md block, recording a
    pre-registered experiment (metric, window, exact revert) — Task 60 P2."""
    from agent_run_ledger.core import claudemd as blockmod
    from agent_run_ledger.core import experiment as exp
    from agent_run_ledger.core import propose as proposemod
    from agent_run_ledger.core.models import utc_now_iso
    from agent_run_ledger.core.storage import list_experiments, save_experiment

    p = proposemod.find_proposal(db, proposal_id)
    if p is None:
        print("proposal not found in this ledger (run `arl propose`; ids re-derive from facts)")
        raise typer.Exit(2)
    # Registry check BEFORE any mutation (Codex P2 review F4): the registry is
    # first-write-wins, so a second apply of the same experiment would mutate
    # CLAUDE.md without a tracking row — refuse up front, write nothing.
    experiment_id = "exp-" + p.proposal_id.removeprefix("sha256:")[:16]
    existing = [e for e in list_experiments(db) if e["experiment_id"] == experiment_id]
    if existing:
        print(
            f"experiment {experiment_id} already exists with status "
            f"{existing[0]['status']!r}; refusing to re-apply (an apply is a fact "
            "about the past — new evidence mints a new proposal id)"
        )
        raise typer.Exit(3)
    if auto:
        # Class-matched history only (Codex P2 review F6): rows of any other
        # class — including crafted ones — never count toward THIS class's
        # autonomy earn-out.
        kept = len(
            [e for e in list_experiments(db, "kept") if e["proposal_class"] == p.proposal_class]
        )
        reverted = len(
            [
                e
                for e in list_experiments(db, "reverted")
                if e["proposal_class"] == p.proposal_class
            ]
        )
        if not exp.auto_earned(kept, reverted):
            print(
                f"--auto NOT earned for class {p.proposal_class}: kept={kept} "
                f"reverted={reverted}; needs P(precision > 4/5) >= 95/100 from THIS "
                "ledger's own kept/reverted history. Autonomy is earned, not granted."
            )
            raise typer.Exit(3)
    # baseline (control arm) is captured BEFORE the mutation: the last 50 runs.
    recent = [r.id for r in list_runs(db)[:50]]
    n0, k0 = proposemod.tool_failure_counts(db, p.tool, recent)
    try:
        result = blockmod.apply_line(claudemd, root, p.line, create=create)
    except blockmod.BlockError as exc:
        print(f"propose-only (fail-closed, nothing written): {exc}")
        raise typer.Exit(3) from None
    saved = save_experiment(
        db,
        {
            "experiment_id": experiment_id,
            "proposal_id": p.proposal_id,
            "proposal_class": p.proposal_class,
            "tool": p.tool,
            "claudemd_path": str(claudemd),
            "line": p.line,
            "before_block": result.before_block,
            "after_block": result.after_block,
            "assignment_basis": (
                "observational-before-after (a CLAUDE.md line applies to every "
                "run; per-run interleaving is impossible for this lane)"
            ),
            "mde": str(exp.DEFAULT_MDE),
            "eps_harm": str(exp.DEFAULT_EPS_HARM),
            "min_n": exp.DEFAULT_MIN_N,
            "baseline_n": n0,
            "baseline_k": k0,
            "applied_at": utc_now_iso(),
            "status": "applied",
        },
    )
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "proposal_id": p.proposal_id,
                "applied_line": p.line,
                "changed": result.changed,
                "registry": saved,
                "pre_registered_metric": (
                    f"retry_loop receipts for tool={p.tool} per run, "
                    "before vs after apply"
                ),
                "baseline": {"n0": n0, "k0": k0},
                "decision_rule": {
                    "keep": "P(improvement) >= 95/100 and E[delta] >= MDE, n >= min_n per arm",
                    "revert": (
                        "P(harm) >= 70/100 and E[harm] >= eps_harm (the gate math "
                        "supports instant guardrail revert, but NO independent "
                        "guardrail metric is wired in this lane version - Task 61)"
                    ),
                    "mde": str(exp.DEFAULT_MDE),
                    "eps_harm": str(exp.DEFAULT_EPS_HARM),
                    "min_n": exp.DEFAULT_MIN_N,
                },
                "revert_path": "CAS on the recorded post-apply block (arl review-applied)",
                "limits": [
                    "observational - regression to the mean not controlled "
                    "(no per-run interleaving on a CLAUDE.md line)",
                    "Bayesian decision rule with a fixed Beta(1,1) prior",
                    "no independent guardrail metric wired in this lane version - "
                    "the harm posterior on the targeted class is the only automatic "
                    "revert trigger (Task 61)",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("review-applied")
def review_applied_cmd(
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
    claudemd: Path = typer.Option(Path("CLAUDE.md"), "--claudemd", help="Target CLAUDE.md."),
    root: Path = typer.Option(Path("."), "--root", help="Project root."),
) -> None:
    """Measure every applied experiment against its pre-registered metric and
    route it: KEEP / AUTO-REVERT (CAS) / CONTINUE — Task 60 P3."""
    from fractions import Fraction

    from agent_run_ledger.core import claudemd as blockmod
    from agent_run_ledger.core import experiment as exp
    from agent_run_ledger.core import propose as proposemod
    from agent_run_ledger.core.storage import list_experiments, set_experiment_status

    reviews = []
    for e in list_experiments(db, "applied"):
        # Cohort formation is fail-closed (Codex P2 review F7): only the pinned
        # UTC timestamp shape participates — for that exact shape lexicographic
        # order IS chronological order; crafted/imported shapes are excluded.
        if not exp.pinned_utc_ts(e["applied_at"]):
            set_experiment_status(db, e["experiment_id"], "review")
            reviews.append(
                {
                    "experiment_id": e["experiment_id"],
                    "action": "review",
                    "detail": (
                        "applied_at is not the pinned UTC shape; a treatment "
                        "cohort cannot be formed - routed to review"
                    ),
                }
            )
            continue
        runs_after = [
            r.id
            for r in list_runs(db)
            if exp.pinned_utc_ts(r.started_at) and r.started_at > e["applied_at"]
        ]
        n1, k1 = proposemod.tool_failure_counts(db, e["tool"], runs_after)
        decision = exp.decide(
            e["baseline_n"],
            e["baseline_k"],
            n1,
            k1,
            mde=Fraction(e["mde"]),
            eps_harm=Fraction(e["eps_harm"]),
            min_n=e["min_n"],
        )
        entry: dict = {
            "experiment_id": e["experiment_id"],
            "tool": e["tool"],
            "control": {"n0": e["baseline_n"], "k0": e["baseline_k"]},
            "treatment": {"n1": n1, "k1": k1},
            "ci95_display": list(exp.ci95_display(e["baseline_n"], e["baseline_k"], n1, k1)),
            "limits": [
                "observational - regression to the mean not controlled",
                "Bayesian decision rule with a fixed Beta(1,1) prior",
                "no independent guardrail metric wired in this lane version - "
                "the harm posterior on the targeted class is the only automatic "
                "revert trigger (Task 61)",
                "treatment cohort = runs whose started_at matches the pinned UTC "
                "shape strictly after applied_at (other shapes excluded, fail-closed)",
            ],
        }
        entry.update(decision.display())
        if decision.decision == "KEEP":
            set_experiment_status(db, e["experiment_id"], "kept")
            entry["action"] = "kept"
        elif decision.decision == "REVERT":
            r = blockmod.revert_block(
                Path(e["claudemd_path"]) if claudemd == Path("CLAUDE.md") else claudemd,
                root,
                e["after_block"],
                e["before_block"],
            )
            new_status = "reverted" if r.status == "reverted" else "review"
            set_experiment_status(db, e["experiment_id"], new_status)
            entry["action"] = new_status
            entry["revert_detail"] = r.detail
        else:
            entry["action"] = "continue (stays applied; posterior shown)"
        reviews.append(entry)
    print(json.dumps({"reviews": reviews}, indent=2, sort_keys=True))


@app.command("auto-status")
def auto_status_cmd(
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    """Show whether the (single) proposal class has EARNED --auto from this
    ledger's own kept/reverted history — Task 60 P4."""
    from agent_run_ledger.core import experiment as exp
    from agent_run_ledger.core.propose import PROPOSAL_CLASS
    from agent_run_ledger.core.storage import list_experiments

    # Class-matched history only (Codex P2 review F6) — same filter as
    # `arl apply --auto`.
    kept = len(
        [e for e in list_experiments(db, "kept") if e["proposal_class"] == PROPOSAL_CLASS]
    )
    reverted = len(
        [e for e in list_experiments(db, "reverted") if e["proposal_class"] == PROPOSAL_CLASS]
    )
    print(
        json.dumps(
            {
                "class": PROPOSAL_CLASS,
                "kept": kept,
                "reverted": reverted,
                "auto_earned": exp.auto_earned(kept, reverted),
                "bar": "P(precision > 4/5) >= 95/100 over Beta(kept+1, reverted+1)",
                "note": (
                    "autonomy is earned per class from THIS ledger's own outcomes; "
                    "the default is and stays propose-only"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("report")
def report(
    run: str = typer.Option(..., "--run", help="Run id."),
    out: Path | None = typer.Option(None, "--out", help="Output HTML path."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    """Write a static local HTML report for a recorded run."""
    bundle = _load_bundle_or_exit(db, run)
    output = out or Path(".arl") / "reports" / f"{run}.html"
    _friendly_or_exit(lambda: write_report(bundle, output))
    console.print(f"wrote report: {output}")


@app.command("compare")
def compare(
    left: str = typer.Option(..., "--left", help="Left/baseline run id."),
    right: str = typer.Option(..., "--right", help="Right/candidate run id."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    """Compare two recorded runs (cost/tokens/outcome deltas)."""
    comparison = compare_bundles(_load_bundle_or_exit(db, left), _load_bundle_or_exit(db, right))
    console.print(render_comparison(comparison))


@app.command("list-runs")
def list_runs_cmd(
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
) -> None:
    """List the runs recorded in the local ledger."""
    rows = list_runs(db)
    table = Table(title="Agent Run Ledger")
    table.add_column("Run")
    table.add_column("Workflow")
    table.add_column("Outcome")
    table.add_column("Cost")
    table.add_column("Tokens")
    for run in rows:
        # L7/LR2: show the cost computed on read from the FACTS, never the cached
        # total_cost_usd (which a price-table change — or a $0 capture cache —
        # makes stale). Load the bundle per run; linear in run count.
        bundle = load_bundle(db, run.id)
        # Use the DISCLOSURE form (A2): an unpriced real-token run reads as
        # "unpriced (...)" here too, never a misleading bare $0 — consistent with the
        # demo + HTML report (was: cost_on_read -> a silent $0 on this surface).
        table.add_row(
            run.id,
            run.workflow,
            run.success_label,
            cost_display(bundle),
            str(run.total_input_tokens + run.total_output_tokens),
        )
    console.print(table)


@app.command("mark-applied")
def mark_applied(
    run_id: str = typer.Argument(..., help="Run id whose receipt's fix you applied."),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
    at: str = typer.Option(
        "", "--at", help="ISO timestamp of the apply (default: now, UTC)."
    ),
) -> None:
    """Record that you APPLIED a receipt's fix — the product's success metric.

    Writes an applied-event into the run's outcome slot. First write wins;
    marking the same run again changes nothing."""
    from agent_run_ledger.core.models import utc_now_iso
    from agent_run_ledger.core.storage import merge_run_outcome

    try:
        result = merge_run_outcome(db, run_id, "applied", {"at": at or utc_now_iso()})
    except KeyError:
        console.print(f"error: run not found: {run_id}")
        raise typer.Exit(1) from None
    if result == "already":
        console.print(f"{run_id} is already marked applied — nothing changed.")
    else:
        console.print(f"marked applied: {run_id}")


# The loop contract (Task 57): a hook / CI step / Ralph-style loop gates on these.
# 0 = clean, 3 = receipt(s) fired, 1 = error (fail closed: unreadable is NOT clean).
# 2 stays reserved for click/typer usage errors.
EXIT_CLEAN = 0
EXIT_RECEIPTS = 3
VERDICT_SCHEMA = "arl.verdict/v1"


def _echo_json(payload: dict) -> None:
    """Stdout JSON for hooks/loops — byte-for-byte parseable (typer.echo, not rich).

    A consumer that closes the pipe early (`| head`, a crashed hook) must not let a
    broken-pipe teardown mask the EXIT CONTRACT: the verdict already happened and
    the exit code is the product (found live 2026-06-11: truncated pipe turned a
    real exit 3 into -1). Swallow the write failure; keep the exit code."""
    try:
        typer.echo(json.dumps(payload, indent=2))
    except (BrokenPipeError, OSError):
        pass

# Coverage honesty (2026-06-10 gauntlet, convergent fix #1): a 'clean' that does
# not name its checked classes gets read as "agent output verified" — which this
# tool does NOT claim. Every verdict states what was and was not checked.
DETECTOR_COVERAGE = {
    "detector_version": "v1",
    "checked": [
        "retry_loop: autonomous same-tool same-input repeated failures "
        "(incl. cross-turn), graded L0-L2",
        "artifact_failure: success-claim vs log-evidence divergence "
        "(R1 test-deletion, R2 no-op completion), graded L0-L1",
    ],
    "not_checked": [
        "specification failure (agent built the wrong thing)",
        "wrong-but-passing patch beyond R1/R2 (e.g. assertion weakening, "
        "'tests pass' claim with no test run after the last edit)",
        "context loss / continuity",
        "cost & quota attribution",
    ],
}


@app.command("verdict")
def verdict(
    path: Path | None = typer.Argument(
        None,
        help="Trace file: single-object trace JSON or a Codex .jsonl rollout. Omit with --latest.",
    ),
    latest: bool = typer.Option(
        False, "--latest", help="Grade the newest local Codex session rollout instead of a path."
    ),
    latest_claude: bool = typer.Option(
        False, "--latest-claude", help="Grade the newest local Claude Code session instead of a path."
    ),
    sessions_root: Path | None = typer.Option(
        None, "--sessions-root", help="Codex sessions root (default: ~/.codex/sessions)."
    ),
    claude_projects_root: Path | None = typer.Option(
        None,
        "--claude-projects-root",
        help="Claude Code projects root (default: ~/.claude/projects).",
    ),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
    json_out: bool = typer.Option(
        False, "--json", help="Machine-readable arl.verdict/v1 JSON on stdout."
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Record the run in the ledger (idempotent)."
    ),
) -> None:
    """Grade a run for a loop: exit 0 = clean, 3 = receipt(s) fired, 1 = error.

    This is the verdict layer for autonomous loops: the graded repair receipt as a
    machine-consumable exit, so "the run finished" and "the run is verified clean"
    stop being the same claim. An unreadable run exits 1 — fail closed, never
    silently clean."""
    if latest and latest_claude:
        console.print("error: pass --latest OR --latest-claude, not both")
        raise typer.Exit(1)
    if latest:
        rollouts = find_recent_rollouts(sessions_root)
        if not rollouts:
            root_label = sessions_root or "~/.codex/sessions"
            console.print(f"error: no Codex session rollouts found under {root_label}")
            raise typer.Exit(1)
        path = rollouts[0]
    if latest_claude:
        sessions = find_recent_sessions(claude_projects_root)
        if not sessions:
            root_label = claude_projects_root or "~/.claude/projects"
            console.print(f"error: no Claude Code sessions found under {root_label}")
            raise typer.Exit(1)
        path = sessions[0]
    if path is None:
        console.print(
            "error: pass a trace path, or use --latest (Codex) / --latest-claude (Claude Code)"
        )
        raise typer.Exit(1)

    # First-user polish: a directory path is a common mistake; catch it before the
    # OS open (Windows raises PermissionError, POSIX IsADirectoryError — neither
    # reads cleanly). Point them at `sweep`, which is what they almost certainly want.
    if path.is_dir():
        console.print(
            f"error: {path} is a directory, not a session file — "
            "use `arl sweep <dir>` to scan a folder"
        )
        raise typer.Exit(1)
    bundle = _friendly_or_exit(lambda: _load_any_trace(path))
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    receipts = build_receipts(bundle)

    if save:
        # Idempotent by design (Rule 5): a verdict on an already-recorded run is a
        # read, not a conflict.
        try:
            save_bundle(db, bundle)
        except RunAlreadyRecorded:
            pass

    if json_out:
        max_level = (
            max((r.proof_level for r in receipts), key=PROOF_LEVELS.index) if receipts else None
        )
        payload = {
            "schema": VERDICT_SCHEMA,
            "run_id": bundle.run.id,
            "verdict": "receipts" if receipts else "clean",
            "receipt_count": len(receipts),
            "max_proof_level": max_level,
            "coverage": DETECTOR_COVERAGE,
            "receipts": [asdict(r) for r in receipts],
        }
        _echo_json(payload)
        raise typer.Exit(EXIT_RECEIPTS if receipts else EXIT_CLEAN)

    if not receipts:
        console.print(
            f"verdict: clean for the checked classes — no structural failure detected "
            f"in {bundle.run.id}"
        )
        console.print(
            "  checked: retry loops · success-claim/log divergence (detector v1)"
        )
        console.print(
            "  NOT checked: spec failures, wrong-but-green patches beyond R1/R2, "
            "context loss"
        )
        console.print(
            "  clean is NOT verified-correct. Run `arl selftest` to see a receipt fire."
        )
        raise typer.Exit(EXIT_CLEAN)

    console.print(f"verdict: {len(receipts)} repair receipt(s) fired on {bundle.run.id}")
    for r in receipts:
        console.print(
            f"  [{r.proof_level} | confidence {r.confidence}] {r.observed_failure}: "
            f"{r.repair_artifact.get('one_line_fix', '')}"
        )
    console.print("  review before applying — ARL advises, you apply:")
    console.print(f"    arl report --run {bundle.run.id}")
    raise typer.Exit(EXIT_RECEIPTS)


SWEEP_SCHEMA = "arl.sweep/v1"

# Codex P2/P3: hard ceiling on how many candidate files a sweep will enumerate
# before it stops walking a (possibly hostile/huge) directory tree. Bounds the
# DoS vector; far above any real session archive. Hitting it is reported, never
# silent.
_SWEEP_MAX_ENUM = 50_000

# The typed read errors a sweep tolerates PER FILE (the batch continues; the
# verdict path's fail-closed contract becomes per-file accounting here).
_SWEEP_READ_ERRORS = (
    OSError,
    json.JSONDecodeError,
    TraceParseError,
    TraceValidationError,
    CodexRolloutError,
    ClaudeCodeSessionError,
    NoSpansCapturedError,
)


@app.command("sweep")
def sweep(
    root: Path = typer.Argument(
        ..., help="Directory swept recursively for agent session logs (*.jsonl)."
    ),
    limit: int = typer.Option(
        200, "--limit", help="Maximum session files to grade, newest first by mtime."
    ),
    db: Path = typer.Option(default_factory=default_db, help="SQLite database path."),
    json_out: bool = typer.Option(
        False, "--json", help="Machine-readable arl.sweep/v1 JSON on stdout."
    ),
    save: bool = typer.Option(
        False,
        "--save/--no-save",
        help="Record graded runs in the ledger (default: read-only sweep).",
    ),
) -> None:
    """Batch-verdict every session log under a root (the archive sweep).

    Exit 0 = no receipts anywhere (clean for the checked classes); exit 3 = at
    least one file fired; exit 1 = total failure (root missing, nothing found,
    or every file unreadable). Read-only by default — pass --save to record.
    Per-file errors are counted and reported, never silently skipped; a
    chat-only session (no tool calls) counts as no-run, not an error."""
    if not root.is_dir():
        console.print(f"error: sweep root is not a directory: {root}")
        raise typer.Exit(1)
    # TODO(p2, Codex 2026-06-11): root.glob follows symlinks, so a symlinked .jsonl
    # under the sweep root is read via its target (possibly outside root). Local
    # read-only behavior, no egress; deferred to the parser-hardening pass. To close:
    # skip entries where p.is_symlink() or resolve-and-bound to the root.
    #
    # Codex P2/P3: bound ENUMERATION, not just the result slice. A hostile/huge tree
    # (millions of *.jsonl) must not be fully globbed+stat'd before slicing — that is
    # a DoS vector. Walk lazily and stop after _SWEEP_MAX_ENUM candidates; if we hit
    # that ceiling, SAY SO (no silent truncation — vault rule). Within the bounded
    # candidate set we still take the newest `limit` by mtime.
    enumerated: list[Path] = []
    truncated = False
    for p in root.glob("**/*.jsonl"):
        enumerated.append(p)
        if len(enumerated) >= _SWEEP_MAX_ENUM:
            truncated = True
            break
    if truncated:
        console.print(
            f"[yellow]note: sweep stopped enumerating at {_SWEEP_MAX_ENUM} files; "
            f"grading the newest {max(limit, 0)} of those. Narrow the root to cover more.[/yellow]"
        )
    candidates = sorted(enumerated, key=lambda p: p.stat().st_mtime, reverse=True)[
        : max(limit, 0)
    ]
    counts = {"clean": 0, "fired": 0, "no_run": 0, "error": 0}
    results: list[dict] = []
    for path in candidates:
        try:
            bundle = _load_any_trace(path)
        except _SWEEP_READ_ERRORS as exc:
            if "no run to record" in str(exc):
                counts["no_run"] += 1
                results.append({"path": str(path), "status": "no_run"})
            else:
                counts["error"] += 1
                results.append({"path": str(path), "status": "error", "error": str(exc)})
            continue
        bundle = bundle.with_prescriptions(analyze_bundle(bundle))
        receipts = build_receipts(bundle)
        if save:
            try:
                save_bundle(db, bundle)
            except RunAlreadyRecorded:
                pass
        if receipts:
            counts["fired"] += 1
            max_level = max((r.proof_level for r in receipts), key=PROOF_LEVELS.index)
            results.append(
                {
                    "path": str(path),
                    "status": "fired",
                    "run_id": bundle.run.id,
                    "receipt_count": len(receipts),
                    "max_proof_level": max_level,
                    "observed_failures": sorted({r.observed_failure for r in receipts}),
                }
            )
        else:
            counts["clean"] += 1
            results.append({"path": str(path), "status": "clean", "run_id": bundle.run.id})

    scanned = len(candidates)
    total_failure = scanned == 0 or counts["error"] == scanned
    exit_code = EXIT_RECEIPTS if counts["fired"] else (1 if total_failure else EXIT_CLEAN)

    if json_out:
        payload = {
            "schema": SWEEP_SCHEMA,
            "root": str(root),
            "scanned": scanned,
            "counts": counts,
            "coverage": DETECTOR_COVERAGE,
            "fired": [r for r in results if r["status"] == "fired"],
            "errors": [r for r in results if r["status"] == "error"],
        }
        _echo_json(payload)
        raise typer.Exit(exit_code)

    if scanned == 0:
        console.print(f"error: no session logs (*.jsonl) found under {root}")
        raise typer.Exit(1)
    console.print(f"sweep: {scanned} session file(s) under {root}")
    console.print(
        f"  clean={counts['clean']} fired={counts['fired']} "
        f"no-run={counts['no_run']} error={counts['error']}"
    )
    def _rel(path_text: str) -> str:
        try:
            return str(Path(path_text).relative_to(root))
        except ValueError:
            return path_text

    for r in results:
        if r["status"] == "fired":
            # soft_wrap: a path must never be hard-wrapped mid-name in a terminal.
            console.print(
                f"  FIRED {_rel(r['path'])} — {r['receipt_count']} receipt(s), "
                f"max {r['max_proof_level']}, {', '.join(r['observed_failures'])}",
                soft_wrap=True,
            )
    for r in results:
        if r["status"] == "error":
            console.print(f"  ERROR {_rel(r['path'])} — {r['error']}", soft_wrap=True)
    if total_failure:
        console.print("error: every scanned file was unreadable — nothing was graded")
        raise typer.Exit(1)
    console.print(
        "  clean = clean for the checked classes only (see `arl verdict --json` coverage)"
    )
    raise typer.Exit(exit_code)


@app.command("selftest")
def selftest() -> None:
    """Prove the alarm fires: a bundled known-bad run through the real pipeline.

    If this passes, a 'clean' verdict on your runs means the detector abstained —
    not that the plumbing is broken. Exit 0 = pass, 1 = this install is broken."""
    from agent_run_ledger.core.selftest import selftest_receipts

    try:
        receipts = selftest_receipts()
    except Exception as exc:  # a selftest must never traceback — report and fail
        console.print(f"selftest: FAIL — pipeline error: {exc}")
        raise typer.Exit(1) from exc
    if not receipts or any(r.proof_level not in PROOF_LEVELS for r in receipts):
        console.print(
            "selftest: FAIL — the bundled known-bad run did not produce a graded "
            "receipt; this install's detector pipeline is broken"
        )
        raise typer.Exit(1)
    r = receipts[0]
    console.print("selftest: running a bundled known-bad run through the real pipeline")
    console.print(
        f"  receipt fired: {r.observed_failure} at {r.proof_level} (confidence {r.confidence})"
    )
    console.print(f"  fix direction: {r.repair_artifact.get('one_line_fix', '')}")
    console.print(
        "selftest: PASS — the alarm fires; 'clean' on your runs means the detector "
        "abstained, not that it is deaf"
    )
    raise typer.Exit(0)


def _load_bundle_or_exit(db: Path, run_id: str):
    try:
        return load_bundle(db, run_id)
    except KeyError as exc:
        console.print(f"error: {exc.args[0]}")
        raise typer.Exit(1) from exc


def _friendly_or_exit(action: Callable[[], T]) -> T:
    try:
        return action()
    except FileNotFoundError as exc:
        # First-user polish: a typo'd path must read as a plain sentence, not a raw
        # platform error ("[WinError 2] The system cannot find the file specified").
        target = exc.filename or ""
        console.print(f"error: file not found: {target}".rstrip(": "))
        raise typer.Exit(1) from exc
    except IsADirectoryError as exc:
        where = f": {exc.filename}" if exc.filename else ""
        console.print(
            f"error: that is a directory, not a session file{where} "
            "(use `arl sweep <dir>` to scan a folder)"
        )
        raise typer.Exit(1) from exc
    except (
        # OSError covers PermissionError and every other OS-level read failure — a
        # directory or unreadable path must produce a typed exit-1 error, never a
        # traceback (Codex finding, 2026-06-10). FileNotFound/IsADirectory are
        # handled above with friendlier wording.
        OSError,
        json.JSONDecodeError,
        TraceParseError,
        TraceValidationError,
        CodexRolloutError,
        ClaudeCodeSessionError,
        NoSpansCapturedError,
    ) as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
