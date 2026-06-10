"""ARL demo helper — zero path-typing.

The CLI is correct but unforgiving for a demo: you have to find a buried Codex
rollout file (UUID names under dated folders) and type its path with exact
quoting (a space in "Hung Huynh" breaks an unquoted path). This script removes
that whole class of friction:

    uv run --extra openai python demo.py

It lists your recent Codex sessions as a NUMBERED menu, you type a number, and
it imports that run + writes + opens the HTML report. No path typing, ever.

It is a thin convenience wrapper over the existing `import` + `report` commands
(same neutral schema, same detector, same storage) — NOT auto-capture, NOT a new
ingestion path. Re-running is safe (import is idempotent per Rule 5).
"""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path

from agent_run_ledger.adapters.codex import bundle_from_rollout, load_codex_rollout
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.report import write_report
from agent_run_ledger.core.storage import (
    RunAlreadyRecorded,
    load_bundle,
    save_bundle,
)

REPO = Path(__file__).resolve().parent
DB = Path(os.environ.get("ARL_DB", REPO / ".arl" / "ledger.sqlite"))
SESSIONS = Path.home() / ".codex" / "sessions"
N_RECENT = 15


def recent_codex_rollouts(limit: int = N_RECENT) -> list[Path]:
    files = sorted(SESSIONS.glob("**/rollout-*.jsonl"), key=lambda p: p.name, reverse=True)
    return files[:limit]


def main() -> int:
    rollouts = recent_codex_rollouts()
    if not rollouts:
        print(f"No Codex sessions found under {SESSIONS}")
        print("Run a Codex CLI session first, or use:  uv run --extra openai arl run-demo --variant retry-loop")
        return 1

    print("\nYour recent Codex runs (newest first):\n")
    for i, f in enumerate(rollouts, 1):
        # rollout-2026-06-08T08-23-07-<uuid>.jsonl -> a human date/time
        stamp = f.name.replace("rollout-", "").split("-019", 1)[0].replace("T", "  ")
        print(f"  [{i:>2}]  {stamp}")
    print()

    choice = input(f"Pick a run to analyze [1-{len(rollouts)}], or Enter for #1: ").strip() or "1"
    if not choice.isdigit() or not (1 <= int(choice) <= len(rollouts)):
        print(f"'{choice}' isn't a number in range. Nothing done.")
        return 1
    path = rollouts[int(choice) - 1]

    print(f"\nReading: {path.name}")
    bundle = bundle_from_rollout(load_codex_rollout(path))
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    try:
        run_id = save_bundle(DB, bundle)
        print(f"Imported: {run_id}")
    except RunAlreadyRecorded:
        run_id = bundle.run.id
        print(f"Already imported: {run_id}")

    bundle = load_bundle(DB, run_id)
    out = REPO / ".arl" / "reports" / f"{run_id}.html"
    write_report(bundle, out)

    fired = bool(getattr(bundle, "prescriptions", None))
    print(f"\nReport: {out}")
    if fired:
        print("ARL found a structural failure and emitted a repair receipt. Opening it.")
    else:
        print("ARL found NO structural failure on this run - that's the honest 'clean' result.")
        print("(Most well-run sessions are clean. To show what a FIRED receipt looks like, run:")
        print("   uv run --extra openai arl run-demo --variant retry-loop")
        print("   uv run --extra openai arl report --run run_retry_loop  )")
    webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
