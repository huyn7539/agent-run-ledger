"""Git-history rework miner — v0 spike (operator directive 2026-06-11).

For projects whose transcripts the 30-day retention destroyed (cipher,
prediction-market, Antelier, LegacyBlocks: 0 CC files in the snapshot), the
repo history is the ONLY surviving outcome evidence. This mines every local
repo for rework signatures:

  R1 reverts              - explicit revert commits
  R2 fix-chains           - >=3 fix-prefixed commits touching one file in 14d
  R3 churn hotspots       - file committed >=5 times within any 14-day window
  R4 rework vocabulary    - subjects admitting rework (redo/actually/wrong/
                            regression/properly/broken/again)
  R5 fix-after-feat       - feat commit, then fix touching same file <=48h
                            (the repo-side "done wasn't done" signature)
  R6 unmerged stale heads - local branches never merged, last commit >14d old

Spike mode (Rule 11 §1a). Output: per-repo evidence for the operator audit and
for date-joining against the cross-session transcript chains.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

PROJ = Path.home() / "proj"
EXTRA = [Path.home() / "Akashic", Path.home() / "dotfiles"]
SKIP_REPOS = {"headroom"}  # competitor clone, not operator work
FIX_RE = re.compile(r"^(fix|bugfix|hotfix)\b|^\w+\(.*\)?:?\s*fix\b|\bfix(es|ed)?\b",
                    re.IGNORECASE)
FEAT_RE = re.compile(r"^feat\b|^\w+\(.*\)?:?\s*feat\b|\badd(s|ed)?\b", re.IGNORECASE)
REWORK_RE = re.compile(r"\b(rework|redo|redone|again|actually|properly|wrong|broken|"
                       r"regression|undo|revert|mistake|incorrect|oops|really fix|"
                       r"correct(ly|ed)?)\b", re.IGNORECASE)
NOISE_FILES = re.compile(r"(^|/)(_index\.md|log\.md|MEMORY\.md|package-lock\.json|"
                         r"uv\.lock|Cargo\.lock)$", re.IGNORECASE)


def _git(repo: Path, *args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           text=True, timeout=120, encoding="utf-8", errors="replace")
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def commits_with_files(repo: Path) -> list[dict]:
    raw = _git(repo, "log", "--all", "--no-merges", "--date=iso-strict",
               "--pretty=format:@@%h|%ad|%s", "--name-only")
    out, cur = [], None
    for line in raw.splitlines():
        if line.startswith("@@"):
            h, d, s = line[2:].split("|", 2)
            try:
                dt = datetime.fromisoformat(d)
            except ValueError:
                continue
            cur = {"h": h, "dt": dt, "s": s, "files": []}
            out.append(cur)
        elif line.strip() and cur is not None and not NOISE_FILES.search(line):
            cur["files"].append(line.strip())
    return out


def mine(repo: Path) -> dict | None:
    commits = commits_with_files(repo)
    if len(commits) < 3:
        return None
    commits.sort(key=lambda c: c["dt"])
    res = {"repo": repo.name, "n_commits": len(commits),
           "first": commits[0]["dt"].date().isoformat(),
           "last": commits[-1]["dt"].date().isoformat(),
           "reverts": [], "fix_chains": [], "churn": [], "rework_msgs": [],
           "fix_after_feat": [], "stale_branches": []}

    for c in commits:
        if c["s"].lower().startswith("revert"):
            res["reverts"].append({"h": c["h"], "date": c["dt"].date().isoformat(),
                                   "msg": c["s"][:100]})
        elif REWORK_RE.search(c["s"]) and not FIX_RE.match(c["s"]):
            res["rework_msgs"].append({"h": c["h"], "date": c["dt"].date().isoformat(),
                                       "msg": c["s"][:100]})

    touches = defaultdict(list)  # file -> [(dt, h, is_fix, subject)]
    for c in commits:
        isfix = bool(FIX_RE.search(c["s"]))
        for f in c["files"]:
            touches[f].append((c["dt"], c["h"], isfix, c["s"]))

    for f, ts in touches.items():
        fixes = [t for t in ts if t[2]]
        for i in range(len(fixes)):
            window = [t for t in fixes if timedelta(0) <= t[0] - fixes[i][0]
                      <= timedelta(days=14)]
            if len(window) >= 3:
                res["fix_chains"].append({
                    "file": f, "n_fixes_14d": len(window),
                    "from": window[0][0].date().isoformat(),
                    "to": window[-1][0].date().isoformat(),
                    "examples": [w[3][:80] for w in window[:3]]})
                break
        for i in range(len(ts)):
            window = [t for t in ts if timedelta(0) <= t[0] - ts[i][0]
                      <= timedelta(days=14)]
            if len(window) >= 5:
                res["churn"].append({
                    "file": f, "touches_14d": len(window),
                    "from": window[0][0].date().isoformat(),
                    "to": window[-1][0].date().isoformat()})
                break

    by_file_feat = defaultdict(list)
    for c in commits:
        if FEAT_RE.search(c["s"]):
            for f in c["files"]:
                by_file_feat[f].append(c)
    for c in commits:
        if not FIX_RE.search(c["s"]):
            continue
        for f in c["files"]:
            for feat in by_file_feat.get(f, []):
                gap = (c["dt"] - feat["dt"]).total_seconds()
                if 0 < gap <= 48 * 3600 and feat["h"] != c["h"]:
                    res["fix_after_feat"].append({
                        "file": f, "feat": feat["s"][:70], "fix": c["s"][:70],
                        "gap_h": round(gap / 3600, 1),
                        "date": c["dt"].date().isoformat()})
                    break
            else:
                continue
            break

    merged = set(_git(repo, "branch", "--merged").split())
    for line in _git(repo, "for-each-ref", "refs/heads",
                     "--format=%(refname:short)|%(committerdate:iso-strict)").splitlines():
        if "|" not in line:
            continue
        name, d = line.split("|", 1)
        if name in merged or name in ("master", "main"):
            continue
        try:
            dt = datetime.fromisoformat(d.strip())
        except ValueError:
            continue
        if (datetime.now(dt.tzinfo) - dt).days > 14:
            res["stale_branches"].append({"branch": name,
                                          "last": dt.date().isoformat()})

    res["totals"] = {k: len(res[k]) for k in
                     ("reverts", "fix_chains", "churn", "rework_msgs",
                      "fix_after_feat", "stale_branches")}
    return res


def main(out_path: str) -> None:
    repos = [d for d in sorted(PROJ.iterdir())
             if (d / ".git").exists() and d.name not in SKIP_REPOS] + \
            [d for d in EXTRA if (d / ".git").exists()]
    results = []
    for r in repos:
        m = mine(r)
        if m:
            results.append(m)
            t = m["totals"]
            print(f"{m['repo']:32s} commits={m['n_commits']:4d} "
                  f"reverts={t['reverts']:2d} fixchains={t['fix_chains']:3d} "
                  f"churn={t['churn']:3d} rework={t['rework_msgs']:3d} "
                  f"fixafterfeat={t['fix_after_feat']:3d} stale={t['stale_branches']}")
    Path(out_path).write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
