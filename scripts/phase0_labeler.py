"""Phase 0 v1.1 git-fate labeler (OUTCOME-side ONLY — Phase 0 v3 amendment #7).

Reads the human-corpus manifest, extracts each Claude session's edits, and
joins them to git history with CONTENT-ANCHORED probes (long lines from the
actual inserted text), producing audit cards with outcome evidence only.

Wall discipline: this tool is the LABELER. It reads edit payloads (outcome
material: what was written and whether it survived), end-shape facts, and git
history. It never reads or emits prompt text. Audit cards contain no
process-side content.

Sealed v1.2 rules (v0→v1 2026-06-11 fixed defects D1-D5; v1→v1.1 same day
adjudicated two-engine review: Codex findings 2/3/4/8 + fleet findings
C1-C16; v1.1→v1.2 same day: the unguarded tree-wide anchor flipped the deck
to 24/24 FINE — quoted pre-existing content was anchoring survival, so probes
are now NOVELTY-GATED. All amendments are PRE-AUDIT and PRE-EXTRACTOR —
blind-legal. Amendment record:
Akashic/06-learning/agent-run-ledger/phase0/2026-06-11-labeler-v1-amendment.md):
  REPO   : resolved PER EDIT from the edited file's own PHYSICAL path
           (symlinks/junctions resolved BEFORE the toplevel prefix strip;
           strip failure -> PATH-UNRESOLVED, never UNCOMMITTED).  [D1; fleet C1/C6/C9]
  PROBE  : up to 3 longest distinct lines (len>=24) drawn from ALL edit bodies
           to that file; for Edit-tool edits, lines also present in old_string
           are EXCLUDED (context lines are not the session's new content);
           max 8 probed files per session (first distinct).  [D4; fleet C8/C13]
  NOVELTY: a probe counts ONLY if it is absent from the whole tree at the
           session-start base commit (`git grep -F` at base) — pre-existing
           or quoted content never anchors survival OR GONE; an edit with no
           novel probe -> NO-NOVEL-CONTENT (no-label).  [fleet C8/C13]
  NO-LABEL fates (excluded from the survival denominator):
           IGNORED (check-ignore AND untracked AND no pre-horizon history),
           OUT-OF-REPO, NO-HISTORY, PATH-UNRESOLVED, NO-NOVEL-CONTENT,
           PROBE-ERROR (git subprocess failure surfaces, never silently
           reads as UNCOMMITTED).
  FATE   : any novel probe present in file at rev14 (last commit <=
           session_end+14d on HEAD history) -> SURVIVED; else any novel probe
           found ANYWHERE in the tree at rev14 (`git grep -F`, rename/move-
           proof content anchor per v3 #4 — sound only because of the novelty
           gate) -> SURVIVED; else any novel probe committed before horizon
           per `git log -S` -> GONE (rewritten/reverted); else UNCOMMITTED.
           base + rev14 commits recorded per repo on each card.  [fleet C3/C7/C10]
  MATURITY: session_end+14d > now -> too-recent stratum, NOT labeled.
  LABEL  : >=2 valid probes required. survival>=0.7 -> FINE-proposed;
           <=0.3 -> SIDEWAYS-proposed; else stratum=ambiguous, NO card
           (v3 #7 explicit no-label state — never offered for CONFIRM).  [fleet C4/C11]
  STRATA : >=2 valid probes in non-vault repos -> code-primary klass;
           else vault-secondary. Codex sessions -> codex-unprocessed stratum
           (adapter-specific extraction is a disclosed v1.1 gap; 29/378 carry
           git metadata).  [fleet C12]
  DECK   : sorted by (klass, proposal, survival, id) — interrupts and other
           distress facts NEVER order or enter the deck; they are displayed
           per card, tagged secondary.  [fleet C2/C16]
  HEADER : records manifest sha256; the labeler's own commit hash is recorded
           in the amendment file at seal time (P4 symmetric commitment).  [fleet C14]
Interrupts are counted ONLY from structured harness markers (a user entry
whose text part exactly equals a known interrupt literal) — never from raw
line scans (Codex finding 8). User-turn counts exclude tool_result-only
entries (fleet C5).
Known disclosed gaps (v1.1): Write-tool bodies may carry pre-existing content
when rewriting a file (no base-blob diff yet); Codex-session edit extraction
not implemented; submodule boundaries unhandled (no submodules in corpus).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

MAX_PROBED_FILES = 8
MAX_PROBES_PER_FILE = 3
MIN_PROBE_LEN = 24
HORIZON_DAYS = 14
EDIT_TOOLS = {"Edit": "new_string", "Write": "content", "NotebookEdit": "new_source"}
VAULT_REPO_NAMES = {"Akashic"}
INTERRUPT_MARKERS = {
    "[Request interrupted by user]",
    "[Request interrupted by user for tool use]",
}
VALID_FATES = {"SURVIVED", "GONE", "UNCOMMITTED"}


class GitProbeError(Exception):
    """A git subprocess failed to run (timeout / OS error). Distinct from a
    clean non-zero exit, which is a meaningful answer (path absent, no hits)."""


def _git(repo: str, *args: str) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True, text=True, timeout=60, encoding="utf-8",
            errors="replace",
        )
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitProbeError(f"git {args[0] if args else '?'}: {exc}") from exc


def _git_ok(repo: str, *args: str) -> bool:
    try:
        r = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True, text=True, timeout=60, encoding="utf-8",
            errors="replace",
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitProbeError(f"git {args[0] if args else '?'}: {exc}") from exc


def _probe_lines(edits: list[dict]) -> list[str]:
    """Top-k longest distinct NEW lines across all edit bodies for one file.
    Lines also present in any old_string are context, not new content (C8/C13)."""
    old_lines = {
        ln.strip()
        for e in edits
        for ln in e.get("old", "").splitlines()
        if len(ln.strip()) >= MIN_PROBE_LEN
    }
    candidates = {
        ln.strip()
        for e in edits
        for ln in e["new"].splitlines()
        if len(ln.strip()) >= MIN_PROBE_LEN and ln.strip() not in old_lines
    }
    ranked = sorted(candidates, key=lambda ln: (-len(ln), ln))
    return ranked[:MAX_PROBES_PER_FILE]


def _physical(path_str: str) -> str:
    """Resolve symlinks/junctions so the path is comparable to git's physical
    toplevel (fleet C1/C6/C9: ~/.claude/... routes into the dotfiles repo)."""
    try:
        return str(Path(path_str.replace("\\", "/")).resolve()).replace("\\", "/")
    except OSError:
        return path_str.replace("\\", "/")


def _repo_of(file_path: str) -> str:
    """Git toplevel of the repo the FILE lives in (D1: never the session cwd).
    Walks to the nearest existing ancestor so deleted files still resolve."""
    d = Path(file_path).parent
    while not d.exists():
        if d == d.parent:
            return ""
        d = d.parent
    return _git(str(d), "rev-parse", "--show-toplevel").strip()


def _is_prompt_turn(content) -> bool:
    """A user entry that is an actual typed turn, not a tool_result carrier
    (fleet C5: tool_results are type=='user' too and inflate the count)."""
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return any(isinstance(p, dict) and p.get("type") == "text" for p in content)
    return False


def parse_session(path: Path) -> dict:
    s: dict = {"id": path.stem, "cwd": "", "branch": "", "start": "", "end": "",
               "interrupts": 0, "user_turns": 0, "last_type": "", "edits": []}
    bodies: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            ts = obj.get("timestamp")
            if isinstance(ts, str):
                s["start"] = s["start"] or ts
                s["end"] = ts
            if not s["cwd"] and isinstance(obj.get("cwd"), str):
                s["cwd"] = obj["cwd"]
            if not s["branch"] and isinstance(obj.get("gitBranch"), str):
                s["branch"] = obj["gitBranch"]
            t = obj.get("type")
            if t:
                s["last_type"] = t
            if t == "user":
                content = (obj.get("message") or {}).get("content")
                if _is_prompt_turn(content):
                    s["user_turns"] += 1
                if isinstance(content, list):
                    for p in content:
                        if (isinstance(p, dict) and p.get("type") == "text"
                                and p.get("text") in INTERRUPT_MARKERS):
                            s["interrupts"] += 1
            if t == "assistant":
                for p in (obj.get("message") or {}).get("content") or []:
                    if (isinstance(p, dict) and p.get("type") == "tool_use"
                            and p.get("name") in EDIT_TOOLS):
                        inp = p.get("input") or {}
                        fp = inp.get("file_path") or inp.get("notebook_path") or ""
                        body = str(inp.get(EDIT_TOOLS[p["name"]]) or "")
                        if not fp or not body:
                            continue
                        rec = {"new": body, "old": str(inp.get("old_string") or "")}
                        # All edits to an already-tracked file accumulate (D4);
                        # the file cap applies to NEW files only.
                        if fp in bodies:
                            bodies[fp].append(rec)
                        elif len(bodies) < MAX_PROBED_FILES:
                            bodies[fp] = [rec]
    for fp, recs in bodies.items():
        probes = _probe_lines(recs)
        if probes:
            s["edits"].append({"file": fp, "probes": probes})
    return s


def _fate_edit(e: dict, start: str, horizon: str, rev_cache: dict) -> None:
    try:
        _fate_edit_inner(e, start, horizon, rev_cache)
    except GitProbeError:
        e["fate"] = "PROBE-ERROR"


def _rev_at(repo: str, when: str, rev_cache: dict) -> str:
    if (repo, when) not in rev_cache:
        rev_cache[(repo, when)] = _git(
            repo, "rev-list", "-1", f"--before={when}", "HEAD").strip()
    return rev_cache[(repo, when)]


def _fate_edit_inner(e: dict, start: str, horizon: str, rev_cache: dict) -> None:
    phys = _physical(e["file"])
    repo = _repo_of(phys)
    if not repo:
        e["fate"] = "OUT-OF-REPO"
        return
    e["repo"] = repo.rsplit("/", 1)[-1]
    root = _physical(repo)
    if not phys.lower().startswith(root.lower() + "/"):
        # Fail closed into a no-label state, never into failure evidence
        # (fleet C1/C6/C9: the v1 fall-through made SURVIVED unreachable).
        e["fate"] = "PATH-UNRESOLVED"
        return
    rel = phys[len(root) + 1:]
    tracked = _git_ok(repo, "ls-files", "--error-unmatch", "--", rel)
    if not tracked and _git_ok(repo, "check-ignore", "-q", "--", rel):
        # Tracked files are never ignored (Codex 3); a path with commit
        # history before the horizon is fate-probed even if ignored now.
        if not _git(repo, "log", "-1", f"--until={horizon}",
                    "--format=%h", "--", rel).strip():
            e["fate"] = "IGNORED"
            return
    rev14 = _rev_at(repo, horizon, rev_cache)
    if not rev14:
        e["fate"] = "NO-HISTORY"
        return
    e["rev14"] = rev14[:7]
    base = _rev_at(repo, start, rev_cache)
    # Novelty gate (C8/C13): pre-existing/quoted content never anchors. A
    # repo with no history before the session makes every probe novel.
    novel = [p for p in e["probes"]
             if not base or not _git_ok(repo, "grep", "-F", "-q", "-e", p, base)]
    if not novel:
        e["fate"] = "NO-NOVEL-CONTENT"
        return
    shown = _git(repo, "show", f"{rev14}:{rel}")
    if any(p in shown for p in novel):
        e["fate"] = "SURVIVED"
    elif any(_git_ok(repo, "grep", "-F", "-q", "-e", p, rev14) for p in novel):
        # Rename/move-proof content anchor: the NOVEL probe lives somewhere
        # in the tree at rev14 (v3 #4; vault task files move pending->done).
        e["fate"] = "SURVIVED"
    elif any(_git(repo, "log", f"--until={horizon}", "-S", p,
                  "--format=%h", "--", rel).strip() for p in novel):
        e["fate"] = "GONE"
    else:
        e["fate"] = "UNCOMMITTED"


def fate_session(s: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    try:
        end = datetime.fromisoformat(s["end"].replace("Z", "+00:00"))
    except ValueError:
        s["stratum"] = "no-timestamp"
        return s
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if end + timedelta(days=HORIZON_DAYS) > now:
        s["stratum"] = "too-recent"
        return s
    horizon = (end + timedelta(days=HORIZON_DAYS)).isoformat()
    try:
        start = datetime.fromisoformat(s["start"].replace("Z", "+00:00")).isoformat()
    except ValueError:
        start = horizon  # degenerate: no usable start; base falls at horizon
    rev_cache: dict = {}
    for e in s["edits"]:
        _fate_edit(e, start, horizon, rev_cache)
    valid = [e for e in s["edits"] if e.get("fate") in VALID_FATES]
    if len(valid) < 2:
        s["stratum"] = "too-few-probes"
        return s
    repos = [e["repo"] for e in valid]
    s["repo"] = max(sorted(set(repos)), key=repos.count)
    code_probes = [e for e in valid if e["repo"] not in VAULT_REPO_NAMES]
    s["klass"] = "code-primary" if len(code_probes) >= 2 else "vault-secondary"
    rate = sum(e["fate"] == "SURVIVED" for e in valid) / len(valid)
    s["survival"] = round(rate, 2)
    if rate >= 0.7:
        s["proposal"], s["stratum"] = "FINE", "labeled"
    elif rate <= 0.3:
        s["proposal"], s["stratum"] = "SIDEWAYS", "labeled"
    else:
        # v3 #7: ambiguous is an explicit NO-LABEL state, never a gold
        # candidate offered for CONFIRM/REJECT (fleet C4/C11).
        s["stratum"] = "ambiguous"
    return s


def _card_sort_key(c: dict):
    """Deck order: klass, proposal, survival, id. Distress facts (interrupts)
    are displayed but NEVER order the deck (fleet C2/C16)."""
    return (c["klass"] != "code-primary", c["proposal"] != "SIDEWAYS",
            c["survival"], c["id"])


def main(manifest_path: str, out_path: str) -> None:
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    cards, strata = [], {}
    for entry in manifest["human_sessions"]:
        if entry["kind"] != "claude":
            strata["codex-unprocessed"] = strata.get("codex-unprocessed", 0) + 1
            continue
        s = fate_session(parse_session(Path(entry["path"])))
        strata[s["stratum"]] = strata.get(s["stratum"], 0) + 1
        if s["stratum"] == "labeled":
            cards.append(s)
    cards.sort(key=_card_sort_key)
    out = Path(out_path)
    with out.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 0 audit cards — OUTCOME EVIDENCE ONLY (no prompt text by protocol)\n\n")
        fh.write(f"Labeler: v1.1 (sealed rules in scripts/phase0_labeler.py docstring) · "
                 f"manifest sha256 `{hashlib.sha256(manifest_bytes).hexdigest()[:16]}…`\n\n"
                 f"Strata: {json.dumps(strata, sort_keys=True)}\n\n"
                 "Operator: mark each card CONFIRM or REJECT (gut + evidence; do NOT re-read "
                 "the session). **Confirm on the git-fate evidence alone** — turn/interrupt "
                 "counts are context, never labels (v3 #5). Confirmed cards become the "
                 "sealed gold set.\n\n")
        for c in cards:
            revs = sorted({f"{e['repo']}@{e['rev14']}" for e in c["edits"]
                           if e.get("rev14")})
            fh.write(f"## {c['proposal']}-proposed · survival {c['survival']} · "
                     f"`{c['id'][:8]}…` · {c['klass']}\n")
            fh.write(f"- repo **{c.get('repo','?')}** branch `{c['branch'] or '?'}` · "
                     f"{c['start'][:10]} · {c['user_turns']} user turns · "
                     f"{c['interrupts']} interrupts (secondary) · ends on `{c['last_type']}`\n")
            for e in c["edits"]:
                where = (f" `[{e['repo']}]`"
                         if e.get("repo") and e["repo"] != c.get("repo") else "")
                fh.write(f"  - `{Path(e['file']).name}`{where} → **{e.get('fate','?')}**\n")
            if revs:
                fh.write(f"- probed at: {', '.join(f'`{r}`' for r in revs)}\n")
            fh.write(f"- full id: `{c['id']}`\n- VERDICT: [ ] CONFIRM  [ ] REJECT\n\n")
    print(f"strata: {json.dumps(strata, sort_keys=True)}")
    print(f"cards: {len(cards)} -> {out}")
    by = {}
    for c in cards:
        key = f"{c['klass']}/{c['proposal']}"
        by[key] = by.get(key, 0) + 1
    print(f"proposals: {json.dumps(by, sort_keys=True)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
