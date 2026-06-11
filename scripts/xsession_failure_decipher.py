"""Cross-session failure decipherer — v0 spike (operator directive 2026-06-11).

Finds the failure classes the operator NAMED from lived experience, which
single-session git-fate cannot see (it saturated FINE on this archive):

  C1 repeat-ask-with-gap   — same session, similar prompt re-asked after a
                             long break (the agent didn't land it the first time)
  C2 loopback-reopen       — different sessions days apart return to the same
                             topic AND re-edit the same files ("done" wasn't done)
  C3 persistence-landed    — operator pushes a similar ask across sessions;
                             early sessions produce no edits (rejected/deflected),
                             a later one finally implements it
  C4 recurring-topic       — cross-session prompt recurrence not matching C2/C3
                             (catch-all; human review decides)

WALL NOTE (recorded, not hidden): this instrument reads PROMPT TEXT — process-
side content under the Phase 0 v3 field table. It is a NEW ground-truth source
authorized by the operator (2026-06-11 directive) because outcome-only labeling
saturated. Consequence for the blind protocol (label/extractor wall redraw) is
flagged for Codex adjudication before any gate run uses these labels.

Spike mode per Rule 11 §1a (research mode). Output: evidence-backed candidate
chains for operator recognition-audit, NOT sealed labels.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
SOURCES = [
    ("cc", HOME / ".claude" / "projects"),
    ("cc", HOME / "arl-archive" / "snapshot-2026-06-11" / "claude-projects"),
    ("codex", HOME / ".codex" / "sessions"),
    ("codex", HOME / "arl-archive" / "snapshot-2026-06-11" / "codex-sessions"),
]

MIN_PROMPT_TOKENS = 6
GAP_MIN_WITHIN = 20 * 60          # C1: >=20min break
GAP_MIN_ACROSS = 6 * 3600         # C2-4: >=6h between sessions
SIM_WITHIN = 0.55
SIM_ACROSS = 0.38                 # paraphrased re-asks; precision via rare tokens
DF_CAP = 400                      # tokens appearing in more prompts are noise
MIN_SHARED_RARE = 4
COMMON_EDIT_NAMES = {"_index.md", "readme.md", "claude.md", "memory.md",
                     "log.md", "index.md", "settings.json", "agents.md",
                     "pyproject.toml", "package.json", "cargo.toml", ".gitignore"}


def norm_project(raw: str) -> str:
    """Join CC dir keys and Codex cwd basenames into one project key:
    'C--Users-Hung-Huynh-proj-cipher' -> 'cipher'; '...-Akashic' -> 'akashic'."""
    p = raw
    for pre in ("C--Users-Hung-Huynh-proj-", "C--Users-Hung-Huynh-"):
        if p.startswith(pre):
            p = p[len(pre):]
            break
    return p.strip("-").lower()

INTERRUPT_MARKERS = ("[Request interrupted",)
CC_EXCLUDE_PREFIX = ("<command-name>", "<local-command", "Caveat: The messages below",
                     "<system-reminder", "[Request interrupted",
                     "# /",  # skill/command expansions (e.g. /loop re-fires are
                             # MACHINE re-asks at sim 1.0 — not operator re-asks)
                     "<task-notification>", "Autonomy loop")
CODEX_EXCLUDE_PREFIX = ("<environment_context>", "<user_instructions>", "<turn_aborted",
                        "<ide_context", "# AGENTS", "<permissions")
MACHINE_FIRST_PROMPT = re.compile(
    r"^(You are |Run a |Adversarial|Launch |Execute |Review the |You're an?\b|"
    r"Autonomy loop|Read C:\\|Process the |Ingest )", re.IGNORECASE)
STOP = {"the", "and", "for", "that", "this", "with", "you", "are", "not", "but",
        "have", "was", "what", "all", "can", "your", "from", "they", "out",
        "use", "its", "now", "then", "into", "just", "should", "would", "need",
        "needs", "there", "here", "when", "where", "how", "why", "also", "see",
        "file", "files", "make", "made", "want", "like", "get", "got", "dont",
        "its", "lets", "let", "actually", "okay", "yes"}


def toks(text: str) -> frozenset:
    words = re.findall(r"[a-z0-9_./-]{3,}", text.lower())
    return frozenset(w for w in words if w not in STOP)


def jac(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def ts_of(s: str):
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return t.replace(tzinfo=t.tzinfo or timezone.utc)
    except (ValueError, AttributeError):
        return None


def parse_cc(path: Path) -> dict | None:
    s = {"id": path.stem, "kind": "cc", "project": path.parent.name, "path": str(path),
         "start": None, "end": None, "prompts": [], "edits": [], "machine": False}
    sidechain_hits = 0
    entries = 0
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                entries += 1
                if obj.get("isSidechain") is True:
                    sidechain_hits += 1
                t = ts_of(obj.get("timestamp") or "")
                if t:
                    s["start"] = s["start"] or t
                    s["end"] = t
                typ = obj.get("type")
                if typ == "user":
                    c = (obj.get("message") or {}).get("content")
                    text = None
                    if isinstance(c, str):
                        text = c
                    elif isinstance(c, list):
                        parts = [p.get("text", "") for p in c
                                 if isinstance(p, dict) and p.get("type") == "text"]
                        text = "\n".join(p for p in parts if p)
                    if text:
                        text = text.strip()
                        if text and not text.startswith(CC_EXCLUDE_PREFIX) \
                                and len(toks(text)) >= MIN_PROMPT_TOKENS:
                            s["prompts"].append((t, text[:1500], len(s["edits"])))
                elif typ == "assistant":
                    for p in (obj.get("message") or {}).get("content") or []:
                        if (isinstance(p, dict) and p.get("type") == "tool_use"
                                and p.get("name") in ("Edit", "Write", "NotebookEdit")):
                            fp = (p.get("input") or {}).get("file_path") or \
                                 (p.get("input") or {}).get("notebook_path")
                            if fp:
                                s["edits"].append(Path(str(fp)).name.lower())
    except OSError:
        return None
    if entries and sidechain_hits > entries * 0.5:
        s["machine"] = True
    if s["prompts"] and MACHINE_FIRST_PROMPT.match(s["prompts"][0][1]):
        s["machine"] = True
    return s if s["prompts"] else None


PATCH_FILE = re.compile(r"\*\*\* (?:Update|Add) File: (.+)")


def parse_codex(path: Path) -> dict | None:
    s = {"id": path.stem, "kind": "codex", "project": "", "path": str(path),
         "start": None, "end": None, "prompts": [], "edits": [], "machine": False}
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                t = ts_of(obj.get("timestamp") or "")
                if t:
                    s["start"] = s["start"] or t
                    s["end"] = t
                pay = obj.get("payload") or {}
                if obj.get("type") == "session_meta":
                    s["project"] = Path(str(pay.get("cwd") or "?")).name
                elif obj.get("type") == "response_item":
                    if pay.get("type") == "message" and pay.get("role") == "user":
                        texts = [c.get("text", "") for c in pay.get("content") or []
                                 if isinstance(c, dict) and c.get("type") == "input_text"]
                        text = "\n".join(x for x in texts if x).strip()
                        if text and not text.startswith(CODEX_EXCLUDE_PREFIX) \
                                and len(toks(text)) >= MIN_PROMPT_TOKENS:
                            s["prompts"].append((t, text[:1500], len(s["edits"])))
                    elif pay.get("type") == "function_call":
                        args = str(pay.get("arguments") or "")
                        for m in PATCH_FILE.finditer(args):
                            s["edits"].append(Path(m.group(1).strip().strip('"')).name.lower())
    except OSError:
        return None
    if s["prompts"] and MACHINE_FIRST_PROMPT.match(s["prompts"][0][1]):
        s["machine"] = True
    return s if s["prompts"] else None


def collect() -> list[dict]:
    seen: dict[str, dict] = {}
    for kind, root in SOURCES:
        if not root.exists():
            continue
        pattern = "**/rollout-*.jsonl" if kind == "codex" else "*/*.jsonl"
        for f in sorted(root.glob(pattern)):
            if f.stem in seen:
                continue  # live dirs come first in SOURCES; snapshot fills gaps
            s = (parse_codex if kind == "codex" else parse_cc)(f)
            if s and s["start"]:
                seen[f.stem] = s
    return list(seen.values())


def main(outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    sessions = collect()
    human = [s for s in sessions if not s["machine"]]
    print(f"sessions parsed: {len(sessions)} (human: {len(human)}, "
          f"machine-flagged: {len(sessions) - len(human)})")

    candidates = []

    # ---- C1: repeat-ask-with-gap (within session) ----
    for s in human:
        ps = s["prompts"]
        for i in range(len(ps)):
            ti, xi, _ = ps[i]
            a = toks(xi)
            for j in range(i + 1, len(ps)):
                tj, xj, _ = ps[j]
                gap = (tj - ti).total_seconds()
                if gap < GAP_MIN_WITHIN:
                    continue
                sim = jac(a, toks(xj))
                if sim >= SIM_WITHIN:
                    candidates.append({
                        "class": "C1-repeat-ask-with-gap", "score": round(sim, 2),
                        "sessions": [s["id"]], "project": s["project"], "kind": s["kind"],
                        "evidence": {
                            "first_ask": {"ts": ti.isoformat(), "text": xi[:200]},
                            "re_ask": {"ts": tj.isoformat(), "text": xj[:200]},
                            "gap_minutes": round(gap / 60)},
                    })
                    break  # one candidate per origin prompt

    # ---- cross-session: inverted index on rare tokens, bucketed by project ----
    by_proj = defaultdict(list)
    for s in human:
        by_proj[norm_project(s["project"])].append(s)

    chains = []
    for proj, group in by_proj.items():
        if len(group) < 2:
            continue
        prompts = []  # (sess_idx, ts, text, tokset, edits_before_prompt)
        for gi, s in enumerate(group):
            for (t, x, cnt) in s["prompts"]:
                prompts.append((gi, t, x, toks(x), cnt))
        df = defaultdict(int)
        for _, _, _, tk, _ in prompts:
            for w in tk:
                df[w] += 1
        index = defaultdict(list)
        for pi, (_, _, _, tk, _) in enumerate(prompts):
            for w in tk:
                if df[w] <= DF_CAP:
                    index[w].append(pi)
        pair_shared = defaultdict(int)
        for w, plist in index.items():
            if len(plist) > 50:
                continue
            for ii in range(len(plist)):
                for jj in range(ii + 1, len(plist)):
                    pair_shared[(plist[ii], plist[jj])] += 1
        # prompt-similarity edges (ordered earlier -> later)
        edges = []
        for (pi, pj), shared in pair_shared.items():
            if shared < MIN_SHARED_RARE:
                continue
            gi, ti, xi, tki, ci = prompts[pi]
            gj, tj, xj, tkj, cj = prompts[pj]
            if gi == gj:
                continue
            if abs((tj - ti).total_seconds()) < GAP_MIN_ACROSS:
                continue
            sim = jac(tki, tkj)
            if sim >= SIM_ACROSS:
                a, b = (pi, pj) if ti <= tj else (pj, pi)
                edges.append((sim, a, b, "prompt"))
        # file-reopen candidates: same uncommon basename re-edited on a LATER
        # calendar day — wording-independent. One candidate PER FILE (no
        # union-find: transitive merging collapsed everything into one blob).
        name_df = defaultdict(set)
        for gi, s in enumerate(group):
            for n in set(s["edits"]):
                name_df[n].add(gi)
        for n, gset in name_df.items():
            if n in COMMON_EDIT_NAMES or not (1 < len(gset) <= 8):
                continue
            gl = sorted(gset, key=lambda gi: group[gi]["start"])
            days = [group[gi]["start"].date() for gi in gl]
            if (days[-1] - days[0]).days < 1:
                continue
            sess = [group[gi] for gi in gl]
            candidates.append({
                "class": "C2-loopback-reopen", "score": 0.99, "project": proj,
                "kind": sess[0]["kind"],
                "sessions": [s["id"] for s in sess],
                "dates": [d.isoformat() for d in days],
                "flags": ["files-reopened"],
                "evidence": {"reopened_files": [n],
                             "span_days": (days[-1] - days[0]).days,
                             "edits_per_session": [len(s["edits"]) for s in sess]},
            })
        # union-find over sessions
        parent = list(range(len(group)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # resolve edges into evidence dicts NOW (prompt texts + edits-after),
        # then union-find; classification later never touches raw indices
        def resolve(sim, a, b, etype):
            if etype == "prompt":
                gi, ti, xi, _, ci = prompts[a]
                gj, tj, xj, _, cj = prompts[b]
                return gi, gj, {
                    "type": "prompt", "sim": round(sim, 2),
                    "earlier": {"ts": ti.isoformat(), "text": xi[:200],
                                "edits_after": len(group[gi]["edits"]) - ci},
                    "later": {"ts": tj.isoformat(), "text": xj[:200],
                              "edits_after": len(group[gj]["edits"]) - cj}}
            n = etype[5:]
            return a, b, {
                "type": "file-reopen", "file": n,
                "earlier": {"session": group[a]["id"],
                            "date": group[a]["start"].date().isoformat()},
                "later": {"session": group[b]["id"],
                          "date": group[b]["start"].date().isoformat()}}

        parent2 = parent  # alias; find() closes over `parent`
        chain_links = defaultdict(list)
        for sim, a, b, etype in sorted(edges, key=lambda e: -e[0]):
            gi, gj, ev = resolve(sim, a, b, etype)
            ra, rb = find(gi), find(gj)
            if ra != rb:
                parent2[ra] = rb
            chain_links[find(gi)].append(ev)
        roots = defaultdict(set)
        for gi in range(len(group)):
            roots[find(gi)].add(gi)
        for root, members in roots.items():
            links = chain_links.get(root, [])
            if len(members) < 2 or not links:
                continue
            chains.append((proj, group,
                           sorted(members, key=lambda gi: group[gi]["start"]),
                           links[:10]))

    # classify chains: C2 loopback-reopen / C3 persistence-landed / C4 recurring
    for proj, group, members, links in chains:
        sess = [group[gi] for gi in members]
        reopened = sorted({lk["file"] for lk in links if lk["type"] == "file-reopen"})
        persistence = [lk for lk in links if lk["type"] == "prompt"
                       and lk["earlier"]["edits_after"] == 0
                       and lk["later"]["edits_after"] > 0]
        flags = (["files-reopened"] if reopened else []) + \
                (["persistence-landed"] if persistence else [])
        if reopened:
            klass = "C2-loopback-reopen"
        elif persistence:
            klass = "C3-persistence-landed"
        else:
            klass = "C4-recurring-topic"
        prompt_links = [lk for lk in links if lk["type"] == "prompt"]
        best = max((lk.get("sim", 0.99) for lk in links), default=0)
        candidates.append({
            "class": klass, "score": round(best, 2),
            "project": proj, "kind": sess[0]["kind"],
            "sessions": [s["id"] for s in sess],
            "dates": [s["start"].date().isoformat() for s in sess],
            "flags": flags,
            "evidence": {
                "reopened_files": reopened[:8],
                "links": (persistence or prompt_links or links)[:4],
                "edits_per_session": [len(s["edits"]) for s in sess]},
        })

    candidates.sort(key=lambda c: (c["class"], -c["score"]))
    jl = out / "candidates.jsonl"
    with jl.open("w", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    by_class = defaultdict(int)
    for c in candidates:
        by_class[c["class"]] += 1
    print(f"candidates: {dict(by_class)} -> {jl}")


if __name__ == "__main__":
    main(sys.argv[1])
