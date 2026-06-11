"""D-tier failure detectors — taxonomy v4 §5 coverage (GREEN 2026-06-11).

Implements the deterministic detectors the two-engine-ratified taxonomy
obligates (§5 D-tier list). Each detector is a pure function over a parsed
session (or session pair) and emits candidate records with quoted evidence,
the taxonomy class id, and an honest no-fire when clean. Proposer/judgment
calls are explicitly OUT of scope here (those are §5 P-tier grading rubrics).

Evidence-role discipline (§3a): PROCESS-side detectors (B3-cont, E4-sub, F1,
F3, I1, I2, I3) read prompt/tool text and are extractor-eligible only under
sealed truncation; LABEL-eligible detectors (C4, I10, I12) read outcome
evidence. This module produces CANDIDATES for the operator audit, never
sealed labels, and never mixes the two roles inside one fire.

Spike mode (Rule 11 §1a) — written, run, observed; tests retrofit on keep.
Reuses xsession_failure_decipher.collect() for the corpus + adapters.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import xsession_failure_decipher as XS

# ---- richer per-session extraction (second pass; the decipherer kept only
# prompts + edit basenames; detectors need tool errors, names, claims, deletes)

CLAIM_RE = re.compile(
    r"\b(all (?:tests?|checks?) pass|tests? (?:are )?green|suite is green|"
    r"done\b|complete(?:d|ly)?\b|ready for (?:first )?user|production[- ]ready|"
    r"works? (?:now|correctly)|fixed\b|resolved\b|verified\b|ship(?:ped|s)?\b|"
    r"good to go|no (?:errors?|regressions?))",
    re.IGNORECASE)
ENV_ERR_RE = re.compile(
    r"(no module named|command not found|is not recognized|cannot find|"
    r"modulenotfounderror|importerror|permission denied|access is denied|"
    r"cp1252|codec can't|charmap|crlf|bom|utf-8|no such file|"
    r"executable not found|exit code [1-9]|non-zero exit|segmentation fault)",
    re.IGNORECASE)
PLATFORM_ERR_RE = re.compile(
    r"(cp1252|charmap|\bbom\b|crlf|powershell|msys|ucrt|win(?:dows|32)|"
    r"\\\\\?\\|backslash|path separator|line ending|set-content|"
    r"\.venv\\scripts|/dev/null|\$env:)", re.IGNORECASE)
PERM_DENY_RE = re.compile(
    r"(permission to use|denied by|not allowed|requires approval|"
    r"auto-?deny|blocked by hook|permission denied for|cannot .* without)",
    re.IGNORECASE)
NOVEL_REF_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{3,})`|\b([A-Z][a-zA-Z]{4,})\(")
DELETE_TOOLS = {"Bash"}  # delete/overwrite surface in CC; inspected by content
DELETE_CMD_RE = re.compile(
    r"\b(rm -rf|rm -r|Remove-Item.*-Recurse|del /|rmdir /s|git clean -[a-z]*f|"
    r"git reset --hard|truncate|> [^|&\s]+\.(py|md|json|ts|rs|cpp|h))",
    re.IGNORECASE)


def session_facts(path: Path, kind: str) -> dict:
    """Second-pass extraction of detector signals (does NOT replace the
    decipherer's prompt/edit pass; complements it)."""
    f = {"id": path.stem, "kind": kind, "tool_errors": [], "tool_calls": 0,
         "claims": [], "deletes": [], "perm_denials": [], "interrupted_calls": 0,
         "novel_refs": [], "env_errors": []}
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"is_error":true' in line.replace(" ", ""):
                    f["tool_errors"].append("is_error")
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                _scan_obj(obj, kind, f)
    except OSError:
        pass
    return f


def _texts_of(obj: dict, kind: str):
    """Yield (role, text) for message-bearing entries, adapter-aware."""
    if kind == "cc":
        msg = obj.get("message") or {}
        role = obj.get("type")
        c = msg.get("content")
        if isinstance(c, str):
            yield role, c
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        yield role, p.get("text", "")
                    elif p.get("type") == "tool_use":
                        yield "tool_use", json.dumps(p.get("input") or {})[:2000]
                    elif p.get("type") == "tool_result":
                        body = p.get("content")
                        yield "tool_result", (body if isinstance(body, str)
                                              else json.dumps(body)[:2000])
    else:  # codex
        pay = obj.get("payload") or {}
        if pay.get("type") == "message":
            for c in pay.get("content") or []:
                if isinstance(c, dict) and c.get("type") in ("input_text", "output_text"):
                    yield pay.get("role", "?"), c.get("text", "")
        elif pay.get("type") == "function_call":
            yield "tool_use", str(pay.get("arguments") or "")[:2000]
        elif pay.get("type") == "function_call_output":
            out = pay.get("output")
            yield "tool_result", (out if isinstance(out, str) else json.dumps(out)[:2000])


def _scan_obj(obj: dict, kind: str, f: dict) -> None:
    for role, text in _texts_of(obj, kind):
        if not text:
            continue
        if role == "tool_use":
            f["tool_calls"] += 1
            if any(d in text for d in ("rm -rf", "Remove-Item", "reset --hard")):
                for m in DELETE_CMD_RE.finditer(text):
                    f["deletes"].append(m.group(0)[:60])
        elif role == "tool_result":
            # env errors are credited ONLY from actual tool OUTPUT (not from
            # assistant prose mentioning the vocabulary) — kills the F3 false-
            # accusation that fired on sessions merely discussing CRLF/BOM/PS.
            for m in ENV_ERR_RE.finditer(text):
                f["env_errors"].append(m.group(0)[:60])
            # permission denials: only the harness's OWN denial phrasing, not
            # the word "denied" appearing in scraped web/file content (I1
            # false-fire seen on a Fellows-scrape tool_result).
            if "denied by the claude code" in text.lower() or \
                    "permission to use" in text.lower() or \
                    re.search(r"\b(tool|action|command) .{0,30}\bdenied\b", text.lower()):
                f["perm_denials"].append(text[:80])
        elif role == "assistant":
            for m in CLAIM_RE.finditer(text):
                f["claims"].append(m.group(0).lower())


# ---------- detectors (each returns list[candidate]) ----------

def detect_i2_tool_error_loops(facts: dict) -> list[dict]:
    """I2 tool-infra failure loop: >=3 tool errors in a session, and they
    dominate (>=25% of tool calls)."""
    n = len(facts["tool_errors"])
    if n >= 3 and facts["tool_calls"] and n >= 0.25 * facts["tool_calls"]:
        return [{"class": "I2-tool-infra-loop", "session": facts["id"],
                 "kind": facts["kind"],
                 "evidence": {"tool_errors": n, "tool_calls": facts["tool_calls"],
                              "samples": facts["env_errors"][:3]}}]
    return []


def detect_i1_permission_deadlock(facts: dict) -> list[dict]:
    if len(facts["perm_denials"]) >= 2:
        return [{"class": "I1-permission-deadlock", "session": facts["id"],
                 "kind": facts["kind"],
                 "evidence": {"denials": len(facts["perm_denials"]),
                              "samples": facts["perm_denials"][:3]}}]
    return []


def detect_f3_platform_recurrence(facts: dict) -> list[dict]:
    # count DISTINCT platform-error signatures, not raw line hits (one cp1252
    # crash echoed across many tool_results is ONE recurrence, not 53).
    plat = {_norm_err(e) for e in facts["env_errors"] if PLATFORM_ERR_RE.search(e)}
    if len(plat) >= 2:
        return [{"class": "F3-wrong-environment", "session": facts["id"],
                 "kind": facts["kind"],
                 "evidence": {"distinct_platform_errors": len(plat),
                              "samples": sorted(plat)[:3]}}]
    return []


def detect_h2_destructive(facts: dict) -> list[dict]:
    if facts["deletes"]:
        return [{"class": "H2-destructive-action", "session": facts["id"],
                 "kind": facts["kind"],
                 "evidence": {"destructive_cmds": facts["deletes"][:5]}}]
    return []


def detect_e4_cross_session_rederivation(all_facts: list[dict]) -> list[dict]:
    """E4-subtype: the SAME environment error string recurs in >=2 sessions on
    different calendar... days handled by caller (facts carry no ts); here we
    join by normalized error signature across sessions."""
    sig_sessions = defaultdict(list)
    for f in all_facts:
        for e in set(_norm_err(x) for x in f["env_errors"]):
            if e:
                sig_sessions[e].append(f["id"])
    out = []
    for sig, sess in sig_sessions.items():
        uniq = sorted(set(sess))
        if len(uniq) >= 2:
            out.append({"class": "E4-cross-session-rederivation",
                        "evidence": {"error_signature": sig,
                                     "sessions": uniq[:6], "n_sessions": len(uniq)}})
    return out


def _norm_err(s: str) -> str:
    s = re.sub(r"\d+", "N", s.lower())
    s = re.sub(r"[\"'`].*?[\"'`]", "X", s)
    return s.strip()[:50]


def detect_i10_green_then_envfix(facts: dict, repo_fixes: list[dict]) -> list[dict]:
    """I10 test-env divergence: a session makes a green/done claim, and the
    SAME repo gets an environment-specific fix shortly after (join lives in the
    caller which has dates; here we surface the claim+have-fixes precondition)."""
    if facts["claims"] and repo_fixes:
        env_fixes = [r for r in repo_fixes
                     if PLATFORM_ERR_RE.search(r.get("msg", ""))
                     or "install" in r.get("msg", "").lower()
                     or "fresh" in r.get("msg", "").lower()
                     or "console" in r.get("msg", "").lower()]
        if env_fixes:
            return [{"class": "I10-test-env-divergence", "session": facts["id"],
                     "kind": facts["kind"],
                     "evidence": {"claims": sorted(set(facts["claims"]))[:4],
                                  "env_fixes_after": [r["msg"][:80]
                                                      for r in env_fixes[:3]]}}]
    return []


def main(outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    sessions = XS.collect()
    human = [s for s in sessions if not s["machine"]]
    facts = [session_facts(Path(s["path"]), s["kind"]) for s in human]

    fired = []
    for f in facts:
        for det in (detect_i2_tool_error_loops, detect_i1_permission_deadlock,
                    detect_f3_platform_recurrence, detect_h2_destructive):
            fired.extend(det(f))
    fired.extend(detect_e4_cross_session_rederivation(facts))

    # I10 join: claim sessions × repo env-fixes by project (dates from XS)
    rework = {}
    rw_path = out / "git-rework.json"
    if rw_path.exists():
        for r in json.loads(rw_path.read_text(encoding="utf-8")):
            rework[r["repo"].lower()] = r
    proj_of = {s["id"]: XS.norm_project(s["project"]) for s in human}
    for f in facts:
        proj = proj_of.get(f["id"], "")
        rw = rework.get(proj, {})
        env_fixes = [{"msg": ff["fix"]} for ff in rw.get("fix_after_feat", [])]
        fired.extend(detect_i10_green_then_envfix(f, env_fixes))

    by_class = defaultdict(int)
    for c in fired:
        by_class[c["class"]] += 1
    jl = out / "detector-fires.jsonl"
    with jl.open("w", encoding="utf-8") as fh:
        for c in fired:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"sessions: {len(human)} human; detector fires: {dict(by_class)}")
    print(f"-> {jl}")


if __name__ == "__main__":
    main(sys.argv[1])
