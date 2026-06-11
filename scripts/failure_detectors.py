"""D-tier failure detectors — taxonomy v4 §5 coverage (dual-GREEN plan W1).

Detectors implemented (taxonomy ids):
  E4x  cross-session error re-derivation     I1  permission deadlock
  I2   tool-error loops                      I3  interrupted-call misread
  F1   hallucinated reference                F3  platform recurrence
  H1   dead dispatch (CC; codex deferred)    H2  destructive events
  I10  green-claim → env-fix join (72h date window, global-shape claims)

Design rules (plan W1/W2, Codex-amended):
- env errors / permission denials credited ONLY from tool OUTPUT, never prose
- I3 requires the completion claim to SHARE SUBJECT TOKENS with the empty call
- F1 fires only on assistant-introduced tokens (never seen in prompts or
  prior tool output); optional-dep probes (--version / try-shape) excluded;
  novelty proxy disclosed: an uninstalled real module is indistinguishable
  from an invented one — graded at audit via the D-tier precision rubric
- H1 excludes user-aborted dispatches and clean structured empty results
- emits AUDIT CANDIDATES, never sealed labels (evidence-role law §3a)

Spike mode (Rule 11 §1a) with W6 retrofit tests in tests/test_failure_detectors.py.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import xsession_failure_decipher as XS

CLAIM_RE = re.compile(
    r"\b(all (?:tests?|checks?) pass|tests? (?:are )?green|suite is green|"
    r"done\b|complete(?:d|ly)?\b|ready for (?:first )?user|production[- ]ready|"
    r"works? (?:now|correctly)|fixed\b|resolved\b|verified\b|ship(?:ped|s)?\b|"
    r"good to go|no (?:errors?|regressions?))", re.IGNORECASE)
GLOBAL_CLAIM_RE = re.compile(
    r"\b(all (?:tests?|checks?) pass|suite is green|ready for (?:first )?user|"
    r"production[- ]ready|no (?:errors?|regressions?))", re.IGNORECASE)
ENV_ERR_RE = re.compile(
    r"(no module named|command not found|is not recognized|cannot find|"
    r"modulenotfounderror|importerror|permission denied|access is denied|"
    r"cp1252|codec can't|charmap|crlf|\bbom\b|no such file|"
    r"executable not found|exit code [1-9]|non-zero exit|segmentation fault)",
    re.IGNORECASE)
PLATFORM_ERR_RE = re.compile(
    r"(cp1252|charmap|\bbom\b|crlf|powershell|msys|ucrt|win(?:dows|32)|"
    r"\\\\\?\\|set-content|\.venv\\scripts|\$env:)", re.IGNORECASE)
DELETE_CMD_RE = re.compile(
    r"\b(rm -rf|rm -r\b|Remove-Item.*-Recurse|del /|rmdir /s|"
    r"git clean -[a-z]*f|git reset --hard)", re.IGNORECASE)
NOTFOUND_PATTERNS = [
    re.compile(r"No module named ['\"]?([\w.]+)", re.IGNORECASE),
    re.compile(r"name '(\w+)' is not defined"),
    re.compile(r"has no attribute '(\w+)'"),
    re.compile(r"'([\w.\-]+)' is not recognized", re.IGNORECASE),
    re.compile(r"(\S+): command not found"),
    re.compile(r"unknown option[:'\s]+['\"]?([\w\-]+)", re.IGNORECASE),
]
PROBE_SHAPE_RE = re.compile(r"--version|--help|\btry:|which |Get-Command|check",
                            re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-Z_][\w.\-]{2,}")


def _tokens(text: str) -> set[str]:
    """Words plus their segments (test_billing.py also yields test, billing)
    so subject overlap survives extension/underscore differences (I3)."""
    out: set[str] = set()
    for w in WORD_RE.findall(text):
        out.add(w.lower())
        for seg in re.split(r"[._/\\-]", w.lower()):
            if len(seg) >= 3:
                out.add(seg)
    return out


def _norm_err(s: str) -> str:
    s = re.sub(r"\d+", "N", s.lower())
    s = re.sub(r"[\"'`].*?[\"'`]", "X", s)
    return s.strip()[:50]


def new_facts(sid: str, kind: str) -> dict:
    return {"id": sid, "kind": kind, "tool_errors": [], "tool_calls": 0,
            "claims": [], "global_claims": [], "deletes": [],
            "perm_denials": [], "env_errors": [],
            "i3_fires": [], "f1_fires": [], "h1_fires": [],
            # scan state (underscored keys are dropped before output)
            "_seen": set(), "_last_call": "", "_last_call_probe": False,
            "_pending_empty": None, "_pending_tasks": {}, "_interrupted": False}


def _on_prompt(f: dict, text: str) -> None:
    f["_seen"] |= _tokens(text)


def _on_tool_use(f: dict, name: str, input_text: str, call_id: str) -> None:
    f["tool_calls"] += 1
    f["_last_call"] = input_text[:300]
    f["_last_call_probe"] = bool(PROBE_SHAPE_RE.search(input_text[:300]))
    f["_interrupted"] = False
    for m in DELETE_CMD_RE.finditer(input_text):
        f["deletes"].append(m.group(0)[:60])
    if name == "Task" and call_id:
        f["_pending_tasks"][call_id] = input_text[:120]


def _on_tool_result(f: dict, text: str, is_error: bool, call_id: str) -> None:
    if is_error:
        f["tool_errors"].append("is_error")
    # ONE env-error event per result: a single crash line matching two
    # vocabulary patterns (cp1252 + charmap) is one error, not two (W6 pin)
    m = ENV_ERR_RE.search(text)
    if m:
        f["env_errors"].append(m.group(0)[:60])
    low = text.lower()
    if ("denied by the claude code" in low or "permission to use" in low
            or re.search(r"\b(tool|action|command) .{0,30}\bdenied\b", low)):
        f["perm_denials"].append(text[:80])
    stripped = text.strip()
    empty = (not stripped) or stripped.startswith("[Request interrupted")
    if stripped.startswith("[Request interrupted"):
        f["_interrupted"] = True
    if empty:
        f["_pending_empty"] = {"head": f["_last_call"], "countdown": 3}
    # H1: paired dead dispatch (empty or error completion), excluding
    # user-aborted and clean structured-empty results
    if call_id and call_id in f["_pending_tasks"]:
        head = f["_pending_tasks"].pop(call_id)
        clean_structured = stripped.startswith("{") or "agentId" in stripped
        if (empty or is_error) and not f["_interrupted"] and not clean_structured:
            f["h1_fires"].append({"dispatch": head,
                                  "result": stripped[:80] or "<empty>"})
    # F1: assistant-introduced reference fails as nonexistent. Tightened
    # after a 274-fire over-count: the token must be ≥4 chars, non-numeric,
    # NOVEL (never in prompts/prior tool output), and present in the
    # assistant's OWN call input (it used the reference it invented).
    if is_error or ENV_ERR_RE.search(text):
        for pat in NOTFOUND_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            token = m.group(1).lower().strip(".")
            # TDD red phase: a test importing a not-yet-built module is
            # planned-failing, not a hallucinated reference. Check BOTH the
            # error text and the originating call (pytest test_<token> ...).
            ctx = (text + " " + f["_last_call"]).lower()
            tdd_red = ("pytest" in ctx or "_bootstrap" in ctx
                       or f"test_{token}" in ctx or "test summary" in ctx
                       or "collected" in ctx)
            if (len(token) >= 4 and not token.isdigit()
                    and token not in f["_seen"]
                    and token in f["_last_call"].lower()
                    and not f["_last_call_probe"] and not tdd_red):
                f["f1_fires"].append({"token": token, "error": text[:100]})
            break
    f["_seen"] |= _tokens(text)


def _on_assistant_text(f: dict, text: str) -> None:
    for m in CLAIM_RE.finditer(text):
        f["claims"].append(m.group(0).lower())
    for m in GLOBAL_CLAIM_RE.finditer(text):
        f["global_claims"].append(m.group(0).lower())
    pe = f["_pending_empty"]
    if pe:
        if (CLAIM_RE.search(text)
                and len(_tokens(pe["head"]) & _tokens(text)) >= 2):
            f["i3_fires"].append({"call": pe["head"][:100],
                                  "claim": text[:120]})
            f["_pending_empty"] = None
            return
        pe["countdown"] -= 1
        if pe["countdown"] <= 0:
            f["_pending_empty"] = None


def scan_cc(obj: dict, f: dict) -> None:
    t = obj.get("type")
    c = (obj.get("message") or {}).get("content")
    if t == "assistant" and isinstance(c, list):
        for p in c:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "tool_use":
                _on_tool_use(f, p.get("name") or "",
                             json.dumps(p.get("input") or {})[:2000],
                             p.get("id") or "")
            elif p.get("type") == "text":
                _on_assistant_text(f, p.get("text") or "")
    elif t == "user":
        if isinstance(c, str):
            _on_prompt(f, c)
        elif isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "tool_result":
                    body = p.get("content")
                    text = body if isinstance(body, str) else json.dumps(body or "")[:2000]
                    _on_tool_result(f, text, bool(p.get("is_error")),
                                    p.get("tool_use_id") or "")
                elif p.get("type") == "text":
                    _on_prompt(f, p.get("text") or "")


def scan_codex(obj: dict, f: dict) -> None:
    pay = obj.get("payload") or {}
    pt = pay.get("type")
    if pt == "message":
        for c in pay.get("content") or []:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "input_text":
                _on_prompt(f, c.get("text") or "")
            elif c.get("type") == "output_text":
                _on_assistant_text(f, c.get("text") or "")
    elif pt == "function_call":
        _on_tool_use(f, pay.get("name") or "",
                     str(pay.get("arguments") or "")[:2000],
                     pay.get("call_id") or "")
    elif pt == "function_call_output":
        out = pay.get("output")
        text = out if isinstance(out, str) else json.dumps(out or "")[:2000]
        _on_tool_result(f, text, "error" in text.lower()[:40],
                        pay.get("call_id") or "")


def session_facts(path: Path, kind: str) -> dict:
    f = new_facts(path.stem, kind)
    scan = scan_codex if kind == "codex" else scan_cc
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                scan(obj, f)
    except OSError:
        pass
    return {k: v for k, v in f.items() if not k.startswith("_")}


# ---------- per-session detectors ----------

def detect_i2_tool_error_loops(facts: dict) -> list[dict]:
    n = len(facts["tool_errors"])
    if n >= 3 and facts["tool_calls"] and n >= 0.25 * facts["tool_calls"]:
        return [_c("I2-tool-infra-loop", facts,
                   {"tool_errors": n, "tool_calls": facts["tool_calls"],
                    "samples": facts["env_errors"][:3]})]
    return []


def detect_i1_permission_deadlock(facts: dict) -> list[dict]:
    if len(facts["perm_denials"]) >= 2:
        return [_c("I1-permission-deadlock", facts,
                   {"denials": len(facts["perm_denials"]),
                    "samples": facts["perm_denials"][:3]})]
    return []


def detect_f3_platform_recurrence(facts: dict) -> list[dict]:
    plat = {_norm_err(e) for e in facts["env_errors"] if PLATFORM_ERR_RE.search(e)}
    if len(plat) >= 2:
        return [_c("F3-wrong-environment", facts,
                   {"distinct_platform_errors": len(plat),
                    "samples": sorted(plat)[:3]})]
    return []


def detect_h2_destructive(facts: dict) -> list[dict]:
    if facts["deletes"]:
        return [_c("H2-destructive-action", facts,
                   {"destructive_cmds": facts["deletes"][:5]})]
    return []


def detect_i3_interrupted_misread(facts: dict) -> list[dict]:
    return [_c("I3-interrupted-call-misread", facts, e)
            for e in facts["i3_fires"][:5]]


def detect_f1_hallucinated_reference(facts: dict) -> list[dict]:
    return [_c("F1-hallucinated-reference", facts, e)
            for e in facts["f1_fires"][:5]]


def detect_h1_dead_dispatch(facts: dict) -> list[dict]:
    return [_c("H1-dead-dispatch", facts, e) for e in facts["h1_fires"][:5]]


def _c(klass: str, facts: dict, evidence: dict) -> dict:
    return {"class": klass, "session": facts["id"], "kind": facts["kind"],
            "evidence": evidence}


def detect_e4_cross_session_rederivation(all_facts: list[dict]) -> list[dict]:
    sig_sessions = defaultdict(list)
    for f in all_facts:
        for e in {_norm_err(x) for x in f["env_errors"]}:
            if e:
                sig_sessions[e].append(f["id"])
    out = []
    for sig, sess in sig_sessions.items():
        uniq = sorted(set(sess))
        if len(uniq) >= 2:
            out.append({"class": "E4-cross-session-rederivation",
                        "evidence": {"error_signature": sig,
                                     "sessions": uniq[:6],
                                     "n_sessions": len(uniq)}})
    return out


def detect_i10_green_then_envfix(facts: dict, end: datetime | None,
                                 repo_fixes: list[dict]) -> list[dict]:
    """Global-shape claim, then an environment-specific fix lands in the same
    repo within 72h AFTER the session end (plan W1 date window)."""
    if not facts["global_claims"] or end is None:
        return []
    hits = []
    for r in repo_fixes:
        msg = r.get("fix", r.get("msg", ""))
        if not (PLATFORM_ERR_RE.search(msg) or "install" in msg.lower()
                or "fresh" in msg.lower() or "console" in msg.lower()):
            continue
        try:
            fix_date = datetime.fromisoformat(r["date"]).date()
        except (KeyError, ValueError):
            continue
        delta = (fix_date - end.date()).days
        if 0 <= delta <= 3:
            hits.append(f"{r['date']}: {msg[:70]}")
    if hits:
        return [_c("I10-test-env-divergence", facts,
                   {"claims": sorted(set(facts["global_claims"]))[:4],
                    "env_fixes_within_72h": hits[:3]})]
    return []


def main(outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    sessions = XS.collect()
    human = [s for s in sessions if not s["machine"]]
    end_by_id = {s["id"]: s["end"] for s in human}
    proj_of = {s["id"]: XS.norm_project(s["project"]) for s in human}
    facts = [session_facts(Path(s["path"]), s["kind"]) for s in human]

    rework = {}
    rw_path = out / "git-rework.json"
    if rw_path.exists():
        for r in json.loads(rw_path.read_text(encoding="utf-8")):
            rework[r["repo"].lower()] = r

    fired = []
    for f in facts:
        for det in (detect_i2_tool_error_loops, detect_i1_permission_deadlock,
                    detect_f3_platform_recurrence, detect_h2_destructive,
                    detect_i3_interrupted_misread,
                    detect_f1_hallucinated_reference,
                    detect_h1_dead_dispatch):
            fired.extend(det(f))
        rw = rework.get(proj_of.get(f["id"], ""), {})
        fired.extend(detect_i10_green_then_envfix(
            f, end_by_id.get(f["id"]), rw.get("fix_after_feat", [])))
    fired.extend(detect_e4_cross_session_rederivation(facts))

    by_class = defaultdict(int)
    for c in fired:
        by_class[c["class"]] += 1
    jl = out / "detector-fires.jsonl"
    with jl.open("w", encoding="utf-8") as fh:
        for c in fired:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"sessions: {len(human)} human; detector fires: "
          f"{json.dumps(dict(sorted(by_class.items())))}")
    print(f"-> {jl}")


if __name__ == "__main__":
    main(sys.argv[1])
