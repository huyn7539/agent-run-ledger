"""Phase 0 corpus builder — PASS 1 hygiene (flags/regex only; sealed rules).

Implements Phase 0 v3 amendment #3: classify every session file in both
archives as HUMAN-corpus or EXCLUDED(reason), with per-reason counts.
Reads ONLY: file paths, isSidechain flags, entry types, and the first user
text (for the sealed dispatch regex). No signal-side content analysis.

Sealed rules (changing any of these re-seals the protocol):
  EX1 path-machine : path contains 'subagents' or 'codex_exec'
  EX2 sidechain    : first user-type entry has isSidechain true
  EX3 dispatch     : first user text matches the sealed dispatch regex
  EX4 turn-floor   : fewer than 3 user turns
  EX5 unreadable   : no parseable entries
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HOME = Path.home()
ROOTS = [("claude", HOME / ".claude" / "projects"), ("codex", HOME / ".codex" / "sessions")]

DISPATCH_RE = re.compile(
    r"^(You are |Run a Codex|Adversarial (security )?review|Codex (exec|review))"
    r"|subtask for repo|ADVERSARIAL REVIEW|fleet of SPECIALIZED",
    re.IGNORECASE,
)
TURN_FLOOR = 3


def classify(path: Path, kind: str) -> tuple[str, str]:
    """Return (verdict, detail): verdict in {human, EX1..EX5}."""
    p = str(path).replace("\\", "/")
    if "subagents" in p or "codex_exec" in p:
        return "EX1", "machine path"
    user_turns = 0
    first_user_text = ""
    first_user_sidechain = None
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 400 and user_turns >= TURN_FLOOR and first_user_text:
                    break
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if kind == "claude" and obj.get("type") == "user":
                    user_turns += 1
                    if first_user_sidechain is None:
                        first_user_sidechain = bool(obj.get("isSidechain"))
                    if not first_user_text:
                        msg = obj.get("message") or {}
                        c = msg.get("content")
                        if isinstance(c, str):
                            first_user_text = c[:300]
                        elif isinstance(c, list):
                            for part in c:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    first_user_text = str(part.get("text", ""))[:300]
                                    break
                elif kind == "codex":
                    payload = obj.get("payload") or {}
                    if payload.get("role") == "user":
                        user_turns += 1
                        if not first_user_text:
                            for part in payload.get("content") or []:
                                if isinstance(part, dict) and part.get("type") in (
                                    "input_text",
                                    "text",
                                ):
                                    t = str(part.get("text", ""))
                                    if (
                                        "environment_context" not in t
                                        and "user_instructions" not in t
                                    ):
                                        first_user_text = t[:300]
                                        break
    except OSError:
        return "EX5", "unreadable"
    if user_turns == 0 and not first_user_text:
        return "EX5", "no entries"
    if first_user_sidechain:
        return "EX2", "sidechain"
    if first_user_text and DISPATCH_RE.search(first_user_text.lstrip()):
        return "EX3", "dispatch prompt"
    if user_turns < TURN_FLOOR:
        return "EX4", f"{user_turns} user turns"
    return "human", f"{user_turns} turns"


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    manifest: list[dict] = []
    for kind, root in ROOTS:
        for f in sorted(root.rglob("*.jsonl")):
            verdict, detail = classify(f, kind)
            counts[f"{kind}:{verdict}"] = counts.get(f"{kind}:{verdict}", 0) + 1
            if verdict == "human":
                manifest.append({"kind": kind, "path": str(f), "detail": detail})
    (out / "corpus-manifest.json").write_text(
        json.dumps({"counts": counts, "human_sessions": manifest}, indent=1),
        encoding="utf-8",
    )
    for k in sorted(counts):
        print(f"{k}: {counts[k]}")
    print(f"HUMAN CORPUS: {len(manifest)} sessions -> {out / 'corpus-manifest.json'}")


if __name__ == "__main__":
    main(sys.argv[1])
