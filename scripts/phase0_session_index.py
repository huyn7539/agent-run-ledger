"""Phase 0 session browser — generates a human-readable index of every session
in the Claude Code and Codex archives so the operator can FIND and LABEL them.

SPIKE-MODE tooling (Rule 11 §1a research carve-out): read-only over the
archives, writes one Markdown index. This is NOT signal extraction — it surfaces
only what the operator already knows (date, project, his own first prompt) so he
can recognize sessions. The Phase 0 blind seals labels BEFORE any signal
extractor runs; browsing metadata does not break the blind.

Output is LOCAL-SECRET (contains raw prompt excerpts) — vault-only, never export.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
CLAUDE_ROOT = HOME / ".claude" / "projects"
CODEX_ROOT = HOME / ".codex" / "sessions"
EXCERPT_LEN = 180
_TAG_RE = re.compile(r"<[^>]{1,80}>")


def _clean(text: str) -> str:
    text = _TAG_RE.sub(" ", text)
    text = " ".join(text.split())
    return text[:EXCERPT_LEN]


def _first_user_text_claude(path: Path) -> tuple[str, str, str]:
    """(start_ts, cwd, first_user_excerpt) — tolerant, first ~80 lines."""
    start, cwd, excerpt = "", "", ""
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 80 and excerpt:
                    break
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not start and isinstance(obj.get("timestamp"), str):
                    start = obj["timestamp"][:16]
                if not cwd and isinstance(obj.get("cwd"), str):
                    cwd = obj["cwd"]
                if not excerpt and obj.get("type") == "user":
                    msg = obj.get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        cand = _clean(content)
                        if len(cand) > 15:
                            excerpt = cand
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                cand = _clean(str(part.get("text", "")))
                                if len(cand) > 15:
                                    excerpt = cand
                                    break
    except OSError:
        pass
    return start, cwd, excerpt


def _first_user_text_codex(path: Path) -> tuple[str, str, str]:
    start, cwd, excerpt = "", "", ""
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 120 and excerpt:
                    break
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not start and isinstance(obj.get("timestamp"), str):
                    start = obj["timestamp"][:16]
                payload = obj.get("payload") or {}
                if not cwd:
                    meta_cwd = payload.get("cwd") or obj.get("cwd")
                    if isinstance(meta_cwd, str):
                        cwd = meta_cwd
                if not excerpt and payload.get("role") == "user":
                    for part in payload.get("content") or []:
                        if isinstance(part, dict) and part.get("type") in (
                            "input_text",
                            "text",
                        ):
                            raw = str(part.get("text", ""))
                            cand = _clean(raw)
                            boilerplate = (
                                "user_instructions" in raw
                                or "environment_context" in raw
                                or "America/" in cand
                                or cand.startswith(("C:", "/c/", "<"))
                            )
                            if len(cand) > 15 and not boilerplate:
                                excerpt = cand
                                break
    except OSError:
        pass
    return start, cwd, excerpt


def main(out_path: str) -> None:
    rows: list[tuple[str, str, str, str, str]] = []  # start, project, kind, id, excerpt
    for f in sorted(CLAUDE_ROOT.glob("*/*.jsonl")):
        start, cwd, excerpt = _first_user_text_claude(f)
        if not start:
            start = datetime.fromtimestamp(
                f.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M")
        project = (cwd or f.parent.name).replace("\\", "/").rsplit("/", 1)[-1]
        rows.append((start, project, "claude", f.stem[:13], excerpt or "(no text found)"))
    for f in sorted(CODEX_ROOT.rglob("*.jsonl")):
        start, cwd, excerpt = _first_user_text_codex(f)
        if not start:
            start = datetime.fromtimestamp(
                f.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M")
        project = (cwd or "?").replace("\\", "/").rsplit("/", 1)[-1]
        rows.append((start, project, "codex", f.stem[-13:], excerpt or "(no text found)"))

    rows.sort(reverse=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Phase 0 session index (LOCAL-SECRET - raw prompt excerpts; never export)\n\n"
            f"Generated from {CLAUDE_ROOT} and {CODEX_ROOT}. {len(rows)} sessions.\n"
            "Newest first. `id` is enough for labeling - copy the whole row.\n\n"
            "| start | project | kind | id | first prompt |\n|---|---|---|---|---|\n"
        )
        for start, project, kind, sid, excerpt in rows:
            safe = excerpt.replace("|", "\\|")
            fh.write(f"| {start} | {project} | {kind} | {sid} | {safe} |\n")
    print(f"wrote {len(rows)} sessions -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
