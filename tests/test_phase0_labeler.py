"""Phase 0 labeler v1 — regression tests for the four measurement defects found
2026-06-11 (pre-audit, pre-extractor; see vault amendment file
06-learning/agent-run-ledger/phase0/2026-06-11-labeler-v1-amendment.md).

Defects pinned here:
  D1 cross-repo misattribution — edits joined against session cwd repo, not the
     repo the edited file lives in (card 1: ARL files committed at 9dc4f75 read
     as UNCOMMITTED against Akashic).
  D2 gitignored files counted as UNCOMMITTED evidence (.env.*, settings.local).
  D3 out-of-repo files (e.g. ~/.claude memory dir) counted as UNCOMMITTED.
  D4 single-probe brittleness — one longest line amended pre-commit flipped a
     committed file to UNCOMMITTED (card 14 fleet-agreement.md).
  D5 no maturity window — sessions younger than end+14d judged against
     incomplete history.
Plus wall discipline (no prompt text in cards) and output determinism.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import phase0_labeler as labeler

MATURE_END = "2026-05-01T12:00:00Z"
COMMIT_DATE = "2026-05-03T12:00:00 +0000"  # inside the +14d horizon of MATURE_END
NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)  # well past the horizon

PROBE_A = "alpha probe line long enough to anchor content survival checks"
PROBE_B = "beta probe line long enough to anchor content survival checks"
PROBE_MISSING = "this longest line was amended before commit and never landed"
PROBE_SECOND = "second-longest line that did land in the commit"


def _run_git(cwd: Path, *args: str, env: dict | None = None) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True, capture_output=True, text=True, env=env,
    )


def make_repo(base: Path, name: str) -> Path:
    repo = base / name
    repo.mkdir()
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "t@test")
    _run_git(repo, "config", "user.name", "t")
    return repo


def commit_all(repo: Path, msg: str, when: str = COMMIT_DATE) -> None:
    env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
    _run_git(repo, "add", "-A")
    _run_git(repo, "commit", "-q", "-m", msg, env=env)


def base_session(cwd: Path, edits: list[dict]) -> dict:
    return {
        "id": "test-session", "cwd": str(cwd), "branch": "main",
        "start": "2026-05-01T10:00:00Z", "end": MATURE_END,
        "interrupts": 0, "user_turns": 5, "last_type": "assistant",
        "edits": edits,
    }


def test_d1_cross_repo_edit_joins_against_file_repo(tmp_path):
    """An edit to a file in repoB must be joined against repoB's history even
    when the session cwd is repoA."""
    repo_a = make_repo(tmp_path, "cwd-repo")
    repo_b = make_repo(tmp_path, "file-repo")
    (repo_b / "f1.py").write_text(f"# {PROBE_A}\n", encoding="utf-8")
    (repo_b / "f2.py").write_text(f"# {PROBE_B}\n", encoding="utf-8")
    commit_all(repo_b, "ship both files")

    s = base_session(repo_a, [
        {"file": str(repo_b / "f1.py"), "probes": [f"# {PROBE_A}"]},
        {"file": str(repo_b / "f2.py"), "probes": [f"# {PROBE_B}"]},
    ])
    s = labeler.fate_session(s, now=NOW)

    assert s["stratum"] == "labeled"
    assert [e["fate"] for e in s["edits"]] == ["SURVIVED", "SURVIVED"]
    assert s["survival"] == 1.0
    assert s["proposal"] == "FINE"


def test_d2_gitignored_file_excluded_from_denominator(tmp_path):
    repo = make_repo(tmp_path, "repo")
    (repo / ".gitignore").write_text(".env.*\n", encoding="utf-8")
    (repo / "a.py").write_text(f"# {PROBE_A}\n", encoding="utf-8")
    (repo / "b.py").write_text(f"# {PROBE_B}\n", encoding="utf-8")
    commit_all(repo, "ship")
    (repo / ".env.local").write_text("KEY=value-never-committed-by-design\n",
                                     encoding="utf-8")

    s = base_session(repo, [
        {"file": str(repo / ".env.local"),
         "probes": ["KEY=value-never-committed-by-design"]},
        {"file": str(repo / "a.py"), "probes": [f"# {PROBE_A}"]},
        {"file": str(repo / "b.py"), "probes": [f"# {PROBE_B}"]},
    ])
    s = labeler.fate_session(s, now=NOW)

    assert s["edits"][0]["fate"] == "IGNORED"
    assert s["stratum"] == "labeled"
    assert s["survival"] == 1.0  # rate over the 2 valid probes only
    assert s["proposal"] == "FINE"


def test_d3_out_of_repo_file_excluded(tmp_path):
    repo = make_repo(tmp_path, "repo")
    (repo / "a.py").write_text(f"# {PROBE_A}\n", encoding="utf-8")
    (repo / "b.py").write_text(f"# {PROBE_B}\n", encoding="utf-8")
    commit_all(repo, "ship")
    loose = tmp_path / "no-repo-here"
    loose.mkdir()
    (loose / "MEMORY.md").write_text("memory file outside any git repository\n",
                                     encoding="utf-8")

    s = base_session(repo, [
        {"file": str(loose / "MEMORY.md"),
         "probes": ["memory file outside any git repository"]},
        {"file": str(repo / "a.py"), "probes": [f"# {PROBE_A}"]},
        {"file": str(repo / "b.py"), "probes": [f"# {PROBE_B}"]},
    ])
    s = labeler.fate_session(s, now=NOW)

    assert s["edits"][0]["fate"] == "OUT-OF-REPO"
    assert s["stratum"] == "labeled"
    assert s["survival"] == 1.0
    assert s["proposal"] == "FINE"


def test_d4_any_of_k_probes_survives(tmp_path):
    """A committed file whose longest in-session line was amended pre-commit
    must still read SURVIVED via a secondary probe line."""
    repo = make_repo(tmp_path, "repo")
    (repo / "doc.md").write_text(
        f"{PROBE_SECOND}\nshort line\n", encoding="utf-8")
    (repo / "other.md").write_text(f"{PROBE_A} second anchor file\n",
                                   encoding="utf-8")
    commit_all(repo, "ship amended doc")

    s = base_session(repo, [
        {"file": str(repo / "doc.md"), "probes": [PROBE_MISSING, PROBE_SECOND]},
        {"file": str(repo / "other.md"), "probes": [f"{PROBE_A} second anchor file"]},
    ])
    s = labeler.fate_session(s, now=NOW)

    assert s["edits"][0]["fate"] == "SURVIVED"
    assert s["survival"] == 1.0


def test_d5_too_recent_session_held_out(tmp_path):
    repo = make_repo(tmp_path, "repo")
    (repo / "a.py").write_text(f"# {PROBE_A}\n", encoding="utf-8")
    commit_all(repo, "ship")

    fresh_end = (NOW - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    s = base_session(repo, [
        {"file": str(repo / "a.py"), "probes": [f"# {PROBE_A}"]},
    ])
    s["end"] = fresh_end
    s = labeler.fate_session(s, now=NOW)

    assert s["stratum"] == "too-recent"
    assert "proposal" not in s


def test_parse_collects_probes_across_all_edits_to_a_file(tmp_path):
    """Probes for a file must be drawn from every edit body, not only the
    first edit (the other half of D4)."""
    first_body = f"{PROBE_MISSING} extra padding text here\nshort\n"
    second_body = f"{PROBE_SECOND} with more padding text\nshort\n"
    jsonl = tmp_path / "sess.jsonl"
    rows = [
        {"type": "user", "timestamp": "2026-05-01T10:00:00Z",
         "cwd": str(tmp_path), "gitBranch": "main",
         "message": {"content": "do the thing"}},
        {"type": "assistant", "timestamp": "2026-05-01T10:01:00Z",
         "message": {"content": [
             {"type": "tool_use", "name": "Write",
              "input": {"file_path": str(tmp_path / "doc.md"),
                        "content": first_body}}]}},
        {"type": "assistant", "timestamp": "2026-05-01T10:02:00Z",
         "message": {"content": [
             {"type": "tool_use", "name": "Edit",
              "input": {"file_path": str(tmp_path / "doc.md"),
                        "old_string": "x", "new_string": second_body}}]}},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    s = labeler.parse_session(jsonl)

    assert len(s["edits"]) == 1
    probes = s["edits"][0]["probes"]
    assert any(PROBE_MISSING in p for p in probes)
    assert any(PROBE_SECOND in p for p in probes)


def _write_e2e_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo = make_repo(tmp_path, "repo")
    (repo / "a.py").write_text(f"# {PROBE_A}\n", encoding="utf-8")
    (repo / "b.py").write_text(f"# {PROBE_B}\n", encoding="utf-8")
    commit_all(repo, "ship")
    rows = [
        {"type": "user", "timestamp": MATURE_END, "cwd": str(repo),
         "gitBranch": "main",
         "message": {"content": "SECRET-PROMPT-MARKER-XYZZY build it"}},
        {"type": "assistant", "timestamp": MATURE_END, "message": {"content": [
            {"type": "tool_use", "name": "Write",
             "input": {"file_path": str(repo / "a.py"),
                       "content": f"# {PROBE_A}\n"}}]}},
        {"type": "assistant", "timestamp": MATURE_END, "message": {"content": [
            {"type": "tool_use", "name": "Write",
             "input": {"file_path": str(repo / "b.py"),
                       "content": f"# {PROBE_B}\n"}}]}},
    ]
    jsonl = tmp_path / "sess.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "counts": {},
        "human_sessions": [{"kind": "claude", "path": str(jsonl)}],
    }), encoding="utf-8")
    return manifest, tmp_path / "cards.md"


def test_wall_no_prompt_text_in_cards(tmp_path):
    manifest, out = _write_e2e_fixture(tmp_path)
    labeler.main(str(manifest), str(out))
    text = out.read_text(encoding="utf-8")
    assert "SECRET-PROMPT-MARKER-XYZZY" not in text
    assert "SURVIVED" in text


def test_output_deterministic_across_runs(tmp_path):
    manifest, out = _write_e2e_fixture(tmp_path)
    labeler.main(str(manifest), str(out))
    first = out.read_text(encoding="utf-8")
    labeler.main(str(manifest), str(out))
    second = out.read_text(encoding="utf-8")
    assert first == second


def test_codex3_tracked_but_ignored_file_is_fate_probed(tmp_path):
    """git check-ignore matches tracked files too; a tracked file matching an
    ignore pattern must still get a real fate, not IGNORED."""
    repo = make_repo(tmp_path, "repo")
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (repo / "a.log").write_text(f"{PROBE_A} inside tracked log\n", encoding="utf-8")
    (repo / "b.py").write_text(f"# {PROBE_B}\n", encoding="utf-8")
    _run_git(repo, "add", "-f", "a.log")
    commit_all(repo, "ship tracked-but-ignore-matching file")

    s = base_session(repo, [
        {"file": str(repo / "a.log"), "probes": [f"{PROBE_A} inside tracked log"]},
        {"file": str(repo / "b.py"), "probes": [f"# {PROBE_B}"]},
    ])
    s = labeler.fate_session(s, now=NOW)

    assert s["edits"][0]["fate"] == "SURVIVED"
    assert s["survival"] == 1.0


def test_codex4_git_failure_routes_to_probe_error(tmp_path, monkeypatch):
    """A git subprocess failure (timeout/OS error) must surface as PROBE-ERROR
    excluded from the denominator — never read as UNCOMMITTED."""
    repo = make_repo(tmp_path, "repo")
    (repo / "a.py").write_text(f"# {PROBE_A}\n", encoding="utf-8")
    (repo / "b.py").write_text(f"# {PROBE_B}\n", encoding="utf-8")
    (repo / "c.py").write_text("# unrelated committed content here\n", encoding="utf-8")
    commit_all(repo, "ship")

    real_git = labeler._git

    def flaky_git(repo_arg, *args):
        if args and args[0] == "show" and any("c.py" in a for a in args):
            raise labeler.GitProbeError("simulated timeout")
        return real_git(repo_arg, *args)

    monkeypatch.setattr(labeler, "_git", flaky_git)
    s = base_session(repo, [
        {"file": str(repo / "c.py"), "probes": ["# probe that will never get checked ok"]},
        {"file": str(repo / "a.py"), "probes": [f"# {PROBE_A}"]},
        {"file": str(repo / "b.py"), "probes": [f"# {PROBE_B}"]},
    ])
    s = labeler.fate_session(s, now=NOW)

    assert s["edits"][0]["fate"] == "PROBE-ERROR"
    assert s["stratum"] == "labeled"
    assert s["survival"] == 1.0  # over the 2 valid probes only


def test_fleet_symlinked_edit_path_resolves_to_physical_repo(tmp_path):
    """C1/C6/C9: an edit recorded under a symlinked dir must join against the
    physical repo the symlink targets (the ~/.claude memory-dir case)."""
    import os
    repo = make_repo(tmp_path, "repo")
    sub = repo / "memory"
    sub.mkdir()
    (sub / "m1.md").write_text(f"{PROBE_A} memory fact one\n", encoding="utf-8")
    (sub / "m2.md").write_text(f"{PROBE_B} memory fact two\n", encoding="utf-8")
    commit_all(repo, "ship memory files")
    alias = tmp_path / "alias-dir"
    try:
        os.symlink(sub, alias, target_is_directory=True)
    except OSError:
        import pytest
        pytest.skip("symlink creation not permitted on this host")

    s = base_session(tmp_path, [
        {"file": str(alias / "m1.md"), "probes": [f"{PROBE_A} memory fact one"]},
        {"file": str(alias / "m2.md"), "probes": [f"{PROBE_B} memory fact two"]},
    ])
    s = labeler.fate_session(s, now=NOW)

    assert [e["fate"] for e in s["edits"]] == ["SURVIVED", "SURVIVED"]
    assert s["proposal"] == "FINE"


def test_fleet_prefix_strip_failure_is_no_label(tmp_path, monkeypatch):
    """If the physical path cannot be made relative to the resolved toplevel,
    the fate is PATH-UNRESOLVED (no-label) — never UNCOMMITTED."""
    repo = make_repo(tmp_path, "repo")
    (repo / "a.py").write_text(f"# {PROBE_A}\n", encoding="utf-8")
    (repo / "b.py").write_text(f"# {PROBE_B}\n", encoding="utf-8")
    commit_all(repo, "ship")
    unrelated = make_repo(tmp_path, "unrelated")

    real_repo_of = labeler._repo_of

    def warped_repo_of(path):
        if "a.py" in path:
            return str(unrelated).replace("\\", "/")
        return real_repo_of(path)

    monkeypatch.setattr(labeler, "_repo_of", warped_repo_of)
    s = base_session(repo, [
        {"file": str(repo / "a.py"), "probes": [f"# {PROBE_A}"]},
        {"file": str(repo / "b.py"), "probes": [f"# {PROBE_B}"]},
        {"file": str(repo / "b.py") + "x", "probes": ["never committed content xx"]},
    ])
    s = labeler.fate_session(s, now=NOW)

    assert s["edits"][0]["fate"] == "PATH-UNRESOLVED"
    assert s["edits"][0]["fate"] not in labeler.VALID_FATES


def test_fleet_renamed_file_still_reads_survived(tmp_path):
    """C7/C10: vault task files move (in-progress -> done) after the session;
    the content anchor must follow the move, not flip to GONE/UNCOMMITTED."""
    repo = make_repo(tmp_path, "repo")
    (repo / "task-spec.md").write_text(f"{PROBE_A} task body line\n", encoding="utf-8")
    (repo / "other.md").write_text(f"{PROBE_B} second file\n", encoding="utf-8")
    commit_all(repo, "ship at original path")
    (repo / "done").mkdir()
    _run_git(repo, "mv", "task-spec.md", "done/task-spec.md")
    commit_all(repo, "queue move to done/", when="2026-05-05T12:00:00 +0000")

    s = base_session(repo, [
        {"file": str(repo / "task-spec.md"), "probes": [f"{PROBE_A} task body line"]},
        {"file": str(repo / "other.md"), "probes": [f"{PROBE_B} second file"]},
    ])
    s = labeler.fate_session(s, now=NOW)

    assert s["edits"][0]["fate"] == "SURVIVED"
    assert s["survival"] == 1.0


def test_fleet_edit_context_lines_are_not_probes(tmp_path):
    """C8/C13: lines present in old_string are pre-existing context, not the
    session's new content — they must not anchor survival."""
    ctx = "this pre-existing context line is long enough to qualify"
    new = "the genuinely new line this edit introduced to the file"
    rows = [
        {"type": "user", "timestamp": MATURE_END, "cwd": str(tmp_path),
         "gitBranch": "main", "message": {"content": "go"}},
        {"type": "assistant", "timestamp": MATURE_END, "message": {"content": [
            {"type": "tool_use", "name": "Edit",
             "input": {"file_path": str(tmp_path / "doc.md"),
                       "old_string": f"{ctx}\nshort",
                       "new_string": f"{ctx}\n{new}\nshort"}}]}},
    ]
    jsonl = tmp_path / "sess.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    s = labeler.parse_session(jsonl)

    probes = s["edits"][0]["probes"]
    assert new in probes
    assert ctx not in probes


def test_fleet_quoted_preexisting_content_does_not_anchor(tmp_path):
    """v1.2 novelty gate: a probe line that already existed anywhere in the
    tree BEFORE the session (quoted spec text, templates) must not anchor
    survival — an edit with no novel probe is NO-NOVEL-CONTENT (no-label)."""
    quoted = "a spec sentence that predates the session and gets quoted a lot"
    repo = make_repo(tmp_path, "repo")
    (repo / "spec.md").write_text(f"{quoted}\n", encoding="utf-8")
    commit_all(repo, "pre-existing spec", when="2026-04-20T12:00:00 +0000")
    (repo / "a.py").write_text(f"# {PROBE_A}\n", encoding="utf-8")
    (repo / "b.py").write_text(f"# {PROBE_B}\n", encoding="utf-8")
    commit_all(repo, "session-era work")

    s = base_session(repo, [
        # review file quoting the old spec line; the file itself never landed
        {"file": str(repo / "review.md"), "probes": [quoted]},
        {"file": str(repo / "a.py"), "probes": [f"# {PROBE_A}"]},
        {"file": str(repo / "b.py"), "probes": [f"# {PROBE_B}"]},
    ])
    s = labeler.fate_session(s, now=NOW)

    assert s["edits"][0]["fate"] == "NO-NOVEL-CONTENT"
    assert s["survival"] == 1.0  # over the 2 novel probes only
    assert s["proposal"] == "FINE"


def test_fleet_ambiguous_is_a_no_label_stratum(tmp_path):
    """C4/C11: mid-band survival is an explicit no-label state, never a card."""
    repo = make_repo(tmp_path, "repo")
    (repo / "a.py").write_text(f"# {PROBE_A}\n", encoding="utf-8")
    commit_all(repo, "ship a only")
    (repo / "b.py").write_text("# never committed line of content\n", encoding="utf-8")

    s = base_session(repo, [
        {"file": str(repo / "a.py"), "probes": [f"# {PROBE_A}"]},
        {"file": str(repo / "b.py"), "probes": ["# never committed line of content"]},
    ])
    s = labeler.fate_session(s, now=NOW)

    assert s["survival"] == 0.5
    assert s["stratum"] == "ambiguous"
    assert "proposal" not in s


def test_fleet_deck_order_ignores_interrupts():
    """C2/C16: distress facts never order the audit deck."""
    a = {"klass": "code-primary", "proposal": "FINE", "survival": 0.8,
         "id": "aaa", "interrupts": 9}
    b = {"klass": "code-primary", "proposal": "FINE", "survival": 0.8,
         "id": "aaa", "interrupts": 0}
    assert labeler._card_sort_key(a) == labeler._card_sort_key(b)


def test_fleet_user_turns_exclude_tool_result_entries(tmp_path):
    rows = [
        {"type": "user", "timestamp": MATURE_END, "cwd": str(tmp_path),
         "message": {"content": "a real typed prompt"}},
        {"type": "user", "timestamp": MATURE_END, "message": {"content": [
            {"type": "tool_result", "content": "tool output noise"}]}},
        {"type": "user", "timestamp": MATURE_END, "message": {"content": [
            {"type": "text", "text": "another real turn"}]}},
    ]
    jsonl = tmp_path / "sess.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    s = labeler.parse_session(jsonl)

    assert s["user_turns"] == 2


def test_codex8_interrupts_counted_from_structured_marker_only(tmp_path):
    """Prompt or tool_result text containing the interrupt phrase must not
    count; only the harness's exact structured marker does."""
    rows = [
        {"type": "user", "timestamp": MATURE_END, "cwd": str(tmp_path),
         "gitBranch": "main",
         "message": {"content": "my prompt mentions Request interrupted by user"}},
        {"type": "user", "timestamp": MATURE_END, "message": {"content": [
            {"type": "tool_result",
             "content": "log line: Request interrupted by user blah"}]}},
        {"type": "user", "timestamp": MATURE_END, "message": {"content": [
            {"type": "text", "text": "[Request interrupted by user]"}]}},
        {"type": "user", "timestamp": MATURE_END, "message": {"content": [
            {"type": "text", "text": "[Request interrupted by user for tool use]"}]}},
        {"type": "user", "timestamp": MATURE_END, "message": {"content": [
            {"type": "text",
             "text": "[Request interrupted by user] plus trailing prompt text"}]}},
    ]
    jsonl = tmp_path / "sess.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    s = labeler.parse_session(jsonl)

    assert s["interrupts"] == 2
