"""Task 60 — the fenced-block mutation contract (the ONLY file ARL ever edits).

Every clause of the Codex fenced-block contract is pinned here, and the
ambiguity locks are proven to BITE (vacuous-lock class, FAILURE-INDEX Cat 4).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_run_ledger.core.claudemd import (
    BEGIN_MARKER,
    END_MARKER,
    ApplyResult,
    BlockError,
    apply_line,
    revert_block,
)

LINE = "- ARL(retry_loop_budget/retry-budget/v1) tool=crm.lookup: cap retries."
LINE2 = "- ARL(retry_loop_budget/retry-budget/v1) tool=web.fetch: cap retries."


def test_apply_creates_block_in_fresh_file_and_is_idempotent_3x(tmp_path: Path) -> None:
    target = tmp_path / "CLAUDE.md"
    r1 = apply_line(target, tmp_path, LINE, create=True)
    assert r1.changed is True
    contents = [target.read_bytes()]
    for _ in range(2):
        r = apply_line(target, tmp_path, LINE, create=True)
        assert r.changed is False  # Rule 5: re-apply is a no-op
        contents.append(target.read_bytes())
    assert contents[0] == contents[1] == contents[2]
    text = target.read_text(encoding="utf-8")
    assert text.count(BEGIN_MARKER) == 1 and text.count(END_MARKER) == 1
    assert LINE in text


def test_apply_appends_block_preserving_user_content(tmp_path: Path) -> None:
    target = tmp_path / "CLAUDE.md"
    user = "# My project\n\nUser rules stay byte-identical.\n"
    target.write_text(user, encoding="utf-8")
    apply_line(target, tmp_path, LINE)
    text = target.read_text(encoding="utf-8")
    assert text.startswith(user.rstrip("\n") + "\n") or user in text
    assert text.index("User rules") < text.index(BEGIN_MARKER)
    # second line goes INSIDE the same block — still exactly one block
    apply_line(target, tmp_path, LINE2)
    text = target.read_text(encoding="utf-8")
    assert text.count(BEGIN_MARKER) == 1
    assert text.index(BEGIN_MARKER) < text.index(LINE2) < text.index(END_MARKER)


def test_crlf_newline_style_is_preserved(tmp_path: Path) -> None:
    target = tmp_path / "CLAUDE.md"
    target.write_bytes(b"# title\r\n\r\nbody\r\n")
    apply_line(target, tmp_path, LINE)
    raw = target.read_bytes()
    assert b"\r\n" in raw
    assert b"\r\r" not in raw  # the classic double-\r corruption
    # and the block round-trips idempotently under CRLF too
    before = target.read_bytes()
    assert apply_line(target, tmp_path, LINE).changed is False
    assert target.read_bytes() == before


@pytest.mark.parametrize(
    "hostile",
    [
        f"{BEGIN_MARKER}\nx\n{END_MARKER}\n{BEGIN_MARKER}\ny\n{END_MARKER}\n",  # two blocks
        f"{END_MARKER}\nx\n{BEGIN_MARKER}\n",  # end before begin
        f"{BEGIN_MARKER}\nx\n",  # begin without end
        f"text mentioning {BEGIN_MARKER} inline\n",  # marker text inside a line
        f"{BEGIN_MARKER}\n inner ARL:END smuggle\n{END_MARKER}\n",  # marker substring inside
    ],
)
def test_ambiguous_blocks_fail_closed_and_write_nothing(tmp_path: Path, hostile: str) -> None:
    target = tmp_path / "CLAUDE.md"
    target.write_text(hostile, encoding="utf-8")
    before = target.read_bytes()
    with pytest.raises(BlockError):
        apply_line(target, tmp_path, LINE)
    assert target.read_bytes() == before  # fail-closed: NOTHING was written


def test_missing_file_without_create_is_propose_only(tmp_path: Path) -> None:
    with pytest.raises(BlockError):
        apply_line(tmp_path / "CLAUDE.md", tmp_path, LINE)


def test_target_escaping_root_is_refused(tmp_path: Path) -> None:
    inside = tmp_path / "proj"
    inside.mkdir()
    outside = tmp_path / "CLAUDE.md"
    outside.write_text("x\n", encoding="utf-8")
    with pytest.raises(BlockError):
        apply_line(outside, inside, LINE)


def test_symlink_target_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real.md"
    real.write_text("x\n", encoding="utf-8")
    link = tmp_path / "CLAUDE.md"
    try:
        os.symlink(real, link)
    except OSError:
        pytest.skip("symlinks unavailable (Windows without Developer Mode)")
    with pytest.raises(BlockError):
        apply_line(link, tmp_path, LINE)


@pytest.mark.parametrize(
    "bad_line",
    [
        "two\nlines",
        "contains ARL:BEGIN marker text",
        "has `backtick`",
        "has <angle>",
        "has {brace}",
        "x" * 401,
    ],
)
def test_line_validation_rejects_unsafe_content(tmp_path: Path, bad_line: str) -> None:
    target = tmp_path / "CLAUDE.md"
    target.write_text("x\n", encoding="utf-8")
    with pytest.raises(BlockError):
        apply_line(target, tmp_path, bad_line)


def test_revert_cas_restores_and_removes_created_block(tmp_path: Path) -> None:
    target = tmp_path / "CLAUDE.md"
    target.write_text("# mine\n", encoding="utf-8")
    result: ApplyResult = apply_line(target, tmp_path, LINE)
    r = revert_block(target, tmp_path, result.after_block, result.before_block)
    assert r.status == "reverted"
    text = target.read_text(encoding="utf-8")
    assert LINE not in text and BEGIN_MARKER not in text
    assert "# mine" in text  # user content untouched


def test_revert_refuses_hand_edited_block(tmp_path: Path) -> None:
    target = tmp_path / "CLAUDE.md"
    target.write_text("# mine\n", encoding="utf-8")
    result = apply_line(target, tmp_path, LINE)
    # user hand-edits INSIDE the managed block after the apply
    text = target.read_text(encoding="utf-8").replace(LINE, LINE + " (edited)")
    target.write_text(text, encoding="utf-8")
    before = target.read_bytes()
    r = revert_block(target, tmp_path, result.after_block, result.before_block)
    assert r.status == "review"  # never clobber user edits
    assert target.read_bytes() == before
