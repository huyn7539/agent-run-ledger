"""Task 60 — the ARL-managed fenced block in CLAUDE.md (the ONLY mutation lane).

Codex spec-review contract (adopted in full; every clause below is test-pinned):
  * Exact marker lines; the parser accepts exactly ZERO or ONE complete block
    (begin before end, no nesting, no duplicates). ANY ambiguity — including a
    marker substring smuggled inside other text — raises ``BlockError`` and the
    caller stays propose-only. Fail-closed, never "best effort".
  * Byte-preserve everything outside the block; preserve the file's dominant
    newline style; never reorder or rewrite user content.
  * The target must be a REGULAR FILE inside the project root after
    ``resolve()`` — no symlinks/junctions, no traversal, no UNC escapes. A
    missing file is propose-only unless the caller passes ``create=True``.
  * Compare-and-swap on the preimage hash + same-directory temp file +
    ``os.replace`` (atomic on the same volume); kill-mid-write leaves either
    the old file or the new file, never a torn one.
  * Revert is CAS-based: only when the CURRENT block hash equals the recorded
    ``after_hash`` — a hand-edited block is never clobbered; the caller gets
    ``"review"`` instead (Codex: blind revert can destroy user edits).
  * Rule 5: applying the same line three times is byte-identical after the
    first apply; exactly one managed block; stable insertion order; no
    timestamps inside the file.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

BEGIN_MARKER = "<!-- ARL:BEGIN managed block (do not edit inside) -->"
END_MARKER = "<!-- ARL:END -->"

_MAX_LINE_LEN = 400


class BlockError(ValueError):
    """The managed block is ambiguous or the target is unsafe — the caller
    must fall back to propose-only (fail-closed; no mutation happened)."""


@dataclass(frozen=True)
class ApplyResult:
    changed: bool
    before_hash: str  # sha256 of the WHOLE file before (preimage for CAS)
    after_hash: str  # sha256 of the WHOLE file after
    before_block: str  # inner block text before ("" if block absent)
    after_block: str  # inner block text after


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def block_hash(inner_text: str) -> str:
    return _sha256(inner_text.encode("utf-8"))


def _check_target(path: Path, root: Path, *, must_exist: bool) -> None:
    if path.is_symlink():
        raise BlockError(f"target is a symlink: {path}")
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise BlockError(f"target escapes the project root: {resolved}") from None
    if path.exists():
        if not path.is_file():
            raise BlockError(f"target is not a regular file: {path}")
    elif must_exist:
        raise BlockError(f"target does not exist (propose-only without create=True): {path}")


def _validate_line(line: str) -> None:
    if "\n" in line or "\r" in line:
        raise BlockError("managed line must be a single line")
    if len(line) > _MAX_LINE_LEN:
        raise BlockError(f"managed line exceeds {_MAX_LINE_LEN} chars")
    if "ARL:BEGIN" in line or "ARL:END" in line:
        raise BlockError("managed line may not contain marker text")
    if any(ch in line for ch in ("`", "<", ">", "{", "}")):
        raise BlockError("managed line contains forbidden characters")


def _split_block(text: str) -> tuple[list[str], list[str], list[str], str, bool]:
    """Return (head_lines, inner_lines, tail_lines, eol, had_block) for exactly
    0/1 block. Lines are NORMALIZED (no trailing \\r); reassembly joins with the
    detected eol, so the file's newline style is preserved without doubling.
    A file mixing newline styles is refused outright — one detected eol cannot
    reassemble it byte-identically, and byte-preservation outranks convenience.

    Marker lines must be BYTE-EXACT (an indented or padded marker is ambiguity,
    not a marker). Any OTHER line containing marker text is ambiguity too."""
    if "\r" in text and (
        text.count("\r\n") != text.count("\n") or text.count("\r") != text.count("\r\n")
    ):
        raise BlockError("mixed newline styles — byte-preserve cannot hold; fail closed")
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    begins = [i for i, ln in enumerate(lines) if ln == BEGIN_MARKER]
    ends = [i for i, ln in enumerate(lines) if ln == END_MARKER]
    strays = [
        i
        for i, ln in enumerate(lines)
        if ("ARL:BEGIN" in ln or "ARL:END" in ln) and ln not in (BEGIN_MARKER, END_MARKER)
    ]
    if strays:
        raise BlockError(f"marker text outside exact marker lines (lines {strays})")
    if len(begins) == 0 and len(ends) == 0:
        return lines, [], [], eol, False
    if len(begins) != 1 or len(ends) != 1 or begins[0] >= ends[0]:
        raise BlockError(
            f"ambiguous managed block (begins at {begins}, ends at {ends}) — fail closed"
        )
    b, e = begins[0], ends[0]
    inner = lines[b + 1 : e]
    return lines[:b], inner, lines[e + 1 :], eol, True


def _atomic_write(path: Path, payload: bytes, preimage_hash: str | None) -> None:
    """Same-directory temp + fsync + os.replace, with a last-instant CAS check.
    A concurrent edit between our read and the replace aborts with BlockError
    (the caller reports propose-only/review; nothing was written).

    ``preimage_hash=None`` means the caller read NO file — the create is
    published with an atomic NO-CLOBBER primitive (``os.link`` refuses an
    existing target), so even a file that appears between the early check and
    the publish cannot be clobbered (Codex T61 review: check-then-replace was
    still TOCTOU). A file that VANISHES after being read fails the CAS read
    and aborts — never silently reclassified as a create."""
    if preimage_hash is not None:
        try:
            current = _sha256(path.read_bytes())
        except FileNotFoundError:
            raise BlockError(
                "target vanished since it was read (CAS mismatch) — aborting"
            ) from None
        if current != preimage_hash:
            raise BlockError("CLAUDE.md changed since it was read (CAS mismatch) — aborting")
    elif path.exists():
        raise BlockError("target appeared since it was read (create race) — aborting")
    fd, tmp_name = tempfile.mkstemp(prefix=".arl-block-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if preimage_hash is None:
            try:
                os.link(tmp_name, path)
            except FileExistsError:
                raise BlockError(
                    "target appeared since it was read (create race) — aborting"
                ) from None
            except OSError as exc:
                # filesystem without hard links: fail closed, never clobber
                raise BlockError(
                    f"atomic no-clobber create unsupported here ({exc}); "
                    "create the file manually, then re-run apply"
                ) from None
            os.unlink(tmp_name)
        else:
            os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def apply_line(path: Path, root: Path, line: str, *, create: bool = False) -> ApplyResult:
    """Insert *line* into the managed block (creating the block, and with
    ``create=True`` the file, if absent). Idempotent: a line already present
    leaves the file byte-identical and returns ``changed=False``."""
    _validate_line(line)
    _check_target(path, root, must_exist=not create)

    # capture existence ONCE at read time (Codex T61 review: recomputing
    # exists() at the write call site let a file deleted mid-call be
    # misclassified as a create and skip the CAS path)
    existed = path.exists()
    if not existed:
        raw = b""
        text = ""
    else:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    before_hash = _sha256(raw)

    head, inner, tail, eol, had_block = _split_block(text)
    before_block = "\n".join(inner)
    if had_block and line in inner:
        return ApplyResult(False, before_hash, before_hash, before_block, before_block)

    new_inner = [*inner, line]
    if not text:
        out_lines = [BEGIN_MARKER, *new_inner, END_MARKER, ""]
    elif not had_block:
        # no block yet: append one at the end, preserving existing content
        body = head  # tail is empty in the no-block case (split returns all in head)
        if body and body[-1] != "":
            body = [*body, ""]
        out_lines = [*body, BEGIN_MARKER, *new_inner, END_MARKER, ""]
    else:
        out_lines = [*head, BEGIN_MARKER, *new_inner, END_MARKER, *tail]
    payload = eol.join(out_lines).encode("utf-8")

    _atomic_write(path, payload, before_hash if existed else None)
    return ApplyResult(
        True, before_hash, _sha256(payload), before_block, "\n".join(new_inner)
    )


@dataclass(frozen=True)
class RevertResult:
    status: str  # "reverted" | "review"
    detail: str


def revert_block(
    path: Path, root: Path, recorded_after_block: str, before_block: str
) -> RevertResult:
    """CAS revert: ONLY when the current inner block matches the recorded
    after-state, restore the recorded before-state. Anything else (hand edits,
    a second tool's changes) -> "review", and NOTHING is written.

    The recorded before-state is NOT trusted (Codex P2 review, CRITICAL): a
    crafted registry row must never become bytes in CLAUDE.md. Every restored
    line passes the same managed-line contract as an applied line, or the
    revert routes to "review" with nothing written."""
    _check_target(path, root, must_exist=True)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    head, inner, tail, eol, had_block = _split_block(text)
    if not had_block:
        return RevertResult("review", "no managed block present; nothing to revert")
    current_inner = "\n".join(inner)
    if block_hash(current_inner) != block_hash(recorded_after_block):
        return RevertResult(
            "review",
            "managed block no longer matches the recorded post-apply state; "
            "not reverting (review the stored before/after manually)",
        )
    restored = before_block.split("\n") if before_block else []
    for ln in restored:
        try:
            _validate_line(ln)
        except BlockError as exc:
            return RevertResult(
                "review",
                f"recorded before-state violates the managed-line contract ({exc}); "
                "refusing to write it (review the stored before/after manually)",
            )
    if restored:
        out_lines = [*head, BEGIN_MARKER, *restored, END_MARKER, *tail]
    else:
        # block was created by the apply: remove it entirely
        out_lines = [*head, *tail]
        while out_lines and out_lines[-1] == "":
            out_lines.pop()
        out_lines.append("")
    payload = eol.join(out_lines).encode("utf-8")
    _atomic_write(path, payload, _sha256(raw))
    return RevertResult("reverted", "managed block restored to the recorded before-state")
