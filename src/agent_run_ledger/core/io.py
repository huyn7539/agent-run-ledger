from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent_run_ledger.core.models import (
    _RAW_CONTENT_METADATA_KEYS,
    TraceBundle,
    TraceValidationError,
)

# Constraint 4: ARL imports UNTRUSTED trace files. Bound size + nesting so a
# hostile trace cannot exhaust memory or blow the parser stack. These are
# generous for real traces (a 25 MB / 200-deep trace is already pathological) and
# tunable; the point is a HARD ceiling, not a tight fit.
MAX_TRACE_BYTES = 25 * 1024 * 1024
MAX_TRACE_DEPTH = 200


class TraceParseError(ValueError):
    """Raised when an untrusted trace file cannot be parsed SAFELY (too large,
    too deeply nested, malformed, or not a trace object). Always a typed, caught
    error — never a crash or an uncaught RecursionError."""


def _check_depth(text: str) -> None:
    """Reject pathological nesting by counting bracket depth on the raw text,
    BEFORE json.loads (which can blow the C-stack on deep input). String contents
    are skipped so brackets inside string literals never inflate the count."""
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > MAX_TRACE_DEPTH:
                raise TraceParseError(
                    f"trace nesting exceeds max depth {MAX_TRACE_DEPTH}"
                )
        elif ch in "]}":
            depth -= 1


def load_json_object(path: Path) -> dict[str, Any]:
    """Defensively read an untrusted JSON file into a top-level object.

    The single safe entry point for every single-object import shape (the neutral
    TraceBundle and any adapter-routed recorded export): size and nesting are
    bounded, encoding and parse failures raise a typed TraceParseError, and the
    result is inert data — nothing is ever evaluated."""
    size = path.stat().st_size
    if size > MAX_TRACE_BYTES:
        raise TraceParseError(
            f"trace file is too large: {size} bytes > {MAX_TRACE_BYTES}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise TraceParseError(f"trace file is not valid UTF-8: {exc}") from exc
    _check_depth(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TraceParseError(f"malformed trace JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TraceParseError("trace top-level must be a JSON object")
    return data


def load_trace(path: Path) -> TraceBundle:
    """Parse an untrusted trace file into a TraceBundle, defensively.

    Every field is treated as inert string/scalar data — nothing in a trace is
    ever evaluated or executed (json.loads only produces data). Size and depth are
    bounded; malformed or non-object input raises a typed TraceParseError."""
    data = load_json_object(path)
    try:
        return TraceBundle.from_dict(data)
    except TraceValidationError as exc:
        raise TraceParseError(f"invalid trace bundle: {exc}") from exc


# A label on the raw export so a reader does not mistake the raw per-span facts for
# the derived view. The base stores ONE raw step per span (a real retry loop keeps
# each attempt's raw ``retry_count=0``); the retry collapse — and the
# ``retry_count=N`` a prescription cites — is a JUDGMENT computed ON READ
# (prescriptions.derive_retry_steps), never baked into the corpus. Without this
# note a raw export showing ``retry_count=0`` looks self-contradictory next to a
# prescription citing ``retry_count=2``. This is a RAW LOCAL facts export.
_EXPORT_NOTE_RAW_LOCAL = (
    "raw facts export (local): one immutable step per captured span. retry_count "
    "here is the per-span raw value (0 for un-collapsed attempts); the derived "
    "retry view (retry_count=N) that prescriptions cite is computed ON READ and is "
    "not persisted. Allowed metadata values are exported verbatim (raw local "
    "export, not a remote leak). Do NOT share this form; the default export is "
    "the share-safe scrubbed form."
)

_EXPORT_NOTE_SHARE = (
    "share-form export (Task 46): raw-content metadata values (before/after/path/"
    "patch targets) and prescription patch bodies are DROPPED at this egress "
    "boundary; dropped key NAMES are disclosed under _scrubbed_keys. retry_count "
    "here is the per-span raw value (0 for un-collapsed attempts); the derived "
    "retry view a prescription cites is computed ON READ and is not persisted. "
    "Full-fidelity LOCAL form: export with --raw-local."
)

# A content-free replacement patch. Shaped to satisfy EVERY patch_type validator
# (unified_diff / config_diff headers; an "=" for code_snippet; "def test_" for
# regression_test) so a scrubbed export re-imports cleanly under its original
# patch_type — while _is_retry_cap_diff still REJECTS it (no retry-budget
# identifier), so a re-imported scrubbed patch can never be upgraded to L2.
_PATCH_SCRUB_MARKER = (
    "--- a/ARL-SCRUBBED\n"
    "+++ b/ARL-SCRUBBED\n"
    "@@ -1 +1 @@\n"
    "-ARL_PATCH_SCRUBBED = 1  [scrubbed at export: the patch embeds local file "
    "paths/source lines; the share-form export drops it (Task 46)]\n"
    "+def test_arl_patch_scrubbed(): pass  [re-generate locally or export with "
    "--raw-local]\n"
)


_TEXT_SCRUB_MARKER = "[scrubbed at export (Task 46): free-text field stays local]"

# The closed grammar of ARL-authored evidence lines that receipts grade from.
# Share-form exports keep ONLY lines that fullmatch one of these — bounded
# charsets, no free text — so grading reproduces on re-import while anything
# else (imported/forged/free-text evidence) is dropped, fail-closed.
_SHARE_EVIDENCE_GRAMMAR = (
    re.compile(r"^step_id=[A-Za-z0-9._:-]{1,128}$"),
    re.compile(r"^retry_count=\d{1,6} additional attempts$"),
    re.compile(r"^total_attempts=\d{1,6}$"),
    re.compile(r"^step_cost_usd=[0-9.]{1,32}$"),
    re.compile(r"^step_error_class=[A-Za-z_]{1,64}$"),
    re.compile(r"^rule=(R1|R2)$"),
)


def _scrub_for_share(data: dict[str, Any]) -> dict[str, Any]:
    """Project a bundle dict to the SHARE form (Task 46 — the egress boundary).

    Drops every ``_RAW_CONTENT_METADATA_KEYS`` VALUE from step metadata (the key
    NAMES are disclosed under ``_scrubbed_keys`` — content-free by construction)
    and replaces a non-empty prescription ``patch`` (which is BUILT from those
    before/path/after values) with a content-free marker. Strictly an egress
    projection: local capture, storage, and render keep raw values per ADR-001
    Category 2 — the diffs the product ships depend on them.

    COPY, NEVER MUTATE: ``to_dict`` shares live inner objects (a step's
    ``metadata`` dict is the dataclass's own dict), so this builds replacement
    dicts instead of popping in place — an export must never alter the
    in-memory bundle (regression-pinned in tests/test_export_scrub.py).

    NOTE for future egress fields (Task 60 CLAUDE.md snapshots etc.): this
    function is THE chokepoint — new local-secret fields must be scrubbed here
    before any share path grows."""
    for step in data.get("steps", []):
        md = step.get("metadata")
        if isinstance(md, dict):
            dropped = sorted(k for k in md if k.lower() in _RAW_CONTENT_METADATA_KEYS)
            if dropped:
                kept = {
                    k: v for k, v in md.items() if k.lower() not in _RAW_CONTENT_METADATA_KEYS
                }
                kept["_scrubbed_keys"] = dropped
                step["metadata"] = kept
    # Codex Rule 8 re-review F-02 (CRITICAL): metadata + patch were not the only
    # raw-content carriers. The run's outcome_json (user/app-authored JSON) and
    # every free-text prescription field can carry arbitrary content — on an
    # IMPORTED bundle they are attacker-authored outright. The share form keeps
    # only the closed-grammar evidence lines receipts actually grade from
    # (fail-closed: a line that does not fullmatch the grammar is dropped and
    # counted), and scrubs the rest.
    run = data.get("run")
    if isinstance(run, dict) and run.get("outcome_json"):
        run["outcome_json"] = None
        run["_outcome_scrubbed"] = True
    for rx in data.get("prescriptions", []):
        if rx.get("patch"):
            rx["patch"] = _PATCH_SCRUB_MARKER
        for field in ("root_cause", "one_line_fix"):
            if rx.get(field):
                rx[field] = _TEXT_SCRUB_MARKER
        if rx.get("regression_test_template"):
            rx["regression_test_template"] = ""
        if rx.get("expected_impact"):
            rx["expected_impact"] = {}
        evidence = rx.get("evidence")
        if isinstance(evidence, list):
            kept_lines = [
                line
                for line in evidence
                if isinstance(line, str)
                and any(g.fullmatch(line.strip()) for g in _SHARE_EVIDENCE_GRAMMAR)
            ]
            dropped_n = len(evidence) - len(kept_lines)
            if dropped_n:
                kept_lines.append(f"[{dropped_n} evidence line(s) scrubbed at export]")
            rx["evidence"] = kept_lines
    return data


def write_trace(bundle: TraceBundle, path: Path, *, raw_local: bool = False) -> None:
    """Write *bundle* as JSON. DEFAULT is the share-safe scrubbed form (Rule 6:
    the safe form is the default; full fidelity is the explicit local opt-in)."""
    bundle.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bundle.to_dict()
    if raw_local:
        note = _EXPORT_NOTE_RAW_LOCAL
    else:
        note = _EXPORT_NOTE_SHARE
        data = _scrub_for_share(data)
    # Prepend the label without disturbing the round-trip: from_dict reads only
    # known keys, so this annotation re-imports as a harmless no-op.
    payload = {"_export_note": note, **data}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic_trace_dict(bundle: TraceBundle) -> dict[str, Any]:
    return json.loads(json.dumps(bundle.to_dict(), sort_keys=True))
