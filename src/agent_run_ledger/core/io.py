from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_run_ledger.core.models import TraceBundle, TraceValidationError

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
_EXPORT_NOTE = (
    "raw facts export (local): one immutable step per captured span. retry_count "
    "here is the per-span raw value (0 for un-collapsed attempts); the derived "
    "retry view (retry_count=N) that prescriptions cite is computed ON READ and is "
    "not persisted. Allowed metadata values are exported verbatim (raw local "
    "export, not a remote leak)."
)


def write_trace(bundle: TraceBundle, path: Path) -> None:
    bundle.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bundle.to_dict()
    # Prepend the label without disturbing the round-trip: from_dict reads only
    # known keys, so this annotation re-imports as a harmless no-op.
    payload = {"_export_note": _EXPORT_NOTE, **data}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic_trace_dict(bundle: TraceBundle) -> dict[str, Any]:
    return json.loads(json.dumps(bundle.to_dict(), sort_keys=True))
