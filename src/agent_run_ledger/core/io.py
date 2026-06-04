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


def load_trace(path: Path) -> TraceBundle:
    """Parse an untrusted trace file into a TraceBundle, defensively.

    Every field is treated as inert string/scalar data — nothing in a trace is
    ever evaluated or executed (json.loads only produces data). Size and depth are
    bounded; malformed or non-object input raises a typed TraceParseError."""
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
    try:
        return TraceBundle.from_dict(data)
    except TraceValidationError as exc:
        raise TraceParseError(f"invalid trace bundle: {exc}") from exc


def write_trace(bundle: TraceBundle, path: Path) -> None:
    bundle.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic_trace_dict(bundle: TraceBundle) -> dict[str, Any]:
    return json.loads(json.dumps(bundle.to_dict(), sort_keys=True))
