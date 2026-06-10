"""Repair-target metadata is LOCAL-ONLY and disclosed (full-suite-audit P2, 2026-06-11).

The allowlisted repair-target keys (``before`` / ``after`` / ``path`` /
``arl_patch_target`` / ``retry_budget_patch_target``) are APP-SUPPLIED patch
targets and are deliberately kept verbatim — they carry the file/line content a
receipt's diff needs. They are NOT auto-captured trace content (those are stripped;
see test_leak_matrix.py), and they only exist when the app instruments its trace
(the L2 path), so a normal captured run never has them.

The audit's correct point: these values DO round-trip into the local sqlite ledger
and the local ``arl export`` JSON, so a user who SHARES that export ships the
content. That is disclosed (io.py export note + receipt.py "NOT remote-egress-safe
until Task 46"). These tests pin two invariants so the disclosure stays true:
  1. the repair-target value is treated as a fact-key but is NOT mistaken for an
     auto-captured field that the leak-matrix expects to be stripped;
  2. the set of "raw-content-bearing, local-only, export-travels" keys is CLOSED —
     a new such key forces this guard to be revisited (the same discipline as
     test_leak_matrix.py::test_matrix_is_closed_over_known_content_fields).
"""

from __future__ import annotations

from agent_run_ledger.core.models import (
    _ALLOWED_METADATA_KEYS,
    _BOOLEAN_FACT_KEYS,
    _RAW_CONTENT_METADATA_KEYS,
    sanitize_metadata,
)

# Single source of truth lives in models; this test pins the disclosure invariants.
_RAW_CONTENT_REPAIR_KEYS = _RAW_CONTENT_METADATA_KEYS


def test_repair_target_values_are_kept_verbatim_locally() -> None:
    """A patch target's value is preserved (the diff needs it) — sanitize keeps it,
    unlike a stripped auto-captured field."""
    cleaned = sanitize_metadata({"before": "MAX_RETRIES = 10", "path": "src/crm.py"})
    assert cleaned["before"] == "MAX_RETRIES = 10"
    assert cleaned["path"] == "src/crm.py"


def test_boolean_fact_keys_and_raw_content_keys_are_disjoint() -> None:
    """The Task 58 content-free fact keys (coerced to bool) must NOT overlap the
    raw-content repair keys (kept verbatim) — a key can't be both."""
    assert _BOOLEAN_FACT_KEYS.isdisjoint(_RAW_CONTENT_REPAIR_KEYS)


def test_raw_content_repair_keys_are_a_closed_known_set() -> None:
    """Guard: every raw-content repair key is allowlisted, and the allowlist adds
    no NEW raw-content-shaped key without this guard being revisited. If this fails
    because a new allowlisted key was added, decide explicitly whether its value is
    content (add here + update the disclosure + Task 46 scrub) or a bounded
    fact/number (no action)."""
    assert _RAW_CONTENT_REPAIR_KEYS <= _ALLOWED_METADATA_KEYS
    # The allowlist's known non-content keys: bounded facts (bool/number/label).
    known_non_content = _BOOLEAN_FACT_KEYS | {
        "cost_usd", "input_tokens", "output_tokens", "total_tokens", "max_tokens",
        "retry_count", "model", "severity", "step_type", "usage",
    }
    unclassified = _ALLOWED_METADATA_KEYS - _RAW_CONTENT_REPAIR_KEYS - known_non_content
    assert not unclassified, (
        "new allowlisted metadata key(s) are unclassified — decide if the VALUE is "
        f"raw content (local-only, export-travels, needs Task 46 scrub) or a bounded "
        f"fact: {unclassified}"
    )
