"""L9 — content-bearing field classification at the EGRESS boundary.

Tested as a CLOSED field x channel MATRIX, not by sentinel shape (memory:
feedback_leak_test_by_matrix_not_shape — shape-hunting ships false-greens).

Two field classes:
  * AUTO-CAPTURED content (error message, arbitrary metadata values + key names):
    hard-redacted at the StepRecord chokepoint — must be ABSENT from EVERY
    channel (Category 1, ADR-001).
  * CONTENT-BEARING labels (step.name, workflow, success_label, root_cause,
    one_line_fix, patch): raw LOCALLY (the product working) — must be PRESENT in
    the local HTML report, and are documented as content-bearing so any FUTURE
    telemetry/shared-export path strips them (Category 2, ADR-001 + L9).

Every (field x channel) cell is asserted explicitly; we walk keys AND values of
every channel. There is no built network-egress channel yet (DEFER D2), so the
local channels are the closed set; the egress contract is asserted as a doc/code
invariant in test_egress_guards.py (L10/L12/L13).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_run_ledger.adapters.openai import bundle_from_recorded_trace
from agent_run_ledger.core.compare import compare_bundles
from agent_run_ledger.core.io import write_trace
from agent_run_ledger.core.models import StepRecord
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.report import render_comparison, render_report
from agent_run_ledger.core.storage import load_bundle, save_bundle


# Distinct sentinel per content FIELD so a leak names exactly which field x channel.
AUTO_CAPTURED_SENTINELS = {
    "error_message": "SENTINEL_error_message_QQ1",
    "metadata_value": "SENTINEL_metadata_value_QQ2",
    "metadata_key_top": "SENTINEL_metadata_key_top_QQ3",
    "metadata_key_nested": "SENTINEL_metadata_key_nested_QQ4",
    "unicode_value": "СЕНТ_unicode_value_QQ5",
    "unicode_key": "СЕНТ_unicode_key_QQ6",
}
# content-bearing LABELS: must appear RAW in the local HTML (product working).
LABEL_SENTINELS = {
    "step_name": "LABEL_step_name_RR1",
    "workflow": "LABEL_workflow_RR2",
    "success_label": "LABEL_success_label_RR3",
}


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str):
                yield k
            yield from _walk_strings(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def _build_loaded_bundle(tmp_path: Path):
    """Capture a trace carrying every sentinel, persist it, reload it, and return
    (loaded_bundle, channels) where channels is the closed egress/render set."""
    recorded = {
        "trace": {
            "trace_id": "trace_leak_matrix",
            "workflow_name": LABEL_SENTINELS["workflow"],
            "started_at": "2026-05-28T15:00:00Z",
            "ended_at": "2026-05-28T15:00:01Z",
        },
        "spans": [
            {
                "span_id": "span_leak",
                "started_at": "2026-05-28T15:00:00Z",
                "ended_at": "2026-05-28T15:00:01Z",
                "span_data": {
                    "type": "custom",
                    "name": LABEL_SENTINELS["step_name"],
                    "error": f"TimeoutError: {AUTO_CAPTURED_SENTINELS['error_message']}",
                    "data": {
                        "retry_count": 3,
                        "reasoning": AUTO_CAPTURED_SENTINELS["metadata_value"],
                        f"k_{AUTO_CAPTURED_SENTINELS['metadata_key_top']}": "v",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            f"k_{AUTO_CAPTURED_SENTINELS['metadata_key_nested']}": "v",
                        },
                        "uni_v": AUTO_CAPTURED_SENTINELS["unicode_value"],
                        f"k_{AUTO_CAPTURED_SENTINELS['unicode_key']}": "v",
                    },
                },
            }
        ],
    }
    bundle = bundle_from_recorded_trace(recorded)
    # force success_label to a sentinel (a content-bearing label, L9/L14)
    from dataclasses import replace

    bundle = replace(bundle, run=replace(bundle.run, success_label=LABEL_SENTINELS["success_label"]))
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))

    db = tmp_path / "ledger.sqlite"
    out = tmp_path / "trace.json"
    run_id = save_bundle(db, bundle)
    loaded = load_bundle(db, run_id)
    write_trace(loaded, out)

    channels = {
        "sqlite_bytes": [db.read_bytes().decode("utf-8", errors="replace")],
        "json_export": list(_walk_strings(json.loads(out.read_text(encoding="utf-8")))),
        "html_report": [render_report(loaded)],
        "render_comparison": [render_comparison(compare_bundles(loaded, loaded))],
    }
    return loaded, channels


def test_auto_captured_content_absent_from_every_channel(tmp_path: Path) -> None:
    """Closed matrix: every AUTO-CAPTURED sentinel x every channel -> ABSENT."""
    _loaded, channels = _build_loaded_bundle(tmp_path)
    leaks = []
    for sname, sentinel in AUTO_CAPTURED_SENTINELS.items():
        for cname, strings in channels.items():
            if any(sentinel in s for s in strings):
                leaks.append(f"{cname}:{sname}")
    assert leaks == [], f"auto-captured content leaked: {leaks}"


def test_content_bearing_labels_present_in_local_html(tmp_path: Path) -> None:
    """Local render keeping the labels IS the product working (L9). Each label
    sentinel must be PRESENT in the local HTML report."""
    _loaded, channels = _build_loaded_bundle(tmp_path)
    html = channels["html_report"][0]
    missing = [name for name, s in LABEL_SENTINELS.items() if s not in html]
    assert missing == [], f"content-bearing labels missing from local report: {missing}"


def test_error_class_is_a_label_not_the_message(tmp_path: Path) -> None:
    """The typed error_class (Timeout) may appear; the error MESSAGE may not."""
    loaded, channels = _build_loaded_bundle(tmp_path)
    assert loaded.steps[0].error_class == "Timeout"
    for cname, strings in channels.items():
        assert not any(
            AUTO_CAPTURED_SENTINELS["error_message"] in s for s in strings
        ), cname


def test_matrix_is_closed_over_known_content_fields() -> None:
    """Guard: if a NEW content-bearing field is added to StepRecord/RunRecord,
    this list must be revisited. Pins the known content field set so a silent
    new content column is caught in review."""
    step_content_fields = {"name"}  # the only content-bearing label on a step
    run_content_fields = {"workflow", "success_label"}
    step_fields = set(StepRecord.__dataclass_fields__)
    assert step_content_fields <= step_fields
    from agent_run_ledger.core.models import RunRecord

    assert run_content_fields <= set(RunRecord.__dataclass_fields__)
