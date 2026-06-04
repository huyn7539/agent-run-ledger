"""L8 — typed error_class, drop the message at the chokepoint.

A bounded, typed label derived from the error class name; the raw message is
dropped at the StepRecord construction chokepoint so nothing leaks at rest or on
egress. The existing redaction chokepoint (sanitize_error -> 'details redacted')
is KEPT intact; error_class is ADDED alongside (operator decision). The wedge's
retry/cost prescription reads error_class for severity. TDD red-first
(Task 44, Phase 4).
"""

from __future__ import annotations

from pathlib import Path

from agent_run_ledger.adapters.openai import bundle_from_recorded_trace
from agent_run_ledger.core.io import load_trace
from agent_run_ledger.core.models import StepRecord, classify_error
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.storage import connect, load_bundle, save_bundle


def _step(**kw) -> StepRecord:
    base = dict(
        id="s1", run_id="r", step_type="tool", name="n",
        started_at="2026-05-28T00:00:00Z", ended_at="2026-05-28T00:00:01Z",
    )
    base.update(kw)
    return StepRecord(**base)


# --- classify_error maps to a bounded label, never the message ----------------

def test_classify_error_extracts_class_drops_message() -> None:
    label = classify_error("TimeoutError: CRM lookup exceeded 5s timeout PROMPT_SECRET")
    assert label == "Timeout"
    # the message (and any secret in it) must NOT appear in the label
    assert "PROMPT_SECRET" not in label
    assert "CRM" not in label


def test_classify_error_from_exception_type() -> None:
    assert classify_error(TimeoutError("boom")) == "Timeout"
    assert classify_error(ValueError("x")) == "Validation"


def test_classify_error_unknown_bucketed_as_other() -> None:
    assert classify_error("WeirdCustomError: detail") == "Other"


def test_classify_error_none_is_none() -> None:
    assert classify_error(None) is None


def test_imported_error_class_cannot_inject_content() -> None:
    """Codex adversarial finding (2026-05-29): an explicit error_class on the
    IMPORT path must NOT bypass classify_error — otherwise a caller injects raw
    content into the bounded label, which then reaches every egress channel.
    error_class must ALWAYS be a bounded label from the closed vocabulary."""
    from agent_run_ledger.core.models import ERROR_CLASSES

    s = StepRecord.from_dict(
        {
            "id": "s", "type": "tool", "name": "n",
            "started_at": "2026-05-28T00:00:00Z", "ended_at": "2026-05-28T00:00:01Z",
            "error_class": "LEAKED_PROMPT_secret_customer_data_12345",
        },
        "r",
    )
    # the injected content is NOT stored verbatim; it is bounded to the vocab
    assert s.error_class in ERROR_CLASSES
    assert "LEAKED_PROMPT" not in (s.error_class or "")
    # a legitimate bounded label still round-trips
    s2 = StepRecord.from_dict(
        {
            "id": "s2", "type": "tool", "name": "n",
            "started_at": "2026-05-28T00:00:00Z", "ended_at": "2026-05-28T00:00:01Z",
            "error_class": "Timeout",
        },
        "r",
    )
    assert s2.error_class == "Timeout"


# --- error_class is a real StepRecord field, persisted ------------------------

def test_step_error_class_field_and_roundtrip(tmp_path: Path) -> None:
    from agent_run_ledger.core.models import RunRecord, TraceBundle

    run = RunRecord(
        id="r", workflow="w", framework="f", provider="p", model="m",
        started_at="2026-05-28T00:00:00Z", ended_at="2026-05-28T00:00:01Z",
        success_label="failed",
    )
    step = _step(run_id="r", error_class="Timeout")
    db = tmp_path / "ledger.sqlite"
    save_bundle(db, TraceBundle(run=run, steps=[step]))
    loaded = load_bundle(db, "r")
    assert loaded.steps[0].error_class == "Timeout"
    with connect(db) as conn:
        cols = {c[1] for c in conn.execute("PRAGMA table_info(steps)")}
    assert "error_class" in cols


# --- the adapter chokepoint stamps error_class, drops the message -------------

def test_adapter_captures_error_class_not_message(tmp_path: Path) -> None:
    recorded = {
        "trace": {
            "trace_id": "trace_ec",
            "workflow_name": "ec",
            "started_at": "2026-05-28T15:00:00Z",
            "ended_at": "2026-05-28T15:00:01Z",
        },
        "spans": [
            {
                "span_id": "span_ec",
                "started_at": "2026-05-28T15:00:00Z",
                "ended_at": "2026-05-28T15:00:01Z",
                "span_data": {
                    "type": "custom",
                    "name": "ec.step",
                    "error": "TimeoutError: secret prompt LEAK_SENTINEL_xyz",
                    "data": {"retry_count": 3},
                },
            }
        ],
    }
    bundle = bundle_from_recorded_trace(recorded)
    step = bundle.steps[0]
    # a known error TYPE survives capture (pinned so it can't silently regress)
    assert step.error_class == "Timeout"
    # the raw message / any secret never reaches error_class
    assert "LEAK_SENTINEL_xyz" not in (step.error_class or "")
    # the existing redaction chokepoint is still intact on step.error
    assert step.error in (None, "details redacted")


# --- prescription severity now reads error_class ------------------------------

def test_severity_high_when_error_class_present() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    # golden's demo.flaky_tool step has a TimeoutError -> error_class Timeout
    rx = analyze_bundle(bundle)
    assert rx[0].severity == "high"
