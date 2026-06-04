"""LR1 — outcome_json attach-point + L14 — success_label semantics.

LR1: a single nullable outcome_json column on runs, the B2 attach-point. Facts
only, NULL = unknown (the honest default), NEVER computed-on-read, inner shape
NOT frozen. Doing it now makes the first B2 attach pure-Python, not a migration.

L14: success_label is the adapter's provisional SELF-REPORT (a fact about what
the run CLAIMED), NOT a verdict. Documented in the RunRecord docstring; detectors
must not treat it as ground truth. (Rename to terminal_status is operator-
optional and NOT done — §6.) TDD red-first (Task 44, Phase 5).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_run_ledger.core.models import RunRecord, TraceBundle, TraceValidationError
from agent_run_ledger.core import models as models_mod
from agent_run_ledger.core.storage import connect, load_bundle, save_bundle


def _run(**kw) -> RunRecord:
    base = dict(
        id="run_o", workflow="w", framework="f", provider="p", model="m",
        started_at="2026-05-28T00:00:00Z", ended_at="2026-05-28T00:00:01Z",
        success_label="passed",
    )
    base.update(kw)
    return RunRecord(**base)


def test_outcome_json_defaults_to_none() -> None:
    assert _run().outcome_json is None


def test_outcome_json_accepts_valid_json() -> None:
    payload = json.dumps({"source": "human", "verdict": "correct"})
    assert _run(outcome_json=payload).outcome_json == payload


def test_outcome_json_rejects_malformed_json() -> None:
    with pytest.raises(TraceValidationError, match="outcome_json"):
        _run(outcome_json="{not json")


def test_outcome_json_persists(tmp_path: Path) -> None:
    from agent_run_ledger.core.models import StepRecord

    payload = json.dumps({"source": "external", "score": 0.9})
    run = _run(outcome_json=payload)
    step = StepRecord(
        id="s1", run_id="run_o", step_type="tool", name="n",
        started_at=run.started_at, ended_at=run.ended_at,
    )
    db = tmp_path / "ledger.sqlite"
    save_bundle(db, TraceBundle(run=run, steps=[step]))
    loaded = load_bundle(db, "run_o")
    assert loaded.run.outcome_json == payload
    with connect(db) as conn:
        cols = {c[1] for c in conn.execute("PRAGMA table_info(runs)")}
    assert "outcome_json" in cols


def test_outcome_json_is_one_column_not_three() -> None:
    # LR1 explicitly: ONE column, not outcome/outcome_source/outcome_at (that
    # would start designing the B2 attestation shape, which is forbidden).
    fields = set(RunRecord.__dataclass_fields__)
    assert "outcome_json" in fields
    assert "outcome_source" not in fields
    assert "outcome_at" not in fields


# --- L14: success_label semantics documented (self-report, not a verdict) -----

def test_success_label_documented_as_self_report() -> None:
    doc = (RunRecord.__doc__ or "") + models_mod.__doc__ if models_mod.__doc__ else (RunRecord.__doc__ or "")
    # the contract must be written somewhere a reader will see it
    src = Path(models_mod.__file__).read_text(encoding="utf-8")
    assert "self-report" in src.lower()
    assert "ground truth" in src.lower()
