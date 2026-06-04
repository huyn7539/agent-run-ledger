"""L4 — call-graph linkage: parent_step_id (NULL = root span) + span_kind.

The OpenAI Agents SDK emits span TREES; the flat steps table loses topology once
flattened. parent_step_id is the only structural fact missing — capture it as a
fact, keep variable payload in metadata (do NOT normalize). TDD red-first
(Task 44, Phase 2).
"""

from __future__ import annotations

from pathlib import Path

from agent_run_ledger.core.models import RunRecord, StepRecord, TraceBundle
from agent_run_ledger.core.storage import connect, load_bundle, save_bundle


def _run() -> RunRecord:
    return RunRecord(
        id="run_cg",
        workflow="w",
        framework="f",
        provider="openai",
        model="gpt-4o-mini",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:03Z",
        success_label="passed",
    )


def _step(sid: str, parent: str | None = None, kind: str | None = None) -> StepRecord:
    return StepRecord(
        id=sid,
        run_id="run_cg",
        step_type="tool",
        name=f"name_{sid}",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
        parent_step_id=parent,
        span_kind=kind,
    )


def test_step_callgraph_fields_default_to_none() -> None:
    s = _step("s1")
    assert s.parent_step_id is None  # root
    assert s.span_kind is None


def test_callgraph_survives_db_roundtrip(tmp_path: Path) -> None:
    bundle = TraceBundle(
        run=_run(),
        steps=[
            _step("agent_1", parent=None, kind="agent"),
            _step("llm_1", parent="agent_1", kind="llm"),
            _step("tool_1", parent="agent_1", kind="tool"),
        ],
    )
    db = tmp_path / "ledger.sqlite"
    save_bundle(db, bundle)
    loaded = load_bundle(db, "run_cg")

    by_id = {s.id: s for s in loaded.steps}
    assert by_id["agent_1"].parent_step_id is None
    assert by_id["agent_1"].span_kind == "agent"
    assert by_id["llm_1"].parent_step_id == "agent_1"
    assert by_id["tool_1"].parent_step_id == "agent_1"
    assert by_id["tool_1"].span_kind == "tool"

    with connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(steps)")}
    assert {"parent_step_id", "span_kind"} <= cols


def test_callgraph_survives_dict_roundtrip() -> None:
    bundle = TraceBundle(
        run=_run(),
        steps=[_step("agent_1", kind="agent"), _step("tool_1", parent="agent_1", kind="tool")],
    )
    rt = TraceBundle.from_dict(bundle.to_dict())
    by_id = {s.id: s for s in rt.steps}
    assert by_id["tool_1"].parent_step_id == "agent_1"
    assert by_id["agent_1"].span_kind == "agent"
