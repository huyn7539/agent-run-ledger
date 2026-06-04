"""L5 — provenance_hash: lock the canonicalization, not just the column.

sha256 over a sorted-key JSON of EXACTLY the immutable facts (id, workflow,
model, provider, billing_mode, prompt_hash, config_hash, started_at, ended_at,
ordered (step.id, parent_step_id, step_type, input_tokens, output_tokens)).
EXCLUDES total_cost_usd and every derived/judgment field. Algorithm-prefixed,
computed locally at capture. Pinned by a golden digest. TDD red-first
(Task 44, Phase 3).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_run_ledger.core.io import load_trace
from agent_run_ledger.core.models import RunRecord, StepRecord, TraceBundle
from agent_run_ledger.core.provenance import compute_provenance_hash
from agent_run_ledger.core.storage import load_bundle, save_bundle


def _bundle() -> TraceBundle:
    run = RunRecord(
        id="run_prov",
        workflow="support-agent",
        framework="openai-agents-python",
        provider="openai",
        model="gpt-4o-mini",
        billing_mode="pay_per_use",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:05Z",
        success_label="passed",
        prompt_hash="",
        config_hash="",
    )
    steps = [
        StepRecord(
            id="s1", run_id="run_prov", step_type="agent", name="plan", span_kind="agent",
            started_at="2026-05-28T00:00:00Z", ended_at="2026-05-28T00:00:02Z",
            input_tokens=100, output_tokens=40,
        ),
        StepRecord(
            id="s2", run_id="run_prov", step_type="tool", name="lookup", parent_step_id="s1",
            started_at="2026-05-28T00:00:02Z", ended_at="2026-05-28T00:00:05Z",
            input_tokens=50, output_tokens=10,
        ),
    ]
    return TraceBundle(run=run, steps=steps)


def test_provenance_hash_is_algorithm_prefixed() -> None:
    h = compute_provenance_hash(_bundle())
    assert h.startswith("sha256:")
    assert len(h.split(":", 1)[1]) == 64  # sha256 hex


def test_provenance_hash_is_deterministic() -> None:
    assert compute_provenance_hash(_bundle()) == compute_provenance_hash(_bundle())


def test_provenance_hash_excludes_cost_and_derived() -> None:
    base = _bundle()
    # changing cost / latency / cached-token / cost cache must NOT change the hash
    mutated = replace(
        base,
        run=replace(base.run, total_cost_usd=9999.0, total_latency_ms=12345),
        steps=[
            replace(base.steps[0], cost_usd=5.0, provider_reported_cost_usd=7.0,
                    cached_input_tokens=99, reasoning_tokens=88),
            base.steps[1],
        ],
    )
    assert compute_provenance_hash(mutated) == compute_provenance_hash(base)


def test_provenance_hash_changes_on_immutable_fact_change() -> None:
    base = _bundle()
    # changing an immutable fact (model) MUST change the hash
    changed = replace(base, run=replace(base.run, model="gpt-4o"))
    assert compute_provenance_hash(changed) != compute_provenance_hash(base)
    # changing the call-graph edge MUST change the hash
    edge = replace(base, steps=[base.steps[0], replace(base.steps[1], parent_step_id=None)])
    assert compute_provenance_hash(edge) != compute_provenance_hash(base)


def test_provenance_hash_step_order_independent() -> None:
    # steps are ordered deterministically (started_at, id) before hashing, so
    # insertion order does not change the digest
    base = _bundle()
    reordered = replace(base, steps=list(reversed(base.steps)))
    assert compute_provenance_hash(reordered) == compute_provenance_hash(base)


def test_provenance_hash_golden() -> None:
    # GOLDEN: pins the exact canonicalization. If this digest changes, the
    # preimage definition changed — that is a breaking provenance change and
    # must be a deliberate, reviewed event (algorithm prefix would bump too).
    assert compute_provenance_hash(_bundle()) == (
        "sha256:5476f154b07afa2291b7d0c654fe903458da78c5a9b8ff01eb32d1f0e58c08a7"
    )


def test_provenance_hash_persists(tmp_path: Path) -> None:
    bundle = _bundle()
    h = compute_provenance_hash(bundle)
    stamped = replace(bundle, run=replace(bundle.run, provenance_hash=h))
    db = tmp_path / "ledger.sqlite"
    save_bundle(db, stamped)
    assert load_bundle(db, "run_prov").run.provenance_hash == h


def test_provenance_hash_local_no_network() -> None:
    # source-agnostic + local: a bundle from a fixture also hashes cleanly
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    h = compute_provenance_hash(bundle)
    assert h.startswith("sha256:")
