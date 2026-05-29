"""L7 — cost is a derived JUDGMENT computed on read; its inputs are FACTS.

runs: billing_mode (closed enum) + price_table_version.
steps: cached_input_tokens, reasoning_tokens (observed facts), provider_reported_cost_usd (the cost FACT).
cost_on_read(): provider-reported sum -> stub price table -> cached fallback.
The stored total_cost_usd / step.cost_usd are a CACHE, not authoritative.
TDD red-first (Task 44, Phase 2).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_run_ledger.core.cost import PRICE_TABLE_VERSION, cost_on_read
from agent_run_ledger.core.io import load_trace
from agent_run_ledger.core.models import BILLING_MODES, RunRecord, StepRecord, TraceBundle
from agent_run_ledger.core.storage import connect, load_bundle, save_bundle


def _run(**kw) -> RunRecord:
    base = dict(
        id="run_cost",
        workflow="w",
        framework="f",
        provider="openai",
        model="gpt-4o-mini",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
        success_label="passed",
    )
    base.update(kw)
    return RunRecord(**base)


def _step(**kw) -> StepRecord:
    base = dict(
        id="s1",
        run_id="run_cost",
        step_type="model",
        name="plan",
        started_at="2026-05-28T00:00:00Z",
        ended_at="2026-05-28T00:00:01Z",
    )
    base.update(kw)
    return StepRecord(**base)


# --- new fact fields exist with safe defaults ---------------------------------

def test_billing_mode_default_and_enum() -> None:
    assert _run().billing_mode == "unknown"
    assert set(BILLING_MODES) == {
        "pay_per_use", "subscription", "enterprise_contract", "local", "unknown"
    }


def test_billing_mode_rejects_unknown_member() -> None:
    with pytest.raises(ValueError, match="billing_mode"):
        _run(billing_mode="freemium")


def test_step_cost_fact_fields_default() -> None:
    s = _step()
    assert s.cached_input_tokens == 0
    assert s.reasoning_tokens == 0
    assert s.provider_reported_cost_usd is None


def test_price_table_version_nullable() -> None:
    assert _run().price_table_version is None
    assert _run(price_table_version=PRICE_TABLE_VERSION).price_table_version == PRICE_TABLE_VERSION


# --- cost_on_read precedence --------------------------------------------------

def test_cost_on_read_prefers_provider_reported() -> None:
    # provider reported the cost FACT -> that wins over any token computation
    bundle = TraceBundle(
        run=_run(),
        steps=[
            _step(id="s1", input_tokens=1000, output_tokens=1000, provider_reported_cost_usd=0.5),
            _step(id="s2", input_tokens=9999, output_tokens=9999, provider_reported_cost_usd=0.25),
        ],
    )
    assert cost_on_read(bundle) == pytest.approx(0.75)


def test_cost_on_read_falls_back_to_stub_price_table() -> None:
    # no provider-reported cost -> compute from token facts x stub table
    bundle = TraceBundle(
        run=_run(model="gpt-4o-mini"),
        steps=[_step(input_tokens=1000, output_tokens=1000)],
    )
    # gpt-4o-mini stub: input 0.00015/1k, output 0.0006/1k -> 0.00015 + 0.0006
    assert cost_on_read(bundle) == pytest.approx(0.00075)


def test_cost_on_read_cached_tokens_discounted() -> None:
    # cached_input_tokens priced at the cheaper cached rate, not full input rate
    full = TraceBundle(run=_run(), steps=[_step(input_tokens=1000, output_tokens=0)])
    cached = TraceBundle(
        run=_run(), steps=[_step(input_tokens=1000, cached_input_tokens=1000, output_tokens=0)]
    )
    assert cost_on_read(cached) < cost_on_read(full)


def test_cost_on_read_unknown_model_falls_back_to_cache() -> None:
    bundle = TraceBundle(
        run=_run(model="mystery-model", total_cost_usd=0.0),
        steps=[_step(input_tokens=1000, output_tokens=1000, cost_usd=0.0)],
    )
    # bundle.run.total_cost_usd is the recomputed cache (step cost sum here) -> 0.0
    assert cost_on_read(bundle) == 0.0


# --- persistence of the new fact fields ---------------------------------------

def test_cost_facts_survive_db_roundtrip(tmp_path: Path) -> None:
    bundle = TraceBundle(
        run=_run(billing_mode="pay_per_use", price_table_version="v1"),
        steps=[
            _step(
                input_tokens=100,
                output_tokens=50,
                cached_input_tokens=40,
                reasoning_tokens=12,
                provider_reported_cost_usd=0.013,
            )
        ],
    )
    db = tmp_path / "ledger.sqlite"
    save_bundle(db, bundle)
    loaded = load_bundle(db, bundle.run.id)

    assert loaded.run.billing_mode == "pay_per_use"
    assert loaded.run.price_table_version == "v1"
    assert loaded.steps[0].cached_input_tokens == 40
    assert loaded.steps[0].reasoning_tokens == 12
    assert loaded.steps[0].provider_reported_cost_usd == pytest.approx(0.013)
    with connect(db) as conn:
        rcols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        scols = {r[1] for r in conn.execute("PRAGMA table_info(steps)")}
    assert {"billing_mode", "price_table_version"} <= rcols
    assert {"cached_input_tokens", "reasoning_tokens", "provider_reported_cost_usd"} <= scols


def test_cost_facts_survive_dict_roundtrip() -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    rt = TraceBundle.from_dict(bundle.to_dict())
    assert rt.run.billing_mode == bundle.run.billing_mode
    assert rt.steps[0].cached_input_tokens == bundle.steps[0].cached_input_tokens


def test_report_shows_cost_on_read_not_cached_total() -> None:
    # provider-reported cost is the FACT; the report must show the computed-on-
    # read value, not a stale/misleading cached total_cost_usd.
    from agent_run_ledger.core.report import render_report

    bundle = TraceBundle(
        run=_run(model="gpt-4o-mini", total_cost_usd=999.99),  # bogus cache
        steps=[_step(input_tokens=0, output_tokens=0, provider_reported_cost_usd=0.42)],
    )
    html = render_report(bundle)
    assert "0.42" in html
    assert "999.99" not in html
