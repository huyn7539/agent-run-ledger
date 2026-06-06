"""compare_bundles must compute cost_delta from cost_on_read, NOT the cached total.

LR2: total_cost_usd is a CACHE, not authoritative; the source of truth is
cost.cost_on_read() (computed on read from token facts x the stub price table).
report + list already render cost_on_read. compare was the one path still
subtracting the cached run.total_cost_usd, so a stale cache could make compare
DISAGREE with report/list — including the SIGN (direction) of the delta.

This test constructs two bundles whose cached totals are deliberately inverted
relative to what cost_on_read computes from real gpt-4o tokens, so the cached
delta and the on-read delta have OPPOSITE signs. compare_bundles must match the
on-read delta. TDD red-first.
"""

from __future__ import annotations

import pytest

from agent_run_ledger.core.compare import compare_bundles
from agent_run_ledger.core.cost import cost_on_read
from agent_run_ledger.core.models import RunRecord, StepRecord, TraceBundle

# gpt-4o stub rates (USD per 1K): input=0.0025, output=0.01, cached=0.00125.
# left  real tokens (small):  in=1000 out=1000 -> 0.0025 + 0.01   = 0.0125
# right real tokens (large):  in=10000 out=10000 -> 0.025 + 0.10  = 0.125
# on-read delta = right - left = 0.125 - 0.0125 = +0.1125 (POSITIVE: right pricier).
#
# Cached totals are inverted so the cached delta has the WRONG sign:
# left.total_cost_usd  = 9.99 (high), right.total_cost_usd = 0.01 (low)
# cached delta = right.total - left.total = 0.01 - 9.99 = -9.98 (NEGATIVE).
_LEFT_TOKENS = (1000, 1000)
_RIGHT_TOKENS = (10000, 10000)
_LEFT_CACHED_TOTAL = 9.99
_RIGHT_CACHED_TOTAL = 0.01


def _bundle(run_id: str, tokens: tuple[int, int], cached_total: float) -> TraceBundle:
    """A valid single-step gpt-4o bundle with REAL tokens and a chosen cached total.

    Built via the dataclass constructors (not from_dict) so the deliberately
    inconsistent cached total_cost_usd survives — from_dict runs
    _with_computed_totals + validate(), which would normalize/clobber it.
    """
    input_tokens, output_tokens = tokens
    run = RunRecord(
        id=run_id,
        workflow="w",
        framework="f",
        provider="openai",
        model="gpt-4o",  # PRICED model -> cost_on_read returns real nonzero numbers.
        started_at="2026-06-06T00:00:00Z",
        ended_at="2026-06-06T00:00:01Z",
        success_label="passed",
        total_cost_usd=cached_total,  # the CACHE, deliberately wrong.
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
    )
    step = StepRecord(
        id=f"{run_id}_s1",
        run_id=run_id,
        step_type="model",
        name="call",
        started_at=run.started_at,
        ended_at=run.ended_at,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return TraceBundle(run=run, steps=[step])


def test_compare_cost_delta_uses_cost_on_read_not_stale_cache() -> None:
    left = _bundle("run_left", _LEFT_TOKENS, _LEFT_CACHED_TOTAL)
    right = _bundle("run_right", _RIGHT_TOKENS, _RIGHT_CACHED_TOTAL)

    # The on-read truth: right is the pricier run, so the delta is POSITIVE.
    on_read_delta = cost_on_read(right) - cost_on_read(left)
    assert on_read_delta == pytest.approx(0.1125)
    assert on_read_delta > 0

    # The stale cache says the opposite (negative) — the exact disagreement,
    # including direction, that this fix exists to prevent.
    cached_delta = right.run.total_cost_usd - left.run.total_cost_usd
    assert cached_delta < 0  # -9.98: wrong sign vs the on-read truth.

    comparison = compare_bundles(left, right)

    # compare must match the on-read delta (correct sign), NOT the cached delta.
    assert comparison.cost_delta_usd == pytest.approx(on_read_delta)
    assert comparison.cost_delta_usd > 0
