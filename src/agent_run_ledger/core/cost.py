"""Cost is a derived JUDGMENT, computed on read (L7 / LR2).

The base record stores the FACTS that determine cost (token counts, the
provider-reported cost when available, billing mode); it does NOT treat any
adapter-supplied total as authoritative. Computing on read means a price-table
update never corrupts the stored corpus — the facts are permanent, the judgment
recomputes.

SCOPE (deliberate one-brick reading of "source of truth = compute-on-read"):
this is the cost *seam*, not a pricing *engine*. The price table here is a
minimal, versioned STUB — enough to make the seam real and the smoke test
meaningful. A full multi-model / multi-region pricing table is the cost PRODUCT,
which is post-gate. Do NOT grow this into a pricing engine here.

Precedence for a run's cost (made explicit in Task 53 F1):
  1. The sum of per-step ``provider_reported_cost_usd`` when ANY step reports it
     (the FACT — what the provider said it charged).
  2. Else a compute from token facts x the stubbed price table for the model:
     per-step sums when attribution is COMPLETE (step token sums cover the run
     totals — preserves cached-input discounts and reasoning billing); the RUN
     totals when per-step attribution is missing OR partial (steps account for
     strictly fewer tokens than the run totals — pricing only the attributed
     fraction would silently underprice the run).
  3. Else the cached ``total_cost_usd`` on the run (last-resort fallback, e.g.
     an unknown model).
"""

from __future__ import annotations

from .models import TraceBundle

# Minimal, versioned STUB price table. USD per 1K tokens: (input, output, cached_input).
# Enough to prove the seam, NOT a production pricing source. Replace/extend with
# the real product later (post-gate).
PRICE_TABLE_VERSION = "stub-2026-05-29"

_PRICES_PER_1K: dict[str, tuple[float, float, float]] = {
    "gpt-4o": (0.0025, 0.01, 0.00125),
    "gpt-4o-mini": (0.00015, 0.0006, 0.000075),
    "gpt-4.1-mini": (0.0004, 0.0016, 0.0001),
    "gpt-5.4-nano": (0.00005, 0.0004, 0.000025),
    "demo-model": (0.0, 0.0, 0.0),
}


def _compute_from_tokens(bundle: TraceBundle) -> float | None:
    """Compute cost from token facts x the stub price table.

    Returns None if the model is unknown to the stub table, so the caller falls
    back to the cached value rather than reporting a misleading $0.
    """
    price = _PRICES_PER_1K.get(bundle.run.model)
    if price is None:
        return None
    in_per_1k, out_per_1k, cached_per_1k = price
    total = 0.0
    for s in bundle.steps:
        billable_input = max(0, s.input_tokens - s.cached_input_tokens)
        total += (billable_input / 1000.0) * in_per_1k
        total += (s.cached_input_tokens / 1000.0) * cached_per_1k
        # reasoning tokens are billed at the output rate by current providers.
        total += ((s.output_tokens + s.reasoning_tokens) / 1000.0) * out_per_1k
    # Run-level token fallback (provider-neutral; keyed on the FACT pattern, not any
    # adapter). Price from the RUN totals instead of the per-step sum whenever the
    # per-step attribution is INCOMPLETE relative to the run totals (Task 53 F1):
    #   * NO step carries tokens but the run does (cumulative-total-only providers) —
    #     the original case; pricing per-step yields a misleading $0;
    #   * SOME steps carry tokens but strictly fewer than the run totals (a future
    #     mixed-attribution adapter) — pricing only the attributed fraction silently
    #     discards the run remainder and underprices the run.
    # Run totals carry no cached/reasoning breakdown, so this prices at the flat
    # input/output rates. FULL attribution (step sums cover the run totals) keeps
    # per-step pricing so the cached-input discount and reasoning billing survive.
    attributed_in = sum(s.input_tokens for s in bundle.steps)
    attributed_out = sum(s.output_tokens for s in bundle.steps)
    run_in = max(0, bundle.run.total_input_tokens)
    run_out = max(0, bundle.run.total_output_tokens)
    if (run_in or run_out) and (attributed_in < run_in or attributed_out < run_out):
        return round((run_in / 1000.0) * in_per_1k + (run_out / 1000.0) * out_per_1k, 8)
    return round(total, 8)


def cost_on_read(bundle: TraceBundle) -> float:
    """Return the authoritative cost for *bundle*, computed on read (L7)."""
    reported = [
        s.provider_reported_cost_usd
        for s in bundle.steps
        if s.provider_reported_cost_usd is not None
    ]
    if reported:
        return round(sum(reported), 8)
    computed = _compute_from_tokens(bundle)
    if computed is not None:
        return computed
    return bundle.run.total_cost_usd


def cost_is_priced(bundle: TraceBundle) -> bool:
    """True iff the cost was actually derived (provider-reported OR model-priced),
    vs. an unpriced fallback. Lets the render layer show "unpriced (model X)" instead
    of a misleading bare $0.00 when the model is unknown to the stub table but real
    tokens exist (A2). A genuine $0 (priced model, zero tokens) is still 'priced'."""
    if any(s.provider_reported_cost_usd is not None for s in bundle.steps):
        return True
    return _compute_from_tokens(bundle) is not None


def cost_display(bundle: TraceBundle) -> str:
    """Human-facing cost string: a dollar figure when priced, else an explicit
    'unpriced' disclosure naming the unknown model — never a misleading bare $0 on a
    real run with real tokens (A2)."""
    if cost_is_priced(bundle):
        return f"${cost_on_read(bundle):.6f}"
    has_tokens = bool(bundle.run.total_input_tokens or bundle.run.total_output_tokens)
    if has_tokens:
        return f"unpriced (model {bundle.run.model!r} not in price table; tokens recorded)"
    return "$0.000000"
