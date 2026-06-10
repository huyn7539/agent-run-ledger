"""compare discloses 'unpriced' consistently with list-runs/report (Task 57).

A confident cost_delta on a run whose model isn't in the price table CONTRADICTS
what list-runs and report disclose ('unpriced (model X ...)'). The honesty brand
requires the three surfaces to agree. This surfaced live in the README quickstart:
the demo bundles use an unpriced model, so `arl compare` showed a dollar delta
while `arl list-runs` said unpriced.
"""

from __future__ import annotations

from agent_run_ledger.core.compare import compare_bundles
from agent_run_ledger.core.cost import cost_is_priced
from agent_run_ledger.core.demo import load_demo_bundle
from agent_run_ledger.core.models import TraceBundle


def test_compare_discloses_unpriced_when_a_side_is_unpriced() -> None:
    """Both demo bundles use 'demo-mini' (not in the stub price table) — compare
    must disclose unpriced, not a confident delta."""
    left = load_demo_bundle("retry-loop")
    right = load_demo_bundle("clean")
    assert not cost_is_priced(left)  # precondition: demo model is unpriced
    cmp = compare_bundles(left, right)
    assert cmp.cost_delta_priced is False
    rows = dict(cmp.to_rows())
    assert "unpriced" in rows["cost_delta_usd"]
    assert "demo-mini" in rows["cost_delta_usd"]
    # the disclosure names it; it must NOT read as a bare confident dollar figure
    assert not rows["cost_delta_usd"].lstrip("-").replace(".", "").isdigit()


def _priced_bundle(run_id: str, in_tok: int, out_tok: int) -> TraceBundle:
    return TraceBundle.from_dict(
        {
            "schema_version": "0.1",
            "run": {
                "id": run_id,
                "workflow": "w",
                "framework": "synthetic",
                "provider": "synthetic",
                "model": "gpt-4o-mini",  # IS in the stub price table
                "started_at": "2026-01-01T00:00:00Z",
                "ended_at": "2026-01-01T00:00:01Z",
                "success_label": "passed",
                "total_input_tokens": in_tok,
                "total_output_tokens": out_tok,
            },
            "steps": [
                {
                    "id": f"{run_id}_s1",
                    "type": "model",
                    "name": "m",
                    "started_at": "2026-01-01T00:00:00Z",
                    "ended_at": "2026-01-01T00:00:01Z",
                    "token_usage": {"input": in_tok, "output": out_tok},
                    "retry_count": 0,
                }
            ],
        }
    )


def test_compare_shows_a_real_delta_when_both_priced() -> None:
    """When both sides are model-priced, the delta is a real dollar figure (no
    over-suppression — the fix must not hide a legitimate delta)."""
    left = _priced_bundle("left", 2000, 1000)
    right = _priced_bundle("right", 1000, 500)
    cmp = compare_bundles(left, right)
    assert cmp.cost_delta_priced is True
    rows = dict(cmp.to_rows())
    assert "unpriced" not in rows["cost_delta_usd"]
    # right is cheaper -> negative delta
    assert cmp.cost_delta_usd < 0
