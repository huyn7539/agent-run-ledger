from pathlib import Path

from agent_run_ledger.core.compare import compare_bundles
from agent_run_ledger.core.io import load_trace


def test_compare_bundles_reports_deltas() -> None:
    left = load_trace(Path("fixtures/golden_retry_loop.json"))
    right = load_trace(Path("fixtures/clean_run.json"))

    comparison = compare_bundles(left, right)

    assert comparison.cost_delta_usd < 0
    assert comparison.latency_delta_ms < 0
    assert comparison.retry_delta < 0
    assert comparison.success_change == "failed -> passed"

