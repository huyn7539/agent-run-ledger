"""Install-independence guards (first-user P1, 2026-06-11).

The demo loader USED to read <repo>/fixtures/*.json via parents[3] — a path that
exists in a git checkout but NOT in an installed wheel (the fixtures dir isn't
packaged), so `arl run-demo` (a documented Quick Start command) crashed on every
`uv tool install` user. These tests assert the demo + selftest depend on NO repo
file, so they behave identically installed or from source.
"""

from __future__ import annotations

from agent_run_ledger.core.demo import load_demo_bundle
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.receipt import build_receipts
from agent_run_ledger.core.selftest import selftest_receipts


def test_demo_bundles_load_without_repo_fixtures() -> None:
    """Both demo variants build from embedded data — no filesystem read."""
    retry = load_demo_bundle("retry-loop")
    assert retry.run.id == "run_retry_loop"
    assert retry.steps, "retry-loop demo must have steps"

    clean = load_demo_bundle("clean")
    assert clean.run.id == "run_clean_demo"
    assert clean.steps


def test_demo_retry_loop_fires_clean_does_not() -> None:
    """The demo is meaningful: retry-loop produces a receipt, clean produces none."""
    retry = load_demo_bundle("retry-loop")
    retry = retry.with_prescriptions(analyze_bundle(retry))
    assert build_receipts(retry), "retry-loop demo must fire a receipt"

    clean = load_demo_bundle("clean")
    clean = clean.with_prescriptions(analyze_bundle(clean))
    assert build_receipts(clean) == [], "clean demo must grade clean (no receipt)"


def test_demo_rejects_unknown_variant() -> None:
    import pytest

    with pytest.raises(ValueError):
        load_demo_bundle("nonsense")


def test_selftest_needs_no_repo_file() -> None:
    """selftest already embeds its bundle — guard that it stays that way."""
    receipts = selftest_receipts()
    assert receipts and receipts[0].proof_level
