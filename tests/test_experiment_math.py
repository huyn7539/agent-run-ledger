"""Task 60 — gate math golden vectors (EXACT rational decision arithmetic).

The decision path must be bit-identical across platforms forever: Fractions in,
Fractions compared, floats only in display helpers. Golden vectors below are
hand-derivable closed forms — if any of them drifts, replay is broken.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from agent_run_ledger.core.experiment import (
    assign_arm,
    auto_earned,
    beta_tail_above,
    ci95_display,
    decide,
    prob_b_greater_a,
)


# --- closed-form golden vectors -------------------------------------------------

def test_prob_b_greater_a_symmetric_uniform_is_exactly_half() -> None:
    assert prob_b_greater_a(1, 1, 1, 1) == Fraction(1, 2)


def test_prob_b_greater_a_one_success_is_exactly_two_thirds() -> None:
    # p_A ~ Beta(1,1), p_B ~ Beta(2,1): P(p_B > p_A) = ∫(1−a²)da = 2/3 exactly.
    assert prob_b_greater_a(1, 1, 2, 1) == Fraction(2, 3)


def test_beta_tail_uniform_at_four_fifths_is_exactly_one_fifth() -> None:
    assert beta_tail_above(1, 1, Fraction(4, 5)) == Fraction(1, 5)


def test_decision_quantities_are_exact_fractions_and_reproducible() -> None:
    d1 = decide(20, 10, 20, 1)
    d2 = decide(20, 10, 20, 1)
    assert isinstance(d1.p_improve, Fraction)
    assert isinstance(d1.e_delta, Fraction)
    assert d1.p_improve == d2.p_improve and d1.e_delta == d2.e_delta


# --- sha256 arm assignment (NEVER Python hash() — salted, breaks replay) --------

def test_assign_arm_is_deterministic_and_uses_both_arms() -> None:
    assert assign_arm("exp1", "run42") == assign_arm("exp1", "run42")
    arms = {assign_arm("exp1", f"run{i}") for i in range(100)}
    assert arms == {0, 1}
    # different experiment => independent assignment stream
    diffs = sum(
        assign_arm("exp1", f"run{i}") != assign_arm("exp2", f"run{i}") for i in range(100)
    )
    assert diffs > 0


# --- three-way routing ------------------------------------------------------------

def test_strong_improvement_with_enough_runs_is_keep() -> None:
    d = decide(20, 10, 20, 0)
    assert d.decision == "KEEP"
    assert d.p_improve >= Fraction(95, 100)
    assert d.e_delta >= Fraction(2, 100)


def test_strong_harm_is_revert() -> None:
    d = decide(20, 0, 20, 10)
    assert d.decision == "REVERT"


def test_guardrail_breach_reverts_instantly_at_any_n() -> None:
    d = decide(0, 0, 0, 0, guardrail_breached=True)
    assert d.decision == "REVERT"
    assert "guardrail" in d.reasons[0]


def test_small_n_strong_evidence_stays_continue() -> None:
    # min_n is a KEEP gate only (fail-closed: uncertainty never auto-applies;
    # but harm needs no minimum — reverting is cheap).
    d = decide(3, 3, 3, 0)
    assert d.decision == "CONTINUE"
    assert any("insufficient" in r for r in d.reasons)


def test_inconclusive_evidence_stays_continue() -> None:
    assert decide(10, 5, 10, 4).decision == "CONTINUE"


def test_zero_mde_or_eps_is_rejected() -> None:
    with pytest.raises(ValueError):
        decide(10, 1, 10, 1, mde=Fraction(0))
    with pytest.raises(ValueError):
        decide(10, 1, 10, 1, eps_harm=Fraction(0))
    with pytest.raises(ValueError):
        decide(10, 1, 10, 1, min_n=0)


def test_invalid_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        decide(5, 6, 5, 0)  # k > n
    with pytest.raises(ValueError):
        decide(-1, 0, 5, 0)


def test_ci95_is_display_only_floats() -> None:
    lo, hi = ci95_display(20, 10, 20, 0)
    assert isinstance(lo, float) and isinstance(hi, float) and lo < hi


def test_guardrail_breach_exact_vectors() -> None:
    from fractions import Fraction

    from agent_run_ledger.core.experiment import guardrail_breach

    # zero treatment exposure = no evidence, never a breach
    assert guardrail_breach(5, 3, 0, 0) is False
    # the Task 61 e2e vector: control 3/5 -> treatment 4/4 overall failures
    assert guardrail_breach(5, 3, 4, 4) is True
    # overall rate IMPROVED -> no breach
    assert guardrail_breach(5, 3, 4, 0) is False
    # tiny worsening below eps_harm -> no breach (expected harm must clear eps)
    assert guardrail_breach(100, 50, 100, 51, eps_harm=Fraction(5, 100)) is False
    # invalid inputs rejected
    with pytest.raises(ValueError):
        guardrail_breach(5, 6, 1, 0)  # k > n
    with pytest.raises(ValueError):
        guardrail_breach(5, 3, 1, 0, eps_harm=Fraction(0))


def test_pinned_utc_ts_accepts_only_the_ledger_shape() -> None:
    from agent_run_ledger.core.experiment import pinned_utc_ts

    assert pinned_utc_ts("2026-06-11T01:02:03Z")
    assert not pinned_utc_ts("2026-06-11T01:02:03+00:00")  # offset form
    assert not pinned_utc_ts("2026-06-11T01:02:03.123Z")  # sub-second form
    assert not pinned_utc_ts("2026-06-11T01:02:03Z\n")  # trailing newline
    assert not pinned_utc_ts("")
    assert not pinned_utc_ts("garbage")


# --- P4: autonomy earned, never granted -------------------------------------------

def test_auto_not_earned_with_no_history() -> None:
    assert auto_earned(0, 0) is False


def test_auto_earned_thresholds_closed_form() -> None:
    # Beta(K+1, 1): P(p > 4/5) = 1 - (4/5)^(K+1). Bar 95/100 crosses at K=13.
    assert auto_earned(12, 0) is False
    assert auto_earned(13, 0) is True
    # a revert pushes it back below the bar
    assert auto_earned(13, 3) is False
