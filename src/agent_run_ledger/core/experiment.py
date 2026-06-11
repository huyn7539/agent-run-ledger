"""Task 60 — the governed-apply gate math. EXACT RATIONAL decision arithmetic.

Design doc: Akashic 06-learning/agent-run-ledger/2026-06-11-gate-math-proposer-
verifier-synthesis.md, as amended by the Codex spec review (determinism pinned).

Every DECISION quantity here is a ``fractions.Fraction`` computed from integer
ledger counts — bit-identical across platforms, replayable forever (the same
discipline as the rest of the facts layer; floats appear ONLY in display
helpers). With integer Beta parameters, P(p_B > p_A) has a closed form that is
itself a rational number (the Evan Miller / Cook identity), so no quadrature,
no sampling, no float drift.

Arm assignment is sha256-keyed — NEVER Python ``hash()``, which is salted per
process and would break exact replay (Codex spec-review CRITICAL).

Decision rule (Bayesian, fixed Beta(1,1) priors — receipts disclose this; it is
NOT a frequentist alpha-spending guarantee):
  KEEP    iff  P(improvement) >= 95/100  AND  E[delta] >= MDE
          AND  n0 >= min_n AND n1 >= min_n  AND no guardrail breach
  REVERT  iff  guardrail breach (INSTANT, any N — fail-closed, Rule 6)
          OR  (P(harm) >= 70/100 AND E[harm] >= eps_harm)
  else CONTINUE (stay in shadow/review; the receipt shows the live posterior).

The asymmetry is deliberate: reverting is cheap, keeping a bad rule compounds.
NOTE an honest substitution: the spec's P(delta < -eps) tail has no closed
form for an interior offset; the implemented harm test is
P(harm) >= 70/100 AND E[harm] >= eps_harm — still exact, asymmetric, and
fail-closed (documented here so nobody mistakes it for the offset tail).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from fractions import Fraction
from math import comb, sqrt

ASSIGNMENT_DOMAIN = "arl-task60-assignment/v1"

# The ledger's pinned second-resolution UTC shape (models.utc_now_iso). For
# strings of EXACTLY this shape, lexicographic order == chronological order;
# anything else (imported/crafted timestamps, other offsets, sub-second forms)
# is excluded from cohort formation, fail-closed (Codex P2 review F7).
_PINNED_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def pinned_utc_ts(value: str) -> bool:
    """True iff *value* is the pinned ``YYYY-MM-DDTHH:MM:SSZ`` UTC shape."""
    return isinstance(value, str) and bool(_PINNED_UTC_RE.fullmatch(value))

KEEP_CONFIDENCE = Fraction(95, 100)
REVERT_CONFIDENCE = Fraction(70, 100)
DEFAULT_MDE = Fraction(2, 100)
DEFAULT_EPS_HARM = Fraction(1, 100)
DEFAULT_MIN_N = 5

# class-level autonomy (P4): --auto is EARNED when this ledger's own
# kept/reverted history says the class's precision clears the bar.
AUTO_PRECISION_BAR = Fraction(4, 5)
AUTO_CONFIDENCE = Fraction(95, 100)


def assign_arm(experiment_id: str, run_id: str) -> int:
    """Deterministic 0/1 arm for *run_id* in *experiment_id*. sha256-keyed
    (domain-separated), big-endian, parity — replayable on any machine."""
    digest = hashlib.sha256(
        ASSIGNMENT_DOMAIN.encode("utf-8")
        + b"\x00"
        + experiment_id.encode("utf-8")
        + b"\x00"
        + run_id.encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") % 2


def _beta_frac(x: int, y: int) -> Fraction:
    """B(x, y) for positive integers, exactly: 1 / ((x+y-1) * C(x+y-2, x-1))."""
    if x < 1 or y < 1:
        raise ValueError("Beta function arguments must be positive integers")
    return Fraction(1, (x + y - 1) * comb(x + y - 2, x - 1))


def prob_b_greater_a(a_a: int, b_a: int, a_b: int, b_b: int) -> Fraction:
    """P(p_B > p_A) for independent p_A~Beta(a_a,b_a), p_B~Beta(a_b,b_b) with
    integer parameters — EXACT (closed-form finite sum; ties have measure 0)."""
    base = _beta_frac(a_a, b_a)
    total = Fraction(0)
    for i in range(a_b):
        total += _beta_frac(a_a + i, b_a + b_b) / ((b_b + i) * _beta_frac(1 + i, b_b) * base)
    return total


def beta_tail_above(a: int, b: int, x: Fraction) -> Fraction:
    """P(p > x) for p~Beta(a, b), integer a,b, rational x in [0,1] — EXACT via
    the binomial identity I_x(a,b) = sum_{j=a}^{a+b-1} C(a+b-1,j) x^j (1-x)^(n-j)."""
    if not (0 <= x <= 1):
        raise ValueError("x must be in [0, 1]")
    n = a + b - 1
    cdf = sum(comb(n, j) * (x**j) * ((1 - x) ** (n - j)) for j in range(a, n + 1))
    return 1 - cdf


@dataclass(frozen=True)
class GateDecision:
    """The routing decision + the exact posterior quantities behind it."""

    decision: str  # "KEEP" | "REVERT" | "CONTINUE"
    p_improve: Fraction
    e_delta: Fraction  # expected DROP in failure rate (positive = better)
    reasons: tuple[str, ...]

    def display(self) -> dict[str, object]:
        """Floats/strings for rendering ONLY — decisions never read from here."""
        return {
            "decision": self.decision,
            "p_improve": float(self.p_improve),
            "p_improve_exact": f"{self.p_improve.numerator}/{self.p_improve.denominator}",
            "e_delta": float(self.e_delta),
            "reasons": list(self.reasons),
        }


def _validate_counts(n: int, k: int, label: str) -> None:
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid counts for {label}: n={n}, k={k}")


def decide(
    n0: int,
    k0: int,
    n1: int,
    k1: int,
    *,
    mde: Fraction = DEFAULT_MDE,
    eps_harm: Fraction = DEFAULT_EPS_HARM,
    min_n: int = DEFAULT_MIN_N,
    guardrail_breached: bool = False,
) -> GateDecision:
    """Three-way routing over control (n0 runs, k0 failures) vs treatment
    (n1, k1). Failure-rate posteriors are Beta(k+1, n-k+1) (uniform prior)."""
    _validate_counts(n0, k0, "control")
    _validate_counts(n1, k1, "treatment")
    if mde <= 0 or eps_harm <= 0 or min_n < 1:
        raise ValueError("mde and eps_harm must be > 0 and min_n >= 1 (zero disables the gate)")

    a0, b0 = k0 + 1, n0 - k0 + 1
    a1, b1 = k1 + 1, n1 - k1 + 1
    p_worse = prob_b_greater_a(a0, b0, a1, b1)  # P(p1 > p0): treatment fails MORE
    p_improve = 1 - p_worse
    e_delta = Fraction(a0, a0 + b0) - Fraction(a1, a1 + b1)

    if guardrail_breached:
        return GateDecision(
            "REVERT", p_improve, e_delta, ("guardrail breach: instant revert (fail-closed)",)
        )
    if (
        n0 >= min_n
        and n1 >= min_n
        and p_improve >= KEEP_CONFIDENCE
        and e_delta >= mde
    ):
        return GateDecision(
            "KEEP",
            p_improve,
            e_delta,
            (f"P(improvement) >= {KEEP_CONFIDENCE} and E[delta] >= MDE with n >= {min_n} per arm",),
        )
    if p_worse >= REVERT_CONFIDENCE and -e_delta >= eps_harm:
        return GateDecision(
            "REVERT",
            p_improve,
            e_delta,
            (f"P(harm) >= {REVERT_CONFIDENCE} and E[harm] >= eps_harm (asymmetric bar)",),
        )
    reasons = []
    if n0 < min_n or n1 < min_n:
        reasons.append(f"insufficient runs per arm (need {min_n})")
    reasons.append("evidence not decisive either way; stay in shadow/review")
    return GateDecision("CONTINUE", p_improve, e_delta, tuple(reasons))


def ci95_display(n0: int, k0: int, n1: int, k1: int) -> tuple[float, float]:
    """Normal-approximation 95% interval for delta — DISPLAY ONLY (the decision
    quantities above are exact; this is the human-readable error bar)."""
    a0, b0 = k0 + 1, n0 - k0 + 1
    a1, b1 = k1 + 1, n1 - k1 + 1

    def _var(a: int, b: int) -> float:
        return (a * b) / (((a + b) ** 2) * (a + b + 1))

    mean = a0 / (a0 + b0) - a1 / (a1 + b1)
    sd = sqrt(_var(a0, b0) + _var(a1, b1))
    return (mean - 1.96 * sd, mean + 1.96 * sd)


def guardrail_breach(
    n0: int, k0: int, n1: int, k1: int, *, eps_harm: Fraction = DEFAULT_EPS_HARM
) -> bool:
    """Task 61 — the wired guardrail: ALL-class failure-receipt rate, treatment
    vs the pre-registered baseline, judged with the SAME exact-rational
    machinery at the REVERT bar. Deliberately has NO min_n requirement —
    a breach reverts instantly from first exposure (asymmetric, Rule 6);
    ``decide()`` already routes ``guardrail_breached=True`` to REVERT at any n.

    Zero treatment exposure (n1 == 0) is no evidence, not a breach."""
    if eps_harm <= 0:
        raise ValueError("eps_harm must be > 0 (zero disables the guardrail)")
    _validate_counts(n0, k0, "guardrail control")
    _validate_counts(n1, k1, "guardrail treatment")
    if n1 == 0:
        return False
    a0, b0 = k0 + 1, n0 - k0 + 1
    a1, b1 = k1 + 1, n1 - k1 + 1
    p_worse = prob_b_greater_a(a0, b0, a1, b1)  # P(p1 > p0): MORE failures overall
    e_harm = Fraction(a1, a1 + b1) - Fraction(a0, a0 + b0)
    return p_worse >= REVERT_CONFIDENCE and e_harm >= eps_harm


def auto_earned(kept: int, reverted: int) -> bool:
    """P4: a proposal class earns --auto only when P(precision > 4/5) >= 95/100
    under a Beta(kept+1, reverted+1) posterior over ITS OWN kept/reverted
    history in THIS ledger. Zero history => not earned (Beta(1,1) tail at 0.8
    is 1/5) — autonomy is earned, never granted."""
    if kept < 0 or reverted < 0:
        raise ValueError("history counts must be non-negative")
    return beta_tail_above(kept + 1, reverted + 1, AUTO_PRECISION_BAR) >= AUTO_CONFIDENCE
