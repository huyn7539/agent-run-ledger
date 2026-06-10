from __future__ import annotations

from dataclasses import dataclass

from agent_run_ledger.core.cost import cost_is_priced, cost_on_read
from agent_run_ledger.core.models import TraceBundle
from agent_run_ledger.core.prescriptions import derive_retry_steps


@dataclass(frozen=True)
class RunComparison:
    left_run_id: str
    right_run_id: str
    cost_delta_usd: float
    latency_delta_ms: int
    input_token_delta: int
    output_token_delta: int
    retry_delta: int
    success_change: str
    # A2 (Task 57): the cost delta is only meaningful when BOTH sides are priced.
    # When either falls back to the cached/unknown-model value, a confident dollar
    # delta would CONTRADICT what list-runs/report disclose ("unpriced"). Carry the
    # priced state so the row discloses instead of overclaiming.
    cost_delta_priced: bool = True
    cost_unpriced_note: str = ""

    def cost_delta_display(self) -> str:
        if self.cost_delta_priced:
            return f"{self.cost_delta_usd:.6f}"
        return f"unpriced ({self.cost_unpriced_note})"

    def to_rows(self) -> list[tuple[str, str]]:
        return [
            ("left", self.left_run_id),
            ("right", self.right_run_id),
            ("cost_delta_usd", self.cost_delta_display()),
            ("latency_delta_ms", str(self.latency_delta_ms)),
            ("input_token_delta", str(self.input_token_delta)),
            ("output_token_delta", str(self.output_token_delta)),
            ("retry_delta", str(self.retry_delta)),
            ("success_change", self.success_change),
        ]


def compare_bundles(left: TraceBundle, right: TraceBundle) -> RunComparison:
    # Derived retry view (collapse-on-read) — the SAME view the detector + report
    # use, so retry_delta agrees with the prescriptions, not the raw attempts.
    left_retries = sum(step.retry_count for step in derive_retry_steps(left))
    right_retries = sum(step.retry_count for step in derive_retry_steps(right))
    success_change = (
        "unchanged"
        if left.run.success_label == right.run.success_label
        else f"{left.run.success_label} -> {right.run.success_label}"
    )
    # A2 (Task 57): a cost delta is honest only when BOTH sides are model-priced
    # (or provider-reported). If either falls back to the cached/unknown-model value,
    # disclose "unpriced" — never a confident delta that contradicts list-runs/report.
    both_priced = cost_is_priced(left) and cost_is_priced(right)
    unpriced_note = ""
    if not both_priced:
        unknown = [b.run.model for b in (left, right) if not cost_is_priced(b)]
        unpriced_note = f"model(s) not in price table: {', '.join(sorted(set(unknown)))}"
    return RunComparison(
        left_run_id=left.run.id,
        right_run_id=right.run.id,
        # LR2: total_cost_usd is a CACHE, not authoritative — a stale cache can
        # flip the SIGN of this delta vs report/list. Compute from cost_on_read
        # (token facts x stub price table), the same source report + list use.
        cost_delta_usd=cost_on_read(right) - cost_on_read(left),
        latency_delta_ms=right.run.total_latency_ms - left.run.total_latency_ms,
        input_token_delta=right.run.total_input_tokens - left.run.total_input_tokens,
        output_token_delta=right.run.total_output_tokens - left.run.total_output_tokens,
        retry_delta=right_retries - left_retries,
        success_change=success_change,
        cost_delta_priced=both_priced,
        cost_unpriced_note=unpriced_note,
    )

