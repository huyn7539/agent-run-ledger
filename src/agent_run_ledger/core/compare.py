from __future__ import annotations

from dataclasses import dataclass

from agent_run_ledger.core.models import TraceBundle


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

    def to_rows(self) -> list[tuple[str, str]]:
        return [
            ("left", self.left_run_id),
            ("right", self.right_run_id),
            ("cost_delta_usd", f"{self.cost_delta_usd:.6f}"),
            ("latency_delta_ms", str(self.latency_delta_ms)),
            ("input_token_delta", str(self.input_token_delta)),
            ("output_token_delta", str(self.output_token_delta)),
            ("retry_delta", str(self.retry_delta)),
            ("success_change", self.success_change),
        ]


def compare_bundles(left: TraceBundle, right: TraceBundle) -> RunComparison:
    left_retries = sum(step.retry_count for step in left.steps)
    right_retries = sum(step.retry_count for step in right.steps)
    success_change = (
        "unchanged"
        if left.run.success_label == right.run.success_label
        else f"{left.run.success_label} -> {right.run.success_label}"
    )
    return RunComparison(
        left_run_id=left.run.id,
        right_run_id=right.run.id,
        cost_delta_usd=right.run.total_cost_usd - left.run.total_cost_usd,
        latency_delta_ms=right.run.total_latency_ms - left.run.total_latency_ms,
        input_token_delta=right.run.total_input_tokens - left.run.total_input_tokens,
        output_token_delta=right.run.total_output_tokens - left.run.total_output_tokens,
        retry_delta=right_retries - left_retries,
        success_change=success_change,
    )

