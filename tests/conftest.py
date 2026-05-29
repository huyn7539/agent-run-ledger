from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_run_ledger.core.models import RunRecord, StepRecord, TraceBundle


@pytest.fixture
def tmp_db(tmp_path: Path):
    def factory(name: str = "ledger.sqlite") -> Path:
        return tmp_path / name

    return factory


@pytest.fixture
def non_demo_bundle() -> TraceBundle:
    run = RunRecord(
        id="run_invoice_audit",
        workflow="invoice-reconciliation",
        framework="langgraph",
        provider="anthropic",
        model="claude-sonnet-4",
        started_at="2026-05-28T16:00:00Z",
        ended_at="2026-05-28T16:00:09Z",
        success_label="failed",
        total_cost_usd=0.21,
        total_latency_ms=9000,
        total_input_tokens=1800,
        total_output_tokens=700,
    )
    steps = [
        StepRecord(
            id="step_parse_invoice",
            run_id=run.id,
            step_type="model",
            name="invoice.parse_pdf",
            started_at=run.started_at,
            ended_at="2026-05-28T16:00:03Z",
            input_tokens=900,
            output_tokens=350,
            cost_usd=0.09,
        ),
        StepRecord(
            id="step_vendor_lookup",
            run_id=run.id,
            step_type="tool",
            name="erp.vendor_lookup",
            started_at="2026-05-28T16:00:03Z",
            ended_at=run.ended_at,
            input_tokens=900,
            output_tokens=350,
            cost_usd=0.12,
            retry_count=3,
            error="TimeoutError: ERP lookup exceeded 2s timeout",
            metadata={"tool_name": "erp.vendor_lookup", "timeout_ms": 2000},
        ),
    ]
    return TraceBundle(run=run, steps=steps)


@pytest.fixture
def non_demo_target_bundle(non_demo_bundle: TraceBundle) -> TraceBundle:
    target_steps = []
    for step in non_demo_bundle.steps:
        if step.id == "step_vendor_lookup":
            target_steps.append(
                StepRecord(
                    id=step.id,
                    run_id=step.run_id,
                    step_type=step.step_type,
                    name=step.name,
                    started_at=step.started_at,
                    ended_at=step.ended_at,
                    input_tokens=step.input_tokens,
                    output_tokens=step.output_tokens,
                    cost_usd=step.cost_usd,
                    retry_count=step.retry_count,
                    error=step.error,
                    metadata={
                        "retry_budget_patch_target": {
                            "path": "services/vendor_retry.py",
                            "before": "VENDOR_LOOKUP_MAX_RETRIES = 3",
                            "after": "VENDOR_LOOKUP_MAX_RETRIES = 0",
                        }
                    },
                )
            )
        else:
            target_steps.append(step)
    return TraceBundle(run=non_demo_bundle.run, steps=target_steps)


def bundle_dict(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": "0.1",
        "run": {
            "id": "run_fixture",
            "workflow": "fixture-workflow",
            "framework": "fixture-framework",
            "provider": "fixture-provider",
            "model": "fixture-model",
            "started_at": "2026-05-28T00:00:00Z",
            "ended_at": "2026-05-28T00:00:01Z",
            "success_label": "passed",
            "total_cost_usd": 0.1,
        },
        "steps": [
            {
                "id": "step_fixture",
                "type": "tool",
                "name": "fixture.step",
                "started_at": "2026-05-28T00:00:00Z",
                "ended_at": "2026-05-28T00:00:01Z",
                "token_usage": {"input": 1, "output": 2},
                "cost_usd": 0.1,
                "retry_count": 0,
            }
        ],
    }
    if overrides:
        data.update(overrides)
    return data
