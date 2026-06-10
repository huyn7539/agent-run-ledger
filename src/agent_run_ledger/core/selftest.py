"""`arl selftest` — prove the alarm fires (2026-06-10 persona-gauntlet fix).

Every gauntlet persona who churned on silence said a version of the same thing:
"I cannot distinguish 'my sessions are clean' from 'the detector is deaf'."
This module makes 'clean' falsifiable in minute one: it runs a BUNDLED known-bad
run (the golden retry-loop shape, embedded so installed wheels need no repo
checkout) through the REAL pipeline — same detector, same grader — and asserts a
graded receipt fires. If selftest passes, silence on your runs means the
detector abstained, not that the plumbing is broken.
"""

from __future__ import annotations

from typing import Any

from agent_run_ledger.core.models import TraceBundle
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.receipt import RepairReceipt, build_receipts

# The golden retry-loop run, embedded. Content-free synthetic facts: a planning
# model step, one tool step that retried 4 times against a 5s timeout, a closing
# model step. Identical shape to fixtures/golden_retry_loop.json.
SELFTEST_BUNDLE: dict[str, Any] = {
    "schema_version": "0.1",
    "run": {
        "id": "run_selftest_retry_loop",
        "workflow": "selftest-known-bad",
        "framework": "synthetic",
        "provider": "synthetic",
        "model": "demo-mini",
        "started_at": "2026-05-28T14:00:00Z",
        "ended_at": "2026-05-28T14:00:42Z",
        "success_label": "failed",
        "prompt_hash": "5e7f042f4203ed8092e33b77b8f7e52f0d64b46769b9ac18c3a892000697ca39",
        # provider/model are neutral synthetics: core/ may name no provider
        # (egress-guard test), and the selftest needs detection, not pricing.
        "config_hash": "ae7aa47355c7946aa6dda2ebe073765f6f7e83782291882c54a7afb40c96b5f7",
        "total_cost_usd": 0.1842,
        "total_latency_ms": 42000,
        "total_input_tokens": 4200,
        "total_output_tokens": 1900,
    },
    "steps": [
        {
            "id": "step_1",
            "type": "model",
            "name": "plan",
            "started_at": "2026-05-28T14:00:00Z",
            "ended_at": "2026-05-28T14:00:08Z",
            "token_usage": {"input": 2200, "output": 700},
            "cost_usd": 0.067,
            "retry_count": 0,
            "redaction_mode": "metadata_only",
        },
        {
            "id": "step_2",
            "type": "tool",
            "name": "demo.flaky_tool",
            "started_at": "2026-05-28T14:00:08Z",
            "ended_at": "2026-05-28T14:00:34Z",
            "token_usage": {"input": 1400, "output": 800},
            "cost_usd": 0.092,
            "retry_count": 4,
            "error": "TimeoutError: CRM lookup exceeded 5s timeout",
            "redaction_mode": "metadata_only",
            "metadata": {"tool_name": "demo.flaky_tool", "timeout_ms": 5000},
        },
        {
            "id": "step_3",
            "type": "model",
            "name": "finalize_failure",
            "started_at": "2026-05-28T14:00:34Z",
            "ended_at": "2026-05-28T14:00:42Z",
            "token_usage": {"input": 600, "output": 400},
            "cost_usd": 0.0252,
            "retry_count": 0,
            "redaction_mode": "metadata_only",
        },
    ],
}


def selftest_receipts() -> list[RepairReceipt]:
    """The known-bad bundle through the REAL pipeline (no shortcuts, no db)."""
    bundle = TraceBundle.from_dict(SELFTEST_BUNDLE)
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    return build_receipts(bundle)
