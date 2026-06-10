"""Built-in demo bundles — EMBEDDED in-package, not read from disk.

First-user P1 (2026-06-11): the old loader read fixtures from
``<repo>/fixtures/*.json`` via ``parents[3]``. That path exists in a git
checkout but NOT in an installed wheel (the fixtures dir is not packaged), so
``arl run-demo`` — a documented Quick Start command — crashed with a traceback on
every ``uv tool install`` user. The demo data is now embedded here, exactly like
``core.selftest.SELFTEST_BUNDLE``, so the demo behaves identically from a checkout
or an installed tool. No file I/O, nothing to package.
"""

from __future__ import annotations

from typing import Any

from agent_run_ledger.core.models import TraceBundle

# The retry-loop demo IS the selftest's known-bad bundle (one source of truth) with
# a stable demo run id so `arl report --run run_retry_loop` is copy-pasteable.
from agent_run_ledger.core.selftest import SELFTEST_BUNDLE

_RETRY_LOOP_BUNDLE: dict[str, Any] = {
    **SELFTEST_BUNDLE,
    "run": {**SELFTEST_BUNDLE["run"], "id": "run_retry_loop", "workflow": "demo-retry-loop"},
}

# A clean run: a plan step + one successful tool step + a finalize step, no retries,
# no errors — the negative example (`arl run-demo --variant clean` -> grades clean).
_CLEAN_BUNDLE: dict[str, Any] = {
    "schema_version": "0.1",
    "run": {
        "id": "run_clean_demo",
        "workflow": "demo-clean",
        "framework": "synthetic",
        "provider": "synthetic",
        "model": "demo-mini",
        "started_at": "2026-05-28T15:00:00Z",
        "ended_at": "2026-05-28T15:00:09Z",
        # "passed" (not "succeeded") is the exact label the report keys its honest
        # "clean run" phrasing on (report.py: success_label == "passed" and no errors).
        "success_label": "passed",
        "total_input_tokens": 1800,
        "total_output_tokens": 600,
    },
    "steps": [
        {
            "id": "step_1",
            "type": "model",
            "name": "plan",
            "started_at": "2026-05-28T15:00:00Z",
            "ended_at": "2026-05-28T15:00:03Z",
            "token_usage": {"input": 1000, "output": 300},
            "cost_usd": 0.03,
            "retry_count": 0,
            "redaction_mode": "metadata_only",
        },
        {
            "id": "step_2",
            "type": "tool",
            "name": "demo.lookup",
            "started_at": "2026-05-28T15:00:03Z",
            "ended_at": "2026-05-28T15:00:06Z",
            "token_usage": {"input": 500, "output": 100},
            "cost_usd": 0.01,
            "retry_count": 0,
            "redaction_mode": "metadata_only",
        },
        {
            "id": "step_3",
            "type": "model",
            "name": "finalize",
            "started_at": "2026-05-28T15:00:06Z",
            "ended_at": "2026-05-28T15:00:09Z",
            "token_usage": {"input": 300, "output": 200},
            "cost_usd": 0.01,
            "retry_count": 0,
            "redaction_mode": "metadata_only",
        },
    ],
}

_DEMO_BUNDLES: dict[str, dict[str, Any]] = {
    "retry-loop": _RETRY_LOOP_BUNDLE,
    "clean": _CLEAN_BUNDLE,
}


def load_demo_bundle(variant: str) -> TraceBundle:
    data = _DEMO_BUNDLES.get(variant)
    if data is None:
        raise ValueError("variant must be one of: retry-loop, clean")
    return TraceBundle.from_dict(data)
