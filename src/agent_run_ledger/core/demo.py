from __future__ import annotations

import json
from pathlib import Path

from agent_run_ledger.core.models import TraceBundle

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "fixtures"


def load_demo_bundle(variant: str) -> TraceBundle:
    fixture_name = {
        "retry-loop": "golden_retry_loop.json",
        "clean": "clean_run.json",
    }.get(variant)
    if fixture_name is None:
        raise ValueError("variant must be one of: retry-loop, clean")
    data = json.loads((FIXTURE_DIR / fixture_name).read_text(encoding="utf-8"))
    return TraceBundle.from_dict(data)

