from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_run_ledger.core.models import TraceBundle


def load_trace(path: Path) -> TraceBundle:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TraceBundle.from_dict(data)


def write_trace(bundle: TraceBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic_trace_dict(bundle: TraceBundle) -> dict[str, Any]:
    return json.loads(json.dumps(bundle.to_dict(), sort_keys=True))

