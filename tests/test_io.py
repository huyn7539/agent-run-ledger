from __future__ import annotations

from pathlib import Path
import json

import pytest

from agent_run_ledger.core.io import (
    TraceParseError,
    load_trace,
    semantic_trace_dict,
    write_trace,
)
from agent_run_ledger.core.models import TraceBundle


def test_json_roundtrip_via_file(tmp_path: Path) -> None:
    bundle = load_trace(Path("fixtures/golden_retry_loop.json"))
    out = tmp_path / "trace.json"

    write_trace(bundle, out)

    assert semantic_trace_dict(load_trace(out)) == semantic_trace_dict(bundle)


def test_malformed_json_raises_parse_error(tmp_path: Path) -> None:
    """Defensive parsing (Constraint 4): malformed JSON surfaces as a typed
    TraceParseError, not a raw json.JSONDecodeError leaking out of core."""
    path = tmp_path / "bad.json"
    path.write_text("{bad", encoding="utf-8")

    with pytest.raises(TraceParseError):
        load_trace(path)


def test_extra_unknown_fields_ignored(tmp_path: Path) -> None:
    data = json.loads(Path("fixtures/clean_run.json").read_text(encoding="utf-8"))
    data["future"] = {"field": True}
    data["steps"][0]["future_step"] = "ignored"
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    bundle = load_trace(path)

    assert bundle.run.id == "run_clean"
    assert "future_step" not in bundle.steps[0].to_dict()


def test_unicode_preserved(tmp_path: Path) -> None:
    data = json.loads(Path("fixtures/clean_run.json").read_text(encoding="utf-8"))
    data["steps"][0]["metadata"] = {
        "retry_budget_patch_target": {
            "path": "settings/retries_\U0001f600.py",
            "before": "CRM_LOOKUP_RETRIES = 4",
            "after": "CRM_LOOKUP_RETRIES = 0",
        },
        "note": "hello \U0001f600",
    }
    path = tmp_path / "unicode.json"
    out = tmp_path / "unicode-out.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    # raw_local: this test pins LOCAL full-fidelity unicode round-trip; the
    # default (share-form) export drops raw-content values (Task 46).
    write_trace(load_trace(path), out, raw_local=True)

    assert "\\ud83d\\ude00" in out.read_text(encoding="utf-8")
    metadata = load_trace(out).steps[0].metadata
    assert metadata["retry_budget_patch_target"]["path"] == "settings/retries_\U0001f600.py"
    assert "note" not in metadata


def test_deeply_nested_metadata_10_levels(tmp_path: Path) -> None:
    data = json.loads(Path("fixtures/clean_run.json").read_text(encoding="utf-8"))
    nested = {"total_tokens": 42}
    for idx in range(10):
        nested = {"usage": nested, "reasoning": f"secret-{idx}"}
    data["steps"][0]["metadata"] = {"usage": nested}
    path = tmp_path / "deep.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    bundle = load_trace(path)

    current = bundle.steps[0].metadata["usage"]
    for _ in range(10):
        assert "reasoning" not in current
        current = current["usage"]
    assert current["total_tokens"] == 42


def test_huge_file_5000_steps_bounded(tmp_path: Path) -> None:
    data = json.loads(Path("fixtures/clean_run.json").read_text(encoding="utf-8"))
    base = data["steps"][0]
    data["steps"] = [{**base, "id": f"step_{idx}"} for idx in range(5000)]
    bundle = TraceBundle.from_dict(data)
    out = tmp_path / "huge.json"

    write_trace(bundle, out)
    loaded = load_trace(out)

    assert len(loaded.steps) == 5000
