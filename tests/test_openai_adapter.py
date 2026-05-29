import json
from pathlib import Path
from typing import Any

import pytest

from agent_run_ledger.adapters.openai import (
    NoSpansCapturedError,
    OpenAILedgerTraceProcessor,
    bundle_from_recorded_trace,
)
from agent_run_ledger.core.compare import compare_bundles
from agent_run_ledger.core.io import write_trace
from agent_run_ledger.core.models import StepRecord
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.report import render_comparison, render_report
from agent_run_ledger.core.storage import load_bundle, save_bundle


def test_processor_emits_neutral_run_from_recorded_trace(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    recorded = json.loads(Path("tests/fixtures/openai_recorded_trace.json").read_text())

    processor = OpenAILedgerTraceProcessor(db)
    processor.on_trace_start(recorded["trace"])
    for span in recorded["spans"]:
        processor.on_span_end(span)
    processor.on_trace_end(recorded["trace"])

    bundle = load_bundle(db, "trace_0123456789abcdef0123456789abcdef")

    assert bundle.run.workflow == "recorded-openai-agent"
    assert bundle.run.framework == "openai-agents-python"
    assert bundle.run.model == "gpt-4.1-mini"
    assert bundle.run.started_at == "2026-05-28T15:00:00Z"
    assert bundle.run.ended_at == "2026-05-28T15:00:12Z"
    assert bundle.run.total_input_tokens == 125
    assert bundle.run.total_output_tokens == 50
    assert any(step.retry_count == 3 for step in bundle.steps)
    assert bundle.prescriptions[0].patch_type == "config_diff"
    # L4: span kind preserved from span_data.type (agent/generation/custom).
    assert {s.span_kind for s in bundle.steps} >= {"agent", "generation", "custom"}
    # L7: the custom span reported cost_usd=0.04 → captured as the cost FACT.
    assert any(s.provider_reported_cost_usd == 0.04 for s in bundle.steps)


def test_openai_processor_fails_closed_when_no_spans(tmp_path: Path) -> None:
    processor = OpenAILedgerTraceProcessor(tmp_path / "ledger.sqlite")
    processor.on_trace_start({"trace_id": "trace_empty", "workflow_name": "empty"})

    with pytest.raises(NoSpansCapturedError):
        processor.on_trace_end({"trace_id": "trace_empty", "workflow_name": "empty"})


def test_bundle_from_recorded_trace_is_provider_neutral() -> None:
    recorded = json.loads(Path("tests/fixtures/openai_recorded_trace.json").read_text())

    bundle = bundle_from_recorded_trace(recorded)

    assert bundle.run.provider == "openai"
    assert all(step.run_id == bundle.run.id for step in bundle.steps)
    # L5: provenance hash stamped locally at capture.
    assert bundle.run.provenance_hash is not None
    assert bundle.run.provenance_hash.startswith("sha256:")


def test_bundle_from_recorded_trace_rejects_empty_spans() -> None:
    recorded = {"trace": {"trace_id": "trace_empty", "workflow_name": "empty"}}

    with pytest.raises(ValueError, match="recorded trace contains no spans"):
        bundle_from_recorded_trace(recorded)


def test_error_does_not_leak_prompt(tmp_path: Path) -> None:
    sentinels = {
        "adapter_error": "PROMPT_SENTINEL_adapter_error_abc123",
        "import_error": "PROMPT_SENTINEL_import_error_abc123",
        "metadata_value": "PROMPT_SENTINEL_metadata_value_abc123",
        "metadata_key_top": "PROMPT_SENTINEL_metadata_key_top_abc123",
        "metadata_key_nested": "PROMPT_SENTINEL_metadata_key_nested_abc123",
        "non_ascii_value": "секрет_СЕНТ_value_123",
        "non_ascii_key": "секрет_СЕНТ_key_123",
    }
    recorded = {
        "trace": {
            "trace_id": "trace_prompt_leak",
            "workflow_name": "redaction-test",
            "started_at": "2026-05-28T15:00:00Z",
            "ended_at": "2026-05-28T15:00:01Z",
        },
        "spans": [
            {
                "span_id": "span_prompt_leak",
                "started_at": "2026-05-28T15:00:00Z",
                "ended_at": "2026-05-28T15:00:01Z",
                "span_data": {
                    "type": "custom",
                    "name": "redaction.step",
                    "error": sentinels["adapter_error"],
                    "data": {
                        "retry_count": 3,
                        "retry_budget_patch_target": {
                            "path": "settings/retries.py",
                            "before": "CRM_LOOKUP_RETRIES = 4",
                            "after": "CRM_LOOKUP_RETRIES = 0",
                        },
                        "current_text": "CRM_LOOKUP_RETRIES = 4",
                        "replacement_text": "CRM_LOOKUP_RETRIES = 0",
                        "current_line": 12,
                        "replacement_line": 12,
                        "reasoning": sentinels["metadata_value"],
                        f"leaked_{sentinels['metadata_key_top']}": "top key value",
                        "usage": {
                            "input_tokens": 20,
                            "output_tokens": 5,
                            f"leaked_{sentinels['metadata_key_nested']}": "nested key value",
                        },
                        "unicode_value": sentinels["non_ascii_value"],
                        f"leaked_{sentinels['non_ascii_key']}": "unicode key value",
                    },
                },
            }
        ],
    }
    adapter_bundle = bundle_from_recorded_trace(recorded)
    imported_step = StepRecord.from_dict(
        {
            "id": "step_import_leak",
            "type": "tool",
            "name": "import.redaction",
            "started_at": "2026-05-28T15:00:00Z",
            "ended_at": "2026-05-28T15:00:01Z",
            "error": f"{sentinels['import_error']} trailing text",
            "retry_count": 3,
            "metadata": {
                "reasoning": sentinels["metadata_value"],
                f"leaked_{sentinels['metadata_key_top']}": "import top key value",
                "usage": {
                    f"leaked_{sentinels['metadata_key_nested']}": "import nested key value",
                },
                "unicode_value": sentinels["non_ascii_value"],
                f"leaked_{sentinels['non_ascii_key']}": "import unicode key value",
            },
        },
        adapter_bundle.run.id,
    )
    bundle = type(adapter_bundle)(run=adapter_bundle.run, steps=[*adapter_bundle.steps, imported_step])
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    db = tmp_path / "ledger.sqlite"
    out = tmp_path / "trace.json"

    run_id = save_bundle(db, bundle)
    loaded = load_bundle(db, run_id)
    write_trace(loaded, out)
    leak_channels = _leak_channels(db, out, loaded, sentinels)

    assert leak_channels == [], leak_channels
    assert loaded.steps[0].metadata["retry_budget_patch_target"] == {
        "path": "settings/retries.py",
        "before": "CRM_LOOKUP_RETRIES = 4",
        "after": "CRM_LOOKUP_RETRIES = 0",
    }
    assert loaded.steps[0].metadata["current_text"] == "CRM_LOOKUP_RETRIES = 4"
    assert loaded.steps[0].metadata["replacement_text"] == "CRM_LOOKUP_RETRIES = 0"
    assert loaded.steps[0].metadata["current_line"] == 12
    assert loaded.steps[0].metadata["replacement_line"] == 12


def _leak_channels(
    db: Path,
    exported_json: Path,
    loaded: Any,
    sentinels: dict[str, str],
) -> list[str]:
    leak_channels = []
    channels = {
        "sqlite": _sqlite_raw_text(db),
        "json_export": list(_walk_strings(json.loads(exported_json.read_text(encoding="utf-8")))),
        "html_report": [render_report(loaded)],
        "render_comparison": [render_comparison(compare_bundles(loaded, loaded))],
    }
    for sentinel_name, sentinel in sentinels.items():
        for channel_name, strings in channels.items():
            if any(sentinel in item for item in strings):
                leak_channels.append(f"{channel_name}:{sentinel_name}")
    return leak_channels


def _sqlite_raw_text(db: Path) -> list[str]:
    return [db.read_bytes().decode("utf-8", errors="replace")]


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _walk_strings(item)
    if isinstance(value, list | tuple):
        for item in value:
            yield from _walk_strings(item)


def test_openai_metadata_redaction_is_recursive() -> None:
    recorded = {
        "trace": {
            "trace_id": "trace_recursive_redaction",
            "workflow_name": "redaction-test",
            "started_at": "2026-05-28T15:00:00Z",
            "ended_at": "2026-05-28T15:00:01Z",
        },
        "spans": [
            {
                "span_id": "span_recursive",
                "started_at": "2026-05-28T15:00:00Z",
                "ended_at": "2026-05-28T15:00:01Z",
                "span_data": {
                    "type": "custom",
                    "name": "redaction.step",
                    "data": {
                        "retry_count": 3,
                        "input": "nested secret prompt",
                        "nested": {"output": "nested secret response", "safe": "keep-me"},
                        "items": [{"input": "list secret"}],
                        "mcp_data": {"arguments": "secret mcp prompt"},
                        "payload": {"arguments": "secret custom args"},
                    },
                },
            }
        ],
    }

    bundle = bundle_from_recorded_trace(recorded)
    metadata_json = json.dumps(bundle.steps[0].metadata, sort_keys=True)

    assert "nested secret" not in metadata_json
    assert "list secret" not in metadata_json
    assert "secret mcp prompt" not in metadata_json
    assert "secret custom args" not in metadata_json
    assert "keep-me" not in metadata_json
    assert bundle.steps[0].metadata["retry_count"] == 3
    assert "input" not in bundle.steps[0].metadata
    assert "nested" not in bundle.steps[0].metadata
    assert "items" not in bundle.steps[0].metadata
    assert "mcp_data" not in bundle.steps[0].metadata
    assert "payload" not in bundle.steps[0].metadata
