import json
from pathlib import Path

import pytest

from agent_run_ledger.adapters.openai import (
    NoSpansCapturedError,
    OpenAILedgerTraceProcessor,
    bundle_from_recorded_trace,
)
from agent_run_ledger.core.storage import load_bundle


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


def test_bundle_from_recorded_trace_rejects_empty_spans() -> None:
    recorded = {"trace": {"trace_id": "trace_empty", "workflow_name": "empty"}}

    with pytest.raises(ValueError, match="recorded trace contains no spans"):
        bundle_from_recorded_trace(recorded)

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
                        "input": "nested secret prompt",
                        "nested": {"output": "nested secret response", "safe": "keep-me"},
                        "items": [{"input": "list secret"}],
                    },
                },
            }
        ],
    }

    bundle = bundle_from_recorded_trace(recorded)
    metadata_json = json.dumps(bundle.steps[0].metadata, sort_keys=True)

    assert "nested secret" not in metadata_json
    assert "list secret" not in metadata_json
    assert "keep-me" in metadata_json
    assert metadata_json.count("[redacted]") == 3
