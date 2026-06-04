"""A REAL OpenAI Agents SDK capture harness — stub Model, no API key, no network.

Shared by the B1 gate test (tests/test_live_capture_receipt.py) and the demo
(scripts/demo_repair_receipt.py) so BOTH exercise the SAME real-run path: the demo's
GATE line and the test assert the same captured receipt level. The stub Model issues
the same erroring tool call across turns (driving a genuine cross-turn retry loop),
then a final answer. The captured tree is the real ``task -> agent -> turn ->
function`` shape; native function spans carry NO patch-target metadata, so the
receipt honestly grades L1.
"""

from __future__ import annotations

from pathlib import Path

import agents
from agents import Agent, ModelSettings, Runner, function_tool
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from agent_run_ledger.adapters.openai import make_trace_processor
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.receipt import build_receipts
from agent_run_ledger.core.storage import load_bundle


@function_tool
def crm_lookup(customer_id: int) -> str:
    """Look up a customer in the CRM. Always raises, to drive a retry loop."""
    raise RuntimeError("CRM upstream timeout")


def _usage() -> Usage:
    return Usage(
        requests=1,
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


class StubRetryModel(Model):
    """First 3 turns: the SAME crm_lookup tool call (the tool errors -> the agent
    retries). 4th turn: a final text answer. No API key, no network."""

    def __init__(self) -> None:
        self._turn = 0

    async def get_response(self, *args, **kwargs) -> ModelResponse:
        self._turn += 1
        if self._turn <= 3:
            call = ResponseFunctionToolCall(
                type="function_call",
                call_id=f"call_{self._turn}",
                name="crm_lookup",
                arguments='{"customer_id": 42}',
            )
            return ModelResponse(output=[call], usage=_usage(), response_id=f"resp_{self._turn}")
        msg = ResponseOutputMessage(
            id="msg_final",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text="done", annotations=[])],
        )
        return ModelResponse(output=[msg], usage=_usage(), response_id="resp_final")

    def stream_response(self, *args, **kwargs):  # pragma: no cover - run_sync uses get_response
        raise NotImplementedError


def capture_real_retry_run(tmp_path: Path):
    """Drive the REAL SDK with the stub model; return (bundle, prescriptions,
    receipts) from the captured run. Saves/restores the global trace processors."""
    db = Path(tmp_path) / "live.sqlite"
    processor = make_trace_processor(db, model="gpt-4o-mini")
    saved = agents.tracing.get_trace_provider()._multi_processor._processors  # type: ignore[attr-defined]
    agents.set_trace_processors([processor])
    agents.set_tracing_disabled(False)
    try:
        agent = Agent(
            name="Support Agent",
            instructions="Look up the customer.",
            tools=[crm_lookup],
            model=StubRetryModel(),
            model_settings=ModelSettings(),
        )
        Runner.run_sync(agent, "look up customer 42", max_turns=8)
    finally:
        agents.set_trace_processors(list(saved))
    bundle = load_bundle(db, processor._trace_id)
    return bundle, analyze_bundle(bundle), build_receipts(bundle)
