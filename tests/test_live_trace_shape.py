"""Task 45 — the live OpenAI Agents SDK trace shape, exercised against the adapter.

These fixtures are hand-written in the EXACT export shape the installed SDK
produces (verified against .venv/.../agents/tracing/spans.py + span_data.py +
usage.py), NOT the simplified recorded fixture. They pin the real $0 root cause
and prove the fix.

Ground truth (SDK source):
  Span.export() -> {"object":"trace.span","id":<span_id>,"trace_id":...,
                    "parent_id":...,"started_at":...,"ended_at":...,
                    "span_data":<span_data.export()>,"error":null-or-dict}
  ResponseSpanData.export() -> {"type":"response","response_id":...,"usage":{...}}
    -> NO model field, NO name field on a response span.
  model_usage_to_span_usage() -> {"requests":1,"input_tokens":N,"output_tokens":M,
    "total_tokens":..,"input_tokens_details":{"cached_tokens":C},
    "output_tokens_details":{"reasoning_tokens":R}}

The default Runner.run uses the Responses API, which emits ONLY response spans
(openai_responses.py) — so a real run carries tokens but NO discoverable model,
which is why _compute_from_tokens returned None -> cost fell back to the $0 cache.
"""

from __future__ import annotations

from agent_run_ledger.adapters.openai import bundle_from_recorded_trace
from agent_run_ledger.core.cost import cost_on_read


def _response_span(
    span_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    started_at: str = "2026-05-31T10:00:00Z",
    ended_at: str = "2026-05-31T10:00:02Z",
    parent_id: str | None = "span_agent_root",
    error: dict | None = None,
) -> dict:
    """A span in the REAL Span.export() shape with a ResponseSpanData payload."""
    return {
        "object": "trace.span",
        "id": span_id,
        "trace_id": "trace_live_0123456789abcdef",
        "parent_id": parent_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "span_data": {
            "type": "response",
            "response_id": f"resp_{span_id}",
            "usage": {
                "requests": 1,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_tokens_details": {"cached_tokens": cached_tokens},
                "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        },
        "error": error,
    }


def _live_trace(spans: list[dict]) -> dict:
    return {
        "trace": {
            "trace_id": "trace_live_0123456789abcdef",
            "workflow_name": "live-responses-agent",
            "started_at": "2026-05-31T10:00:00Z",
            "ended_at": "2026-05-31T10:00:05Z",
        },
        "spans": spans,
    }


# --- 2a: the $0 root cause is model-unknown, NOT missing tokens ----------------


def test_live_response_span_tokens_extract_but_cost_is_zero_without_model() -> None:
    """Pins the real bug + corrects the handoff: tokens DO extract from a live
    response span; cost is $0 only because no span carries a model, so
    _compute_from_tokens returns None and falls back to the $0 cache."""
    bundle = bundle_from_recorded_trace(
        _live_trace([_response_span("span_resp_1", input_tokens=1200, output_tokens=300)])
    )

    # tokens extract fine from span_data.usage (handoff's "tokens" cause is wrong)
    assert bundle.run.total_input_tokens == 1200
    assert bundle.run.total_output_tokens == 300
    # but the response span exports no model -> unknown
    assert bundle.run.model == "unknown"
    # -> cost_on_read falls back to the $0 cache: the real symptom
    assert cost_on_read(bundle) == 0.0


def test_model_hint_makes_cost_on_read_nonzero() -> None:
    """The app knows its model when it calls Runner.run; the processor accepts a
    model hint used only when no span carries one. With it, cost_on_read > 0."""
    bundle = bundle_from_recorded_trace(
        _live_trace([_response_span("span_resp_1", input_tokens=1200, output_tokens=300)]),
        model="gpt-4o-mini",
    )

    assert bundle.run.model == "gpt-4o-mini"
    assert cost_on_read(bundle) > 0.0


def test_trace_extracted_model_wins_over_hint() -> None:
    """A span that DOES carry a model (e.g. Chat Completions generation span)
    takes precedence over the hint — the hint is a fallback, not an override."""
    trace = _live_trace(
        [
            {
                "object": "trace.span",
                "id": "span_gen_1",
                "trace_id": "trace_live_0123456789abcdef",
                "parent_id": "span_agent_root",
                "started_at": "2026-05-31T10:00:00Z",
                "ended_at": "2026-05-31T10:00:02Z",
                "span_data": {
                    "type": "generation",
                    "model": "gpt-4o",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
                "error": None,
            }
        ]
    )

    bundle = bundle_from_recorded_trace(trace, model="gpt-4o-mini")

    assert bundle.run.model == "gpt-4o"


# --- 2b: cached/reasoning tokens are captured (un-backfillable facts) ----------


def test_live_response_span_captures_cached_and_reasoning_tokens() -> None:
    bundle = bundle_from_recorded_trace(
        _live_trace(
            [
                _response_span(
                    "span_resp_1",
                    input_tokens=1000,
                    output_tokens=200,
                    cached_tokens=400,
                    reasoning_tokens=150,
                )
            ]
        ),
        model="gpt-4o-mini",
    )

    step = bundle.steps[0]
    assert step.cached_input_tokens == 400
    assert step.reasoning_tokens == 150


def test_cost_on_read_reflects_reasoning_surcharge_and_cached_discount() -> None:
    """Reasoning tokens bill at the OUTPUT rate; cached input bills at the cheaper
    cached rate. A bundle with reasoning/cached must cost MORE/differently than the
    same token totals with neither — proving the facts actually flow into cost."""
    plain = bundle_from_recorded_trace(
        _live_trace([_response_span("s1", input_tokens=1000, output_tokens=200)]),
        model="gpt-4o-mini",
    )
    with_details = bundle_from_recorded_trace(
        _live_trace(
            [
                _response_span(
                    "s1",
                    input_tokens=1000,
                    output_tokens=200,
                    cached_tokens=400,
                    reasoning_tokens=150,
                )
            ]
        ),
        model="gpt-4o-mini",
    )

    # reasoning adds 150 output-rate tokens; cached moves 400 input tokens to the
    # cheaper cached rate. Net for gpt-4o-mini (out >> in > cached) is HIGHER.
    assert cost_on_read(with_details) > cost_on_read(plain)
