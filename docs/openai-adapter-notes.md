# OpenAI Adapter Notes

Checked: 2026-05-28

Sources used:
- OpenAI Agents SDK tracing docs
- Installed `openai-agents` package via `uv run --extra openai`
- Local source under `.venv/Lib/site-packages/agents/tracing`

## SDK hook

The adapter uses the public tracing processor surface:

- `add_trace_processor(processor)` registers an additional processor.
- A processor receives `on_trace_start`, `on_trace_end`, `on_span_start`, `on_span_end`,
  `force_flush`, and `shutdown`.
- Processor methods are synchronous and must return quickly.

## Trace shape

Current SDK traces export fields shaped like:

- `trace_id`
- `workflow_name`
- `group_id`
- `metadata`

The adapter maps:

- `trace_id` -> `RunRecord.id`
- `workflow_name` -> `RunRecord.workflow`
- provider -> `openai`
- framework -> `openai-agents-python`

## Span shape

Current SDK spans export fields shaped like:

- `trace_id`
- `span_id`
- `parent_id`
- `started_at`
- `ended_at`
- `span_data`
- `error`

The `span_data` object exports by type:

- `AgentSpanData`: `type`, `name`, `handoffs`, `tools`, `output_type`
- `GenerationSpanData`: `type`, `input`, `output`, `model`, `model_config`, `usage`
- `FunctionSpanData`: `type`, `name`, `input`, `output`, `mcp_data`
- `CustomSpanData`: `type`, `name`, `data`

The adapter recursively redacts raw `input` and `output` fields from stored metadata, including nested dictionaries and list entries under custom span data.

## Cost and retries

The SDK generation span exposes token usage, not price. Agent Run Ledger therefore treats
real SDK price as unknown unless a custom span supplies `cost_usd`.

Retry counts are not a native SDK field. For V0, retry counts come from custom span data:

```python
custom_span(
    "crm.lookup_customer",
    {
        "arl_step_type": "tool",
        "retry_count": 3,
        "cost_usd": 0.04,
    },
)
```

This is not pretending the SDK emits retry budgets. It is the explicit app-level
instrumentation path needed for the retry/cost-loop prescription.

## Failure mode

Zero captured spans is an instrumentation failure. The adapter raises
`NoSpansCapturedError` instead of fabricating an `empty_trace` run.
