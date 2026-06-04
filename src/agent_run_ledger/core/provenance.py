"""Provenance hash (L5) — a local, deterministic fingerprint of immutable facts.

The provenance hash is the seed of the future proof-of-real brick. It is the ONE
thing impossible to backfill, so it is computed LOCALLY at capture over EXACTLY
the immutable fact fields and nothing derived.

Contract (pinned by a golden test in tests/test_provenance.py):
  * Algorithm-prefixed value ``"sha256:<lowercase-hex>"`` — the hash function is
    itself versioned, so a future construction change is visible in the value.
  * Preimage = a sorted-key JSON serialization of:
      id, workflow, model, provider, billing_mode, prompt_hash, config_hash,
      started_at, ended_at, and the ORDERED list of step tuples
      (id, parent_step_id, step_type, input_tokens, output_tokens).
  * Steps are ordered DETERMINISTICALLY by (started_at, id) before hashing —
    never insertion order.
  * EXCLUDES total_cost_usd, cost_usd, cached/reasoning tokens, latency, and
    every derived / judgment field. A hash that included cost would break on the
    next price-table update; the whole point is a stable fact-fingerprint while
    judgments recompute.
  * Separators are pinned so whitespace cannot drift the digest across versions;
    None serializes as JSON null.

Computed locally; hashing performs zero network egress.
"""

from __future__ import annotations

import hashlib
import json

from .models import RunRecord, StepRecord, TraceBundle

ALGORITHM = "sha256"


def _step_tuple(step: StepRecord) -> list:
    """The immutable, structural facts of a step (no content, no derived cost)."""
    return [
        step.id,
        step.parent_step_id,
        step.step_type,
        step.input_tokens,
        step.output_tokens,
    ]


def _preimage(run: RunRecord, steps: tuple[StepRecord, ...] | list[StepRecord]) -> dict:
    ordered = sorted(steps, key=lambda s: (s.started_at, s.id))
    return {
        "id": run.id,
        "workflow": run.workflow,
        "model": run.model,
        "provider": run.provider,
        "billing_mode": run.billing_mode,
        "prompt_hash": run.prompt_hash,
        "config_hash": run.config_hash,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "steps": [_step_tuple(s) for s in ordered],
    }


def compute_provenance_hash(bundle: TraceBundle) -> str:
    """Return the algorithm-prefixed provenance hash for *bundle* (L5)."""
    preimage = _preimage(bundle.run, bundle.steps)
    canonical = json.dumps(
        preimage, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{ALGORITHM}:{digest}"
