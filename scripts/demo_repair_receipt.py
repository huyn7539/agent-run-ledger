"""Demo: a REAL-shape agent trace -> a RepairReceipt that honestly reaches L2.

This is the artifact for the customer conversations. It drives the live-shape
fixture (fixtures/live_retry_loop_interleaved.json) through the full pipeline:

    real interleaved retry loop (no retry_count field in the trace)
      -> adapter DERIVES retry_count from repeated same-input failing tool spans
      -> retry/cost detector fires
      -> RepairReceipt graded L2 (the templated retry-cap diff mechanically
         removes the unbounded-retry path, provable WITHOUT a re-run)

Run:  uv run python scripts/demo_repair_receipt.py
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_run_ledger.adapters.openai import bundle_from_recorded_trace
from agent_run_ledger.core.cost import cost_on_read
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.receipt import build_receipts

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "live_retry_loop_interleaved.json"


def main() -> None:
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # The app knows its model when it calls Runner.run; the SDK Responses API does
    # not serialize it into the trace, so it is supplied as a hint.
    bundle = bundle_from_recorded_trace(recorded, model="gpt-4o-mini")
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))

    print("=" * 70)
    print("BASE RECORD (immutable FACTS)")
    print("=" * 70)
    print(f"  run id          : {bundle.run.id}")
    print(f"  model           : {bundle.run.model}")
    print(f"  provenance_hash : {bundle.run.provenance_hash}")
    print(f"  cost_on_read    : ${cost_on_read(bundle):.6f}")
    print(f"  steps           : {len(bundle.steps)}")
    for s in bundle.steps:
        print(
            f"     {s.id:<14} kind={s.span_kind or '-':<9} "
            f"retry_count={s.retry_count} error_class={s.error_class}"
        )
    print(f"  prescriptions   : {len(bundle.prescriptions)}")

    receipts = build_receipts(bundle)
    print()
    print("=" * 70)
    print(f"REPAIR RECEIPT(S)  ({len(receipts)})  — JUDGMENT computed on read")
    print("=" * 70)
    for r in receipts:
        print(f"  observed_failure : {r.observed_failure}")
        print(f"  PROOF LEVEL      : {r.proof_level}   confidence: {r.confidence}")
        print(f"  claim            : {r.claim}")
        print("  evidence         :")
        for e in r.evidence:
            print(f"     - {e}")
        print(
            f"  repair_artifact  : templated={r.repair_artifact['templated']} "
            f"patch_type={r.repair_artifact['patch_type']}"
        )
        print("  outcome_delta    :")
        for k, v in r.outcome_delta.items():
            print(f"     - {k}: {v}")
        print("  limits           :")
        for limit in r.limits:
            print(f"     - {limit}")
        print("  next_evidence    :")
        for n in r.next_evidence:
            print(f"     - {n}")

    assert receipts, "expected at least one receipt"
    assert receipts[0].proof_level == "L2", f"expected L2, got {receipts[0].proof_level}"
    print()
    print(">>> GATE: one real receipt honestly reaches L2 on a real retry loop. OK")


if __name__ == "__main__":
    main()
