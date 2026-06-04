"""Demo: a REAL captured agent retry loop -> an HONEST RepairReceipt at its TRUE level.

The GATE drives a REAL OpenAI Agents SDK run (stub Model, no API key, no network —
the way Codex drove it). The native SDK function span exports only
``type/name/input/output/mcp_data`` — NO ``retry_budget_patch_target`` — so a real
captured retry loop honestly grades **L1**: ARL proves the retry-loop DETECTION; the
cheap-strong L2 static-repair proof needs app instrumentation that supplies a safe
patch target. The GATE line prints the level of the REAL run, not a fixture.

Two sections:
  1. GATE (load-bearing): a REAL captured SDK run -> honest L1 receipt.
  2. Illustration (NOT the gate): the same pipeline on a hand-instrumented fixture
     that DOES carry a patch target, to show what an L2 receipt looks like once an
     app supplies one. This is explicitly labelled synthetic so it can never be
     mistaken for the real-run gate.

Run:  uv run python scripts/demo_repair_receipt.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from agent_run_ledger.adapters.openai import bundle_from_recorded_trace
from agent_run_ledger.core.cost import cost_on_read
from agent_run_ledger.core.prescriptions import analyze_bundle
from agent_run_ledger.core.receipt import build_receipts

ILLUSTRATION_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "instrumented_retry_loop_l2.json"
)


def _print_receipt(r) -> None:
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


def _gate_real_capture() -> str:
    """Drive the REAL SDK with a stub Model and return the receipt's proof level."""
    # live_capture_harness lives beside this script; reuse the SAME harness the B1
    # gate test asserts on, so the demo's GATE line and the test share one real run.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from live_capture_harness import capture_real_retry_run

    tmp = Path(tempfile.mkdtemp())
    bundle, prescriptions, receipts = capture_real_retry_run(tmp)

    print("=" * 70)
    print("GATE — REAL captured OpenAI Agents SDK run (stub Model, no API key)")
    print("=" * 70)
    print(f"  run id          : {bundle.run.id}")
    print(f"  model (hint)    : {bundle.run.model}")
    print(f"  provenance_hash : {bundle.run.provenance_hash}")
    print(f"  cost_on_read    : ${cost_on_read(bundle):.6f}")
    print(f"  captured spans  : {len(bundle.steps)}")
    fn_steps = [s for s in bundle.steps if s.span_kind == "function"]
    print(
        f"  function spans  : {len(fn_steps)} (distinct turn parents="
        f"{len({s.parent_step_id for s in fn_steps})}, one agent scope="
        f"{len({s.retry_scope for s in fn_steps}) == 1})"
    )
    print(f"  prescriptions   : {len(prescriptions)}")
    print()
    print(f"REPAIR RECEIPT(S)  ({len(receipts)})  — JUDGMENT computed on read")
    for r in receipts:
        _print_receipt(r)
    return receipts[0].proof_level if receipts else "L0"


def _illustration_instrumented_l2() -> str:
    """The same pipeline on a HAND-INSTRUMENTED fixture that supplies a safe patch
    target. NOT the gate — it shows what L2 looks like once an app instruments the
    trace. Returns the receipt level."""
    recorded = json.loads(ILLUSTRATION_FIXTURE.read_text(encoding="utf-8"))
    bundle = bundle_from_recorded_trace(recorded, model="gpt-4o-mini")
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    receipts = build_receipts(bundle)
    print()
    print("=" * 70)
    print("ILLUSTRATION (NOT the gate) — hand-instrumented fixture WITH a patch")
    print("target, to show what L2 looks like once an app supplies one")
    print("=" * 70)
    for r in receipts:
        _print_receipt(r)
    return receipts[0].proof_level if receipts else "L0"


def main() -> None:
    real_level = _gate_real_capture()
    illustration_level = _illustration_instrumented_l2()

    # The gate reflects the REAL run's HONEST level. A native SDK retry loop proves
    # L1 (detection); L2 needs the tested app-instrumentation path shown in the
    # illustration. The gate is NOT "reaches L2" — it is "honest level on a real run".
    assert real_level == "L1", f"expected the real captured run to grade L1, got {real_level}"
    assert illustration_level == "L2", (
        f"expected the instrumented illustration to reach L2, got {illustration_level}"
    )
    print()
    print(
        f">>> GATE: a REAL captured SDK retry loop produces an HONEST {real_level} receipt "
        "(detection proven; L2 needs app instrumentation, shown above as illustration)."
    )


if __name__ == "__main__":
    main()
