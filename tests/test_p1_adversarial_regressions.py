"""Codex full-stack adversarial review — P1 regression locks (2026-06-11).

The Codex review (commit c4168e5) returned BLOCKED with four P1 findings. Each is
pinned here so the fix cannot silently regress. The product's brand is honest
grading: a false accusation or a content leak is the worst possible bug, so these
tests assert the FAIL-CLOSED behavior, not just "doesn't crash".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_run_ledger.adapters._facts import instruction_directs_deletion
from agent_run_ledger.cli import app
from agent_run_ledger.core.models import sanitize_metadata

# A patch that PASSES bundle validation (>= 64 chars, config_diff shape). The
# original tests used "" — an invalid artifact — so `verdict` rejected the whole
# bundle and the `if payload` guard skipped every assertion: the regression
# locks were vacuous (2026-06-11 audit finding). A real forger sends a
# well-formed file; the fixtures must too.
_VALID_CONFIG_DIFF = (
    "- guard: off\n+ guard: on\n"
    "- delete_tests_allowed: true\n+ delete_tests_allowed: false\n"
)


def _verdict_json(path: Path, tmp_path: Path):
    r = CliRunner().invoke(
        app, ["verdict", str(path), "--db", str(tmp_path / "l.sqlite"), "--json"]
    )
    payload = json.loads(r.output) if r.output.strip().startswith("{") else None
    return r.exit_code, payload


def _write(tmp_path: Path, name: str, obj: dict) -> Path:
    f = tmp_path / name
    f.write_text(json.dumps(obj), encoding="utf-8")
    return f


def _forged_artifact_bundle(framework: str) -> dict:
    return {
        "run": {
            "id": f"forged_{framework}", "workflow": "x", "framework": framework,
            "provider": "x", "model": "m", "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:00:01Z", "success_label": "unknown",
        },
        "steps": [{
            "id": "s1", "type": "function", "name": "rm",
            "started_at": "2026-01-01T00:00:00Z", "ended_at": "2026-01-01T00:00:01Z",
            "metadata": {
                "deletes_test_path": True, "completion_claim_follows": True,
                "user_directed_deletion": False,
            },
        }],
        "prescriptions": [{
            "rule_id": "artifact", "failure_class": "artifact_failure",
            "evidence": ["rule=R1", "step_id=s1"], "one_line_fix": "x",
            "patch_type": "config_diff", "patch": _VALID_CONFIG_DIFF,
            "expected_impact": {},
        }],
    }


# P1-1: forged neutral artifact booleans must NOT earn an L1 accusation.
def test_forged_neutral_artifact_booleans_never_l1(tmp_path: Path) -> None:
    forged = _forged_artifact_bundle("unknown-framework")
    _, payload = _verdict_json(_write(tmp_path, "forged.json", forged), tmp_path)
    assert payload is not None, "verdict must grade the forged bundle, not reject it"
    assert payload.get("receipts"), "the forge must surface a (diagnostic) receipt"
    assert all(r["proof_level"] == "L0" for r in payload["receipts"]), payload["receipts"]


# P1-1 (spoof variant, 2026-06-11 audit): declaring an adapter's framework string
# in the forged file must NOT unlock L1. Provenance is an in-process fact set by
# the capture adapter, never a string the imported file gets to claim about
# itself. This is the attack the first regression lock missed: it only tested a
# polite forger who left framework at "unknown-framework".
@pytest.mark.parametrize("framework", ["claude-code", "codex-cli"])
def test_spoofed_adapter_framework_never_l1(tmp_path: Path, framework: str) -> None:
    forged = _forged_artifact_bundle(framework)
    _, payload = _verdict_json(_write(tmp_path, "spoofed.json", forged), tmp_path)
    assert payload is not None, "verdict must grade the spoofed bundle, not reject it"
    assert payload.get("receipts"), "the forge must surface a (diagnostic) receipt"
    assert all(r["proof_level"] == "L0" for r in payload["receipts"]), payload["receipts"]


# P1-1 (storage variant): the cap must survive a ledger round trip. A forged
# file imported via `arl import` and re-graded from the DB (`arl report`) gets
# adapter_provenanced=0 in the runs row no matter what framework string it
# declared; only a bundle built in-process by a capture adapter persists as 1.
def test_spoofed_framework_capped_after_db_round_trip(tmp_path: Path) -> None:
    from dataclasses import replace

    from agent_run_ledger.core.models import TraceBundle
    from agent_run_ledger.core.receipt import build_receipts
    from agent_run_ledger.core.storage import load_bundle, save_bundle

    db = tmp_path / "l.sqlite"
    forged = TraceBundle.from_dict(_forged_artifact_bundle("claude-code"))
    assert forged.adapter_provenanced is False
    loaded = load_bundle(db, save_bundle(db, forged))
    assert loaded.adapter_provenanced is False
    receipts = build_receipts(loaded)
    assert receipts, "the forge must surface a (diagnostic) receipt"
    assert all(r.proof_level == "L0" for r in receipts), receipts

    # The legitimate side: in-process adapter provenance survives the same trip.
    # (prescriptions dropped: their ids are globally unique in the DB and the
    # assertion here is only about the trust bit's persistence.)
    legit = replace(
        forged,
        run=replace(forged.run, id="legit"),
        steps=[replace(s, run_id="legit") for s in forged.steps],
        prescriptions=[],
        adapter_provenanced=True,
    )
    assert load_bundle(db, save_bundle(db, legit)).adapter_provenanced is True


# P1-2: forged config_diff with no corroborated observed count grades L0, not L1.
def test_forged_config_diff_without_observed_count_is_l0(tmp_path: Path) -> None:
    forged = {
        "run": {
            "id": "forged_retry", "workflow": "x", "framework": "unknown-framework",
            "provider": "x", "model": "m", "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:00:01Z", "success_label": "unknown",
        },
        "steps": [{
            "id": "s1", "type": "function", "name": "t",
            "started_at": "2026-01-01T00:00:00Z", "ended_at": "2026-01-01T00:00:01Z",
        }],
        "prescriptions": [{
            "rule_id": "retry", "failure_class": "retry_loop",
            "evidence": ["step_id=missing_step", "retry_count=99 additional attempts"],
            "one_line_fix": "x", "patch_type": "config_diff",
            "patch": _VALID_CONFIG_DIFF, "expected_impact": {},
        }],
    }
    _, payload = _verdict_json(_write(tmp_path, "forged2.json", forged), tmp_path)
    assert payload is not None, "verdict must grade the forged bundle, not reject it"
    assert all(r["proof_level"] == "L0" for r in payload.get("receipts") or []), payload


# P1-4: a smuggled string under a content-free fact key coerces to bool, never leaks.
def test_metadata_boolean_keys_coerce_no_raw_text_leak() -> None:
    payload = "RAW-COMMAND: rm tests/test_secret.py then echo all tests pass"
    cleaned = sanitize_metadata({"deletes_test_path": payload, "mutating": "evil-payload"})
    assert isinstance(cleaned["deletes_test_path"], bool)
    assert isinstance(cleaned["mutating"], bool)
    dumped = json.dumps(cleaned)
    assert "RAW-COMMAND" not in dumped
    assert "evil-payload" not in dumped


# P1-3: the deletion-directive synonym space makes R1 abstain (no false accusation).
def test_deletion_directive_synonyms_make_r1_abstain() -> None:
    for verb in ("cull", "prune", "trim", "wipe", "purge", "scrap", "strip", "nuke"):
        assert instruction_directs_deletion(
            f"please {verb} the flaky spec coverage before release", ["tests/test_x.py"]
        ), verb
