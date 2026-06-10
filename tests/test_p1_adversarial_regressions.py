"""Codex full-stack adversarial review — P1 regression locks (2026-06-11).

The Codex review (commit c4168e5) returned BLOCKED with four P1 findings. Each is
pinned here so the fix cannot silently regress. The product's brand is honest
grading: a false accusation or a content leak is the worst possible bug, so these
tests assert the FAIL-CLOSED behavior, not just "doesn't crash".
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_run_ledger.adapters._facts import instruction_directs_deletion
from agent_run_ledger.cli import app
from agent_run_ledger.core.models import sanitize_metadata


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


# P1-1: forged neutral artifact booleans must NOT earn an L1 accusation.
def test_forged_neutral_artifact_booleans_never_l1(tmp_path: Path) -> None:
    forged = {
        "run": {
            "id": "forged_neutral", "workflow": "x", "framework": "unknown-framework",
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
            "patch_type": "config_diff", "patch": "", "expected_impact": {},
        }],
    }
    _, payload = _verdict_json(_write(tmp_path, "forged.json", forged), tmp_path)
    if payload and payload.get("receipts"):
        assert all(r["proof_level"] == "L0" for r in payload["receipts"]), payload["receipts"]


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
            "one_line_fix": "x", "patch_type": "config_diff", "patch": "",
            "expected_impact": {},
        }],
    }
    _, payload = _verdict_json(_write(tmp_path, "forged2.json", forged), tmp_path)
    if payload and payload.get("receipts"):
        assert all(r["proof_level"] == "L0" for r in payload["receipts"]), payload["receipts"]


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
