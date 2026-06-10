"""`arl verdict` — the loop verdict layer (Task 57 P1/P2).

The contract a loop gates on (README "loop contract"):

  exit 0  clean   — no structural failure detected (the honest negative)
  exit 3  fired   — at least one repair receipt (attention)
  exit 1  error   — unreadable/invalid input fails CLOSED: an unparseable run is
                    NOT verified (it must never read as clean)
  (exit 2 is reserved by click/typer for usage errors)

`--json` emits a stable machine schema (``arl.verdict/v1``) on stdout so a hook,
CI step, or Ralph-style loop can consume the receipt without scraping prose.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from agent_run_ledger.cli import app
from agent_run_ledger.core.receipt import PROOF_LEVELS

CODEX_FIXTURES = Path(__file__).parent / "fixtures" / "codex"


def _invoke(args: list[str]):
    return CliRunner().invoke(app, args)


# ---------------------------------------------------------------- exit codes


def test_verdict_clean_run_exits_zero(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    result = _invoke(["verdict", "fixtures/clean_run.json", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "clean" in result.output.lower()
    # The honest negative must never render as an empty/blank success.
    assert "no structural failure" in result.output.lower()


def test_verdict_retry_loop_fires_exit_3(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    result = _invoke(["verdict", "fixtures/golden_retry_loop.json", "--db", str(db)])
    assert result.exit_code == 3, result.output
    assert "retry_loop" in result.output
    # A proof level is always shown — the grade IS the product.
    assert any(level in result.output for level in PROOF_LEVELS)


def test_verdict_malformed_file_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "garbage.json"
    bad.write_text("{not json", encoding="utf-8")
    result = _invoke(["verdict", str(bad), "--db", str(tmp_path / "l.sqlite")])
    assert result.exit_code == 1, result.output
    assert "error" in result.output.lower()


def test_verdict_missing_path_and_no_latest_is_an_error(tmp_path: Path) -> None:
    result = _invoke(["verdict", "--db", str(tmp_path / "l.sqlite")])
    assert result.exit_code == 1, result.output
    assert "error" in result.output.lower()


def test_verdict_directory_input_fails_closed_no_traceback(tmp_path: Path) -> None:
    """Codex advisor finding (2026-06-10): a directory produced a raw traceback.
    Fail-closed means TYPED error + exit 1 on every unreadable input shape."""
    some_dir = tmp_path / "a_directory"
    some_dir.mkdir()
    result = _invoke(["verdict", str(some_dir), "--db", str(tmp_path / "l.sqlite")])
    assert result.exit_code == 1, result.output
    assert "error" in result.output.lower()
    assert "Traceback" not in result.output


# ------------------------------------------------------- coverage honesty


def test_verdict_json_states_detector_coverage(tmp_path: Path) -> None:
    """Gauntlet convergent fix #1: 'clean' must say what was checked, or exit 0
    launders unverified work as verified (anti-AI persona: 'a green checkmark
    from a single-detector tool is active laundering of slop')."""
    db = tmp_path / "ledger.sqlite"
    result = _invoke(["verdict", "fixtures/clean_run.json", "--db", str(db), "--json"])
    payload = json.loads(result.output)
    assert "coverage" in payload
    assert any("retry" in c for c in payload["coverage"]["checked"])
    assert payload["coverage"]["not_checked"], "unchecked classes must be named"


def test_verdict_clean_human_output_names_unchecked_classes(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    result = _invoke(["verdict", "fixtures/clean_run.json", "--db", str(db)])
    assert result.exit_code == 0
    out = result.output.lower()
    assert "checked" in out and "not checked" in out


# ------------------------------------------------------------- selftest


def test_selftest_fires_a_graded_receipt() -> None:
    """Gauntlet convergent fix #2 (burned-skeptic): users must be able to SEE the
    alarm fire in minute one, or 'clean' is indistinguishable from 'deaf'."""
    result = _invoke(["selftest"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert "receipt fired" in result.output
    assert any(level in result.output for level in PROOF_LEVELS)


# ---------------------------------------------------------------- json schema


def test_verdict_json_receipts_schema(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    result = _invoke(["verdict", "fixtures/golden_retry_loop.json", "--db", str(db), "--json"])
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "arl.verdict/v1"
    assert payload["run_id"] == "run_retry_loop"
    assert payload["verdict"] == "receipts"
    assert payload["receipt_count"] >= 1
    assert payload["max_proof_level"] in PROOF_LEVELS
    receipt = payload["receipts"][0]
    # The receipt's honesty surface travels whole: grade + limits + next steps.
    for key in (
        "claim",
        "observed_failure",
        "proof_level",
        "confidence",
        "limits",
        "next_evidence",
        "repair_artifact",
    ):
        assert key in receipt, f"receipt missing {key}"
    assert receipt["observed_failure"] == "retry_loop"


def test_verdict_json_clean_schema(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    result = _invoke(["verdict", "fixtures/clean_run.json", "--db", str(db), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "arl.verdict/v1"
    assert payload["verdict"] == "clean"
    assert payload["receipt_count"] == 0
    assert payload["receipts"] == []
    assert payload["max_proof_level"] is None


def test_verdict_json_l2_on_instrumented_fixture(tmp_path: Path) -> None:
    """The instrumented fixture carries a patch target -> the applyable diff grades L2."""
    db = tmp_path / "ledger.sqlite"
    result = _invoke(
        ["verdict", "fixtures/instrumented_retry_loop_l2.json", "--db", str(db), "--json"]
    )
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["max_proof_level"] == "L2"


# ---------------------------------------------------------------- persistence


def test_verdict_is_idempotent_across_invocations(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    first = _invoke(["verdict", "fixtures/golden_retry_loop.json", "--db", str(db)])
    second = _invoke(["verdict", "fixtures/golden_retry_loop.json", "--db", str(db)])
    assert first.exit_code == 3
    assert second.exit_code == 3, second.output
    assert "Traceback" not in second.output
    assert "RunAlreadyRecorded" not in second.output


def test_verdict_saves_to_ledger_by_default(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    _invoke(["verdict", "fixtures/golden_retry_loop.json", "--db", str(db)])
    listed = _invoke(["list-runs", "--db", str(db)])
    assert "run_retry_loop" in listed.output


def test_verdict_no_save_leaves_ledger_untouched(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    _invoke(["init", "--db", str(db)])
    result = _invoke(
        ["verdict", "fixtures/golden_retry_loop.json", "--db", str(db), "--no-save"]
    )
    assert result.exit_code == 3
    listed = _invoke(["list-runs", "--db", str(db)])
    assert "run_retry_loop" not in listed.output


# ---------------------------------------------------------------- --latest (P2)


def _sessions_tree(tmp_path: Path) -> Path:
    """A fake ~/.codex/sessions tree: two dated rollouts, newest must win."""
    root = tmp_path / "sessions"
    old_dir = root / "2026" / "06" / "01"
    new_dir = root / "2026" / "06" / "09"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    fixture = CODEX_FIXTURES / "fire_no_edit_retry.jsonl"
    shutil.copy(fixture, old_dir / "rollout-2026-06-01T08-00-00-aaa.jsonl")
    shutil.copy(fixture, new_dir / "rollout-2026-06-09T21-30-00-bbb.jsonl")
    return root


def test_find_recent_rollouts_newest_first(tmp_path: Path) -> None:
    from agent_run_ledger.adapters.codex import find_recent_rollouts

    root = _sessions_tree(tmp_path)
    found = find_recent_rollouts(root)
    assert [p.name for p in found[:2]] == [
        "rollout-2026-06-09T21-30-00-bbb.jsonl",
        "rollout-2026-06-01T08-00-00-aaa.jsonl",
    ]


def test_find_recent_rollouts_missing_root_is_empty(tmp_path: Path) -> None:
    from agent_run_ledger.adapters.codex import find_recent_rollouts

    assert find_recent_rollouts(tmp_path / "nope") == []


def test_verdict_latest_grades_newest_session(tmp_path: Path) -> None:
    root = _sessions_tree(tmp_path)
    db = tmp_path / "ledger.sqlite"
    result = _invoke(
        ["verdict", "--latest", "--sessions-root", str(root), "--db", str(db), "--json"]
    )
    assert result.exit_code in (0, 3), result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "arl.verdict/v1"
    # the fire fixture detects a retry loop -> receipts
    assert payload["verdict"] == "receipts"


def test_verdict_latest_empty_root_fails_closed(tmp_path: Path) -> None:
    empty = tmp_path / "sessions"
    empty.mkdir()
    result = _invoke(
        ["verdict", "--latest", "--sessions-root", str(empty), "--db", str(tmp_path / "l.sqlite")]
    )
    assert result.exit_code == 1, result.output
    assert "error" in result.output.lower()


def test_verdict_exit_code_survives_broken_pipe(tmp_path: Path, monkeypatch) -> None:
    """The exit contract must survive a consumer that closes the pipe early
    (`| head`, crashed hook): write failure is swallowed, exit code kept."""
    import typer as _typer

    def _boom(*args, **kwargs):
        raise BrokenPipeError

    monkeypatch.setattr(_typer, "echo", _boom)
    result = _invoke(
        ["verdict", "fixtures/golden_retry_loop.json", "--db", str(tmp_path / "l.sqlite"), "--json"]
    )
    assert result.exit_code == 3, result.output
