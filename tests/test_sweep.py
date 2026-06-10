"""`arl sweep <root>` — batch-verdict an archive of session logs (Task 58 C).

The burned-skeptic's flip: "the archive sweep IS the demo — every prospect
already owns months of evidence." Contract:

  exit 0  no receipts anywhere (clean for the checked classes)
  exit 3  at least one file fired
  exit 1  total failure (root missing / nothing found / every file unreadable)

Read-only by default (--no-save): a sweep must never write the ledger unless
asked. `--json` emits the stable ``arl.sweep/v1`` schema.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from agent_run_ledger.cli import app

FIXTURES = Path(__file__).parent / "fixtures" / "claude_code"
CODEX_FIXTURES = Path(__file__).parent / "fixtures" / "codex"


def _invoke(args: list[str]):
    return CliRunner().invoke(app, args)


def _archive(tmp_path: Path, names: list[str]) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    for name in names:
        src = FIXTURES / name if (FIXTURES / name).exists() else CODEX_FIXTURES / name
        shutil.copy(src, root / name)
    return root


# ---------------------------------------------------------------- exit codes


def test_sweep_clean_archive_exits_zero(tmp_path: Path) -> None:
    root = _archive(tmp_path, ["clean_session.jsonl", "qa_session.jsonl"])
    result = _invoke(["sweep", str(root), "--db", str(tmp_path / "l.sqlite")])
    assert result.exit_code == 0, result.output
    assert "clean=2" in result.output.replace(" ", "")


def test_sweep_fired_archive_exits_three(tmp_path: Path) -> None:
    root = _archive(tmp_path, ["clean_session.jsonl", "lie_test_deletion_session.jsonl"])
    result = _invoke(["sweep", str(root), "--db", str(tmp_path / "l.sqlite")])
    assert result.exit_code == 3, result.output
    assert "lie_test_deletion_session.jsonl" in result.output


def test_sweep_missing_root_fails_closed(tmp_path: Path) -> None:
    result = _invoke(["sweep", str(tmp_path / "nope"), "--db", str(tmp_path / "l.sqlite")])
    assert result.exit_code == 1, result.output
    assert "error" in result.output.lower()


def test_sweep_empty_root_fails_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _invoke(["sweep", str(empty), "--db", str(tmp_path / "l.sqlite")])
    assert result.exit_code == 1, result.output


def test_sweep_all_unreadable_is_total_failure(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "garbage.jsonl").write_text("{not json\n", encoding="utf-8")
    result = _invoke(["sweep", str(root), "--db", str(tmp_path / "l.sqlite")])
    assert result.exit_code == 1, result.output


# ---------------------------------------------------------------- json schema


def test_sweep_json_schema(tmp_path: Path) -> None:
    root = _archive(
        tmp_path,
        [
            "clean_session.jsonl",
            "lie_test_deletion_session.jsonl",
            "fire_test_deletion_lie.jsonl",
        ],
    )
    result = _invoke(["sweep", str(root), "--db", str(tmp_path / "l.sqlite"), "--json"])
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "arl.sweep/v1"
    assert payload["scanned"] == 3
    assert payload["counts"] == {"clean": 1, "fired": 2, "no_run": 0, "error": 0}
    fired_paths = {Path(f["path"]).name for f in payload["fired"]}
    assert fired_paths == {"lie_test_deletion_session.jsonl", "fire_test_deletion_lie.jsonl"}
    for entry in payload["fired"]:
        assert entry["receipt_count"] >= 1
        assert entry["max_proof_level"]
        assert "artifact_failure" in entry["observed_failures"]


def test_sweep_json_counts_no_run_and_error(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    shutil.copy(FIXTURES / "clean_session.jsonl", root / "clean.jsonl")
    # a real-shaped chat-only session: no tool calls -> "no run to record"
    lines = (FIXTURES / "qa_session.jsonl").read_text(encoding="utf-8").splitlines()
    chat_only = [json.loads(ln) for ln in lines]
    chat_only = [
        rec
        for rec in chat_only
        if not any(
            isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
            for b in (rec.get("message", {}) or {}).get("content", [])
            if isinstance((rec.get("message", {}) or {}).get("content"), list)
        )
    ]
    (root / "chat_only.jsonl").write_text(
        "\n".join(json.dumps(rec) for rec in chat_only) + "\n", encoding="utf-8"
    )
    (root / "garbage.jsonl").write_text("{not json\n", encoding="utf-8")
    result = _invoke(["sweep", str(root), "--db", str(tmp_path / "l.sqlite"), "--json"])
    payload = json.loads(result.output)
    assert payload["counts"]["no_run"] == 1
    assert payload["counts"]["error"] == 1
    assert payload["counts"]["clean"] == 1
    assert result.exit_code == 0, result.output  # errors reported, not total failure


# ---------------------------------------------------------------- read-only default


def test_sweep_is_read_only_by_default(tmp_path: Path) -> None:
    root = _archive(tmp_path, ["lie_test_deletion_session.jsonl"])
    db = tmp_path / "ledger.sqlite"
    result = _invoke(["sweep", str(root), "--db", str(db)])
    assert result.exit_code == 3
    assert not db.exists(), "a default sweep must not create or write the ledger"


def test_sweep_save_records_runs(tmp_path: Path) -> None:
    root = _archive(tmp_path, ["lie_test_deletion_session.jsonl"])
    db = tmp_path / "ledger.sqlite"
    result = _invoke(["sweep", str(root), "--db", str(db), "--save"])
    assert result.exit_code == 3
    listed = _invoke(["list-runs", "--db", str(db)])
    assert "cc_5acc0000" in listed.output


def test_sweep_save_is_idempotent(tmp_path: Path) -> None:
    root = _archive(tmp_path, ["lie_test_deletion_session.jsonl"])
    db = tmp_path / "ledger.sqlite"
    first = _invoke(["sweep", str(root), "--db", str(db), "--save"])
    second = _invoke(["sweep", str(root), "--db", str(db), "--save"])
    assert first.exit_code == 3
    assert second.exit_code == 3, second.output
    assert "Traceback" not in second.output


# ---------------------------------------------------------------- limit


def test_sweep_limit_caps_scanned_files(tmp_path: Path) -> None:
    root = _archive(tmp_path, ["clean_session.jsonl", "qa_session.jsonl"])
    result = _invoke(
        ["sweep", str(root), "--db", str(tmp_path / "l.sqlite"), "--limit", "1", "--json"]
    )
    payload = json.loads(result.output)
    assert payload["scanned"] == 1


# ---------------------------------------------------------------- routing


def test_sweep_routes_codex_and_claude_shapes(tmp_path: Path) -> None:
    root = _archive(
        tmp_path, ["clean_session.jsonl", "abstain_fix_then_rerun.jsonl"]
    )
    result = _invoke(["sweep", str(root), "--db", str(tmp_path / "l.sqlite"), "--json"])
    payload = json.loads(result.output)
    assert payload["counts"]["clean"] == 2
    assert result.exit_code == 0, result.output
