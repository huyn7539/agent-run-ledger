"""`arl init --hooks` — one-command Stop-hook install (launch friction floor).

The README's Stop-hook recipe required hand-editing .claude/settings.json; the
adoption population expects setup to be automatic. Constraints under test:

- Non-destructive merge: existing settings keys and existing hooks survive.
- Rule 5 idempotency: calls 2/3 change NOTHING (same content, no rewrite).
- Fail closed: malformed settings.json -> exit 1, file untouched, no guessing.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_run_ledger.cli import HOOK_COMMAND, app


def _settings(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.json"


def _invoke(tmp_path: Path):
    return CliRunner().invoke(app, ["init", "--hooks", "--dir", str(tmp_path)])


def test_fresh_dir_creates_settings_with_stop_hook(tmp_path: Path) -> None:
    r = _invoke(tmp_path)
    assert r.exit_code == 0, r.output
    data = json.loads(_settings(tmp_path).read_text(encoding="utf-8"))
    [entry] = data["hooks"]["Stop"]
    assert {"type": "command", "command": HOOK_COMMAND} in entry["hooks"]


def test_merge_preserves_existing_settings_and_hooks(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    s.parent.mkdir(parents=True)
    s.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash(ls:*)"]},
                "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo done"}]}]},
            }
        ),
        encoding="utf-8",
    )
    r = _invoke(tmp_path)
    assert r.exit_code == 0, r.output
    data = json.loads(s.read_text(encoding="utf-8"))
    assert data["permissions"] == {"allow": ["Bash(ls:*)"]}  # untouched
    commands = [h["command"] for e in data["hooks"]["Stop"] for h in e["hooks"]]
    assert "echo done" in commands and HOOK_COMMAND in commands


def test_three_times_idempotent(tmp_path: Path) -> None:
    assert _invoke(tmp_path).exit_code == 0
    first = _settings(tmp_path).read_bytes()
    for _ in range(2):
        r = _invoke(tmp_path)
        assert r.exit_code == 0, r.output
        assert "already" in r.output.lower()
        assert _settings(tmp_path).read_bytes() == first  # byte-identical, no rewrite


def test_malformed_settings_fails_closed(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    s.parent.mkdir(parents=True)
    s.write_text("{not json", encoding="utf-8")
    r = _invoke(tmp_path)
    assert r.exit_code == 1
    assert s.read_text(encoding="utf-8") == "{not json"  # untouched
    assert "settings.json" in r.output
