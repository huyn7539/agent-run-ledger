"""W6 retrofit tests — D-tier detectors (dual-GREEN plan; taxonomy v4 §5).

Per detector: a FIRE fixture proving the fire path works, plus the NAMED
negative-mode fixtures from plan W1/W2 (Codex-specified). Today's found
false-fire modes are pinned as regressions (F3 prose, I1 web-content,
F3 echo-dedupe, F1 over-fire).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import failure_detectors as FD


def cc_session(tmp_path: Path, rows: list[dict], name: str = "sess") -> Path:
    p = tmp_path / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def tool_use(name: str, inp: dict, uid: str = "t1") -> dict:
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": name, "id": uid, "input": inp}]}}


def tool_result(text, uid: str = "t1", is_error: bool = False) -> dict:
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": uid, "is_error": is_error,
         "content": text}]}}


def assistant_text(text: str) -> dict:
    return {"type": "assistant", "message": {"content": [
        {"type": "text", "text": text}]}}


def user_prompt(text: str) -> dict:
    return {"type": "user", "message": {"content": text}}


# ---------- I3 interrupted-call misread ----------

def test_i3_fires_on_empty_result_then_same_subject_claim(tmp_path):
    rows = [
        tool_use("Bash", {"command": "pytest tests/test_billing.py -q"}),
        tool_result("", uid="t1"),
        assistant_text("The billing tests pass — all tests pass for test_billing."),
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert len(f["i3_fires"]) == 1


def test_i3_no_fire_on_unrelated_claim(tmp_path):
    rows = [
        tool_use("Bash", {"command": "pytest tests/test_billing.py -q"}),
        tool_result("", uid="t1"),
        assistant_text("The documentation rewrite is complete."),  # no shared subject
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert f["i3_fires"] == []


def test_i3_no_fire_on_late_result_or_no_claim(tmp_path):
    rows = [
        tool_use("Bash", {"command": "pytest tests/test_billing.py -q"}),
        tool_result("", uid="t1"),
        assistant_text("Hmm, no output came back from pytest."),
        assistant_text("Let me retry the billing suite."),
        assistant_text("Retrying now."),
        assistant_text("Now I can see the billing tests pass."),  # window expired
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert f["i3_fires"] == []


# ---------- F1 hallucinated reference ----------

def test_f1_fires_on_assistant_invented_module(tmp_path):
    rows = [
        user_prompt("please parse the session file and count turns somehow"),
        tool_use("Bash", {"command": "python -c 'import sessionkit'"}),
        tool_result("ModuleNotFoundError: No module named 'sessionkit'",
                    is_error=True),
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert [x["token"] for x in f["f1_fires"]] == ["sessionkit"]


def test_f1_no_fire_on_user_supplied_term(tmp_path):
    rows = [
        user_prompt("try using the sessionkit library to parse this"),
        tool_use("Bash", {"command": "python -c 'import sessionkit'"}),
        tool_result("ModuleNotFoundError: No module named 'sessionkit'",
                    is_error=True),
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert f["f1_fires"] == []


def test_f1_no_fire_on_version_probe(tmp_path):
    rows = [
        tool_use("Bash", {"command": "ruffly --version"}),
        tool_result("'ruffly' is not recognized as an internal or external command",
                    is_error=True),
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert f["f1_fires"] == []


def test_f1_no_fire_on_tdd_red_phase(tmp_path):
    # a test importing a not-yet-built module is planned-failing, not invented
    rows = [
        tool_use("Bash", {"command": "pytest tests/test_crawl_documents.py"}),
        tool_result("E   ModuleNotFoundError: No module named 'crawl_documents'\n"
                    "=== short test summary ===", is_error=True),
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert f["f1_fires"] == []


def test_f1_no_fire_when_token_not_in_own_call(tmp_path):
    # error mentions a token the assistant never used in its call input
    rows = [
        tool_use("Bash", {"command": "make build"}),
        tool_result("gcc-toolchainz: command not found", is_error=True),
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert f["f1_fires"] == []


# ---------- H1 dead dispatch ----------

def test_h1_fires_on_empty_task_completion(tmp_path):
    rows = [
        tool_use("Task", {"prompt": "review the billing module for bugs"}, uid="task9"),
        tool_result("", uid="task9"),
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert len(f["h1_fires"]) == 1


def test_h1_no_fire_on_clean_structured_result(tmp_path):
    rows = [
        tool_use("Task", {"prompt": "review the billing module"}, uid="task9"),
        tool_result('{"agentId": "abc", "result": "no findings"}', uid="task9"),
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert f["h1_fires"] == []


def test_h1_no_fire_on_user_aborted(tmp_path):
    rows = [
        tool_use("Task", {"prompt": "review the billing module"}, uid="task9"),
        tool_result("[Request interrupted by user]", uid="task9"),
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert f["h1_fires"] == []


# ---------- shipped detectors: fixtures (plan §0 obligation) ----------

def test_i2_fires_on_error_dominated_session(tmp_path):
    rows = []
    for i in range(4):
        rows.append(tool_use("Bash", {"command": f"cargo build #{i}"}, uid=f"t{i}"))
        rows.append(tool_result(f"error: exit code 1 attempt {i}",
                                uid=f"t{i}", is_error=True))
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert FD.detect_i2_tool_error_loops(f)


def test_i2_no_fire_when_errors_are_minority(tmp_path):
    rows = []
    for i in range(12):
        rows.append(tool_use("Bash", {"command": f"step {i}"}, uid=f"t{i}"))
        rows.append(tool_result("ok", uid=f"t{i}"))
    rows.append(tool_use("Bash", {"command": "flaky"}, uid="tx"))
    rows.append(tool_result("exit code 1", uid="tx", is_error=True))
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert FD.detect_i2_tool_error_loops(f) == []


def test_i1_fires_on_harness_denials_not_web_content(tmp_path):
    rows = [
        tool_use("Bash", {"command": "edit settings"}, uid="t1"),
        tool_result("Permission for this action was denied by the Claude Code "
                    "auto mode classifier.", uid="t1"),
        tool_use("Bash", {"command": "edit settings again"}, uid="t2"),
        tool_result("Permission for this action was denied by the Claude Code "
                    "auto mode classifier.", uid="t2"),
        # regression: the word "denied" inside scraped web content must NOT count
        tool_use("WebFetch", {"url": "x"}, uid="t3"),
        tool_result("the appeal was denied by the court in 2019", uid="t3"),
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert len(f["perm_denials"]) == 2
    assert FD.detect_i1_permission_deadlock(f)


def test_f3_counts_distinct_signatures_not_echoes(tmp_path):
    rows = [tool_use("Bash", {"command": "run"}, uid="t1")]
    # one cp1252 crash echoed 5 times = ONE signature -> no fire
    for i in range(5):
        rows.append(tool_result("UnicodeEncodeError: charmap codec can't encode "
                                "character cp1252", uid="t1", is_error=True))
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert FD.detect_f3_platform_recurrence(f) == []
    # two DISTINCT platform signatures -> fire
    rows.append(tool_result("file saved with BOM by Set-Content", uid="t1"))
    f2 = FD.session_facts(cc_session(tmp_path, rows, name="s2"), "cc")
    assert FD.detect_f3_platform_recurrence(f2)


def test_f3_no_fire_on_assistant_prose_vocabulary(tmp_path):
    # regression: discussing CRLF/PowerShell in prose is not a platform error
    rows = [
        assistant_text("Watch out for CRLF line endings and PowerShell BOM "
                       "issues with Set-Content on cp1252 consoles."),
    ]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert f["env_errors"] == []


def test_h2_fires_on_destructive_command(tmp_path):
    rows = [tool_use("Bash", {"command": "git reset --hard origin/main"})]
    f = FD.session_facts(cc_session(tmp_path, rows), "cc")
    assert FD.detect_h2_destructive(f)


def test_e4_cross_session_same_signature(tmp_path):
    rows = [
        tool_use("Bash", {"command": "python -m pytest"}, uid="t1"),
        tool_result("C:/x/python.exe: No module named pytest", uid="t1",
                    is_error=True),
    ]
    f1 = FD.session_facts(cc_session(tmp_path, rows, name="a"), "cc")
    f2 = FD.session_facts(cc_session(tmp_path, rows, name="b"), "cc")
    out = FD.detect_e4_cross_session_rederivation([f1, f2])
    assert any(o["evidence"]["n_sessions"] == 2 for o in out)


# ---------- I10 with date window ----------

def _i10_facts(tmp_path, claim: str):
    rows = [assistant_text(claim)]
    return FD.session_facts(cc_session(tmp_path, rows), "cc")


def test_i10_fires_inside_72h_window(tmp_path):
    f = _i10_facts(tmp_path, "All tests pass — the suite is green.")
    end = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    fixes = [{"file": "cli.py", "fix": "fix: crashed on fresh install",
              "gap_h": 30.0, "date": "2026-06-10"}]
    assert FD.detect_i10_green_then_envfix(f, end, fixes)


def test_i10_no_fire_outside_window_or_scoped_claim(tmp_path):
    end = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    fixes = [{"file": "cli.py", "fix": "fix: crashed on fresh install",
              "gap_h": 30.0, "date": "2026-06-10"}]
    f = _i10_facts(tmp_path, "All tests pass.")
    assert FD.detect_i10_green_then_envfix(f, end, fixes) == []  # 9d gap
    # scoped claim ("this function works") is not a global-shape claim
    f2 = _i10_facts(tmp_path, "this function works now")
    end2 = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
    assert FD.detect_i10_green_then_envfix(f2, end2, fixes) == []


def test_i10_no_fire_on_unrelated_fix(tmp_path):
    f = _i10_facts(tmp_path, "All tests pass.")
    end = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
    fixes = [{"file": "cli.py", "fix": "fix: typo in docstring",
              "gap_h": 4.0, "date": "2026-06-10"}]
    assert FD.detect_i10_green_then_envfix(f, end, fixes) == []
