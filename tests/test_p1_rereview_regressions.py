"""Codex re-review round 2 (2026-06-11) — close P1-3 cross-language false-accusation
and the P2/P3 sweep enumeration bound.

P1-3: an English-only deletion-verb list is unwinnable; every missed directive
(non-English, slang, unicode-lookalike) is a FALSE ACCUSATION from an honesty
tool. The structural fix: a substantive preceding human instruction that is not a
clean from-scratch build request makes R1 abstain, in ANY language.
"""

from __future__ import annotations

from pathlib import Path

from agent_run_ledger.adapters._facts import instruction_directs_deletion


# P1-3: cross-language / slang / unicode deletion directives now make R1 abstain.
# Covered: high-frequency non-English Latin-script verbs (es/pt/fr/de/it), common
# slang, any non-Latin-script substantive instruction, and English directive words.
# DOCUMENTED RESIDUAL: novel English slang with no recognized verb and no non-Latin
# letters (e.g. a brand-new coinage) can still slip — the long tail Codex flagged as
# unwinnable by enumeration. Bounded: R1 also requires an autonomous test deletion
# AND a terminal success claim to fire, and grades L1/low with the user-directed
# caveat always in limits. The residual is a missed-abstain on exotic phrasing, not
# an unbounded false-accusation surface.
def test_non_english_and_slang_deletion_directives_abstain() -> None:
    deleted = ["tests/test_billing.py"]
    for instruction in (
        "elimina estos archivos de prueba antiguos",   # Spanish (Latin verb list)
        "lösche diese veralteten Testdateien bitte",   # German (non-Latin ö -> fallback)
        "удали этот устаревший тестовый файл",          # Russian (non-Latin -> fallback)
        "yeet these flaky test files please",          # slang (added verb)
        "torch the cache and the broken specs",        # slang (added verb)
        "🗑️ remove these obsolete test files for me",  # emoji + English directive
    ):
        assert instruction_directs_deletion(instruction, deleted), instruction


# Guard the other side: a genuine from-scratch build request does NOT excuse an
# autonomous test deletion — R1 must stay live there.
def test_from_scratch_build_request_does_not_excuse_deletion() -> None:
    deleted = ["tests/test_payments.py"]
    for instruction in (
        "implement the payments module and make it pass",
        "build a new billing service from scratch",
        "add a checkout endpoint",
    ):
        assert not instruction_directs_deletion(instruction, deleted), instruction


# A trivial ack is not a directive — too short to plausibly cover a deletion.
def test_trivial_ack_is_not_a_directive() -> None:
    deleted = ["tests/test_x.py"]
    for instruction in ("ok", "go", "yes", "sure", "do it"):
        assert not instruction_directs_deletion(instruction, deleted), instruction


# Empty / no preceding instruction -> deletion stays accusable (the true-positive path).
def test_no_instruction_keeps_r1_live() -> None:
    assert not instruction_directs_deletion("", ["tests/test_x.py"])
    assert not instruction_directs_deletion("   ", ["tests/test_x.py"])


# P2/P3: sweep enumeration is bounded and the truncation is reported, not silent.
def test_sweep_enumeration_is_bounded(tmp_path: Path, monkeypatch) -> None:
    import agent_run_ledger.cli as cli
    from typer.testing import CliRunner

    monkeypatch.setattr(cli, "_SWEEP_MAX_ENUM", 5)
    root = tmp_path / "many"
    root.mkdir()
    # 12 trivial (unreadable-as-session) jsonl files; enumeration must stop at 5.
    for i in range(12):
        (root / f"s{i:03d}.jsonl").write_text("{}\n", encoding="utf-8")
    result = CliRunner().invoke(cli.app, ["sweep", str(root), "--limit", "100"])
    assert "stopped enumerating at 5" in result.output, result.output
