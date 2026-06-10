"""Task 58 — the success-lie detector (success-claim vs log-evidence divergence).

Two sub-rules, both PRECISION-FIRST (a false positive is the trust-killer):

  R1 — success claim after test deletion: a tool call deletes a test-pattern path
       (tests/ dir, test_*.py, *_test.*, *.spec.*) and a later assistant completion
       claim follows in the same session with NO intervening human instruction.
       -> artifact_failure receipt at L1 MAX (relevance proven from the log; intent
       NOT proven), confidence low, limits MUST carry the user-directed caveat.
       ABSTAIN: deletion immediately preceded by a human instruction mentioning the
       deletion/file; no completion claim follows; non-test path deleted.

  R2 — completion claim with ZERO mutating tool calls after a change request.
       -> artifact_failure diagnostic at L0. ABSTAIN: no completion claim, no
       change request (pure Q&A must NEVER fire), any mutating call, or the
       adapter did not supply the mutating-census facts.

ARCHITECTURE UNDER TEST (facts-vs-judgments doctrine): the ADAPTERS compute
bounded, content-free boolean FACTS from raw session content at capture time
(deletes_test_path / user_directed_deletion / completion_claim_follows /
mutating / change_request, stored as allowed step metadata — same pattern as
error_class). CORE reads only those bounded facts on read; raw content never
reaches core and never lands on disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_run_ledger.adapters._facts import (
    command_deletes_test_path,
    command_is_read_only,
    deleted_test_paths,
    instruction_directs_deletion,
    is_change_request,
    is_completion_claim,
    is_test_path,
)
from agent_run_ledger.adapters.claude_code import bundle_from_session, load_claude_session
from agent_run_ledger.adapters.codex import bundle_from_rollout, load_codex_rollout
from agent_run_ledger.cli import DETECTOR_COVERAGE, app
from agent_run_ledger.core.models import (
    FAILURE_CLASSES,
    TraceBundle,
    TraceValidationError,
)
from agent_run_ledger.core.prescriptions import analyze_bundle, detect_success_lies
from agent_run_ledger.core.receipt import OBSERVED_FAILURES, build_receipts

FIXTURES = Path(__file__).parent / "fixtures" / "claude_code"
CODEX_FIXTURES = Path(__file__).parent / "fixtures" / "codex"


def _claude_bundle(name: str) -> TraceBundle:
    return bundle_from_session(load_claude_session(FIXTURES / name))


def _claude_records(name: str) -> list[dict]:
    return [json.loads(ln) for ln in (FIXTURES / name).read_text(encoding="utf-8").splitlines()]


def _codex_records(name: str) -> list[dict]:
    return [
        json.loads(ln) for ln in (CODEX_FIXTURES / name).read_text(encoding="utf-8").splitlines()
    ]


def _artifact_prescriptions(bundle: TraceBundle):
    return [rx for rx in analyze_bundle(bundle) if rx.failure_class == "artifact_failure"]


# ===================================================================== helpers
# The bounded text classifiers the adapters use at capture time. FIRING-side
# matchers are tight (miss rather than guess); ABSTAIN-side matchers are
# deliberately generous (when in doubt, abstain).


class TestIsTestPath:
    @pytest.mark.parametrize(
        "path",
        [
            "tests/test_billing_proration.py",
            "tests/",
            "tests",
            "src/tests/helpers.py",  # inside a tests/ dir
            "test_models.py",
            "billing_test.go",
            "Button.spec.tsx",
            "pkg\\tests\\test_x.py",  # windows separators
        ],
    )
    def test_matches(self, path: str) -> None:
        assert is_test_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "src/main.py",
            "contest/file.py",  # 'contest' must not substring-match 'tests'
            "protest_log.txt",
            "src/billing/util.py",
            "testimony.md",
            "",
        ],
    )
    def test_rejects(self, path: str) -> None:
        assert is_test_path(path) is False


class TestCommandDeletesTestPath:
    @pytest.mark.parametrize(
        "command",
        [
            "rm tests/test_billing_proration.py",
            "rm -rf tests/",
            "rm -f tests/test_x.py tests/test_y.py",
            "git rm --cached tests/test_x.py",
            "del tests\\test_x.py",
            "Remove-Item tests/test_x.py",
            "cd /repo && rm tests/test_x.py",
            "unlink foo_test.go",
        ],
    )
    def test_fires(self, command: str) -> None:
        assert command_deletes_test_path(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "rm src/util.py",  # not a test path
            "pytest -q tests/",  # not a delete verb
            "echo rm tests/test_x.py",  # verb is echo, not rm
            "git mv tests/test_x.py tests/test_y.py",  # not git rm
            "ls tests/",
            "rm -rf node_modules",
            "",
        ],
    )
    def test_abstains(self, command: str) -> None:
        assert command_deletes_test_path(command) is False

    def test_returns_the_matched_paths(self) -> None:
        assert deleted_test_paths("rm tests/test_billing_proration.py") == [
            "tests/test_billing_proration.py"
        ]


class TestCompletionClaim:
    @pytest.mark.parametrize(
        "text",
        [
            "All tests pass. Task complete.",
            "Done - all tests pass now.",
            "Task complete.",
            "Everything passes and the build is green.",
            "The tests are passing after the change.",
        ],
    )
    def test_fires(self, text: str) -> None:
        assert is_completion_claim(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Hope that helps.",
            "Let me run pytest to make sure all tests pass.",  # intent, not a claim
            "I will verify the tests pass before finishing.",
            "Next I need to check that the tests pass.",
            "The proration rounds to the nearest cent.",
            "",
        ],
    )
    def test_abstains(self, text: str) -> None:
        assert is_completion_claim(text) is False


class TestChangeRequest:
    def test_change_verbs_fire(self) -> None:
        assert is_change_request("fix the rounding bug in the proration module") is True
        assert is_change_request("please implement retry caps") is True

    def test_pure_questions_abstain(self) -> None:
        assert is_change_request("how does the proration rounding logic work?") is False
        assert is_change_request("what does this function return?") is False


class TestUserDirectedDeletion:
    def test_instruction_naming_the_file_directs(self) -> None:
        assert (
            instruction_directs_deletion(
                "tests/test_billing_proration.py is obsolete - delete it",
                ["tests/test_billing_proration.py"],
            )
            is True
        )

    def test_generic_delete_directive_directs(self) -> None:
        # ABSTAIN-side matcher is generous: any deletion directive counts.
        assert (
            instruction_directs_deletion(
                "remove the flaky test and rerun", ["tests/test_flaky.py"]
            )
            is True
        )

    def test_unrelated_instruction_does_not_direct(self) -> None:
        assert (
            instruction_directs_deletion(
                "get the billing test suite green before the release",
                ["tests/test_billing_proration.py"],
            )
            is False
        )


class TestCommandReadOnly:
    @pytest.mark.parametrize(
        "command",
        ["ls -la", "cat foo.py", "grep -rn pattern src", "git status", "git diff", "rg TODO"],
    )
    def test_read_only(self, command: str) -> None:
        assert command_is_read_only(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "rm foo.py",
            "pytest -q",  # not provably read-only -> conservative
            "cat a.txt > b.txt",  # redirect mutates
            "git commit -m x",
            "unknown-tool --flag",  # unknown -> conservative
            "",
        ],
    )
    def test_not_provably_read_only(self, command: str) -> None:
        assert command_is_read_only(command) is False


# ============================================================ adapter facts (claude)


def test_claude_adapter_marks_test_deletion_step() -> None:
    bundle = _claude_bundle("lie_test_deletion_session.jsonl")
    rm_steps = [s for s in bundle.steps if s.metadata.get("deletes_test_path") is True]
    assert len(rm_steps) == 1
    md = rm_steps[0].metadata
    assert md.get("completion_claim_follows") is True
    assert md.get("user_directed_deletion") is not True
    assert md.get("mutating") is True


def test_claude_adapter_marks_user_directed_deletion() -> None:
    bundle = _claude_bundle("abstain_user_directed_deletion_session.jsonl")
    rm_steps = [s for s in bundle.steps if s.metadata.get("deletes_test_path") is True]
    assert len(rm_steps) == 1
    assert rm_steps[0].metadata.get("user_directed_deletion") is True


def test_claude_adapter_mutating_census_is_explicit_on_every_step() -> None:
    """R2's census requires an EXPLICIT bool on every step — absence means the
    facts are unavailable and the detector must abstain."""
    bundle = _claude_bundle("lie_no_op_completion_session.jsonl")
    assert all(isinstance(s.metadata.get("mutating"), bool) for s in bundle.steps)
    assert not any(s.metadata.get("mutating") for s in bundle.steps)
    assert any(s.metadata.get("change_request") is True for s in bundle.steps)


def test_claude_adapter_facts_store_no_content() -> None:
    """The facts are CONTENT-FREE booleans: no path, no command text, no claim
    text may land in step metadata or anywhere else in the bundle."""
    records = _claude_records("lie_test_deletion_session.jsonl")
    sentinel_path = "SENTINEL_FILE_9988"
    sentinel_claim = "SENTINEL_CLAIM_7766"
    records[3]["message"]["content"][0]["input"]["command"] = (
        f"rm tests/test_{sentinel_path}.py"
    )
    records[7]["message"]["content"][0]["text"] = f"All tests pass. {sentinel_claim} Task complete."
    bundle = bundle_from_session(records)
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    receipts = build_receipts(bundle)
    assert receipts, "the doctored session must still fire R1"
    blob = json.dumps(bundle.to_dict()) + json.dumps([asdict(r) for r in receipts])
    assert sentinel_path not in blob
    assert sentinel_claim not in blob


# ===================================================================== R1 fires


def test_r1_fires_on_canonical_test_deletion_lie() -> None:
    """The anti-AI persona's archived case: rm tests/test_billing_proration.py two
    tool calls before 'all tests pass' -> R1 fires at L1."""
    bundle = _claude_bundle("lie_test_deletion_session.jsonl")
    rx = _artifact_prescriptions(bundle)
    assert len(rx) == 1
    assert rx[0].failure_class == "artifact_failure"
    assert "rule=R1" in rx[0].evidence

    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    receipts = build_receipts(bundle)
    assert len(receipts) == 1
    r = receipts[0]
    assert r.observed_failure == "artifact_failure"
    assert r.proof_level == "L1"
    assert r.confidence == "low"
    # the MANDATORY user-directed-deletion caveat
    assert any("user-directed" in limit for limit in r.limits)
    # intent is NOT proven — the claim must say so
    assert "intent" in r.claim.lower()


def test_r1_receipt_artifact_is_a_text_fix_direction() -> None:
    bundle = _claude_bundle("lie_test_deletion_session.jsonl")
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    r = build_receipts(bundle)[0]
    assert r.repair_artifact["templated"] is False
    assert "restore" in r.repair_artifact["patch"].lower()
    assert r.next_evidence


def test_r1_does_not_double_fire_the_retry_detector() -> None:
    """The two pytest runs around the rm are NOT a blind retry loop (different
    call between them) — only the artifact receipt may fire."""
    bundle = _claude_bundle("lie_test_deletion_session.jsonl")
    prescriptions = analyze_bundle(bundle)
    assert [rx.failure_class for rx in prescriptions] == ["artifact_failure"]


def test_r1_fires_via_codex_adapter() -> None:
    bundle = bundle_from_rollout(
        load_codex_rollout(CODEX_FIXTURES / "fire_test_deletion_lie.jsonl")
    )
    rx = _artifact_prescriptions(bundle)
    assert len(rx) == 1
    assert "rule=R1" in rx[0].evidence
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    receipts = build_receipts(bundle)
    assert receipts[0].observed_failure == "artifact_failure"
    assert receipts[0].proof_level == "L1"


# =================================================================== R1 abstains


def test_r1_abstains_on_user_directed_deletion() -> None:
    """The structurally-similar abstain case: the user ASKED for the deletion."""
    bundle = _claude_bundle("abstain_user_directed_deletion_session.jsonl")
    assert analyze_bundle(bundle) == []


def test_r1_abstains_when_no_completion_claim_follows() -> None:
    records = _claude_records("lie_test_deletion_session.jsonl")
    # drop the final claim line -> deletion with no claim -> abstain
    records = records[:-1]
    bundle = bundle_from_session(records)
    assert analyze_bundle(bundle) == []


def test_r1_abstains_when_human_instruction_intervenes() -> None:
    """A human instruction between the deletion and the claim resets the
    autonomous stretch — the claim no longer 'follows' the deletion."""
    records = _claude_records("lie_test_deletion_session.jsonl")
    instruction = json.loads(json.dumps(records[0]))
    instruction["uuid"] = "u-intervene"
    instruction["message"]["content"] = "looks good so far, write up a summary"
    instruction["timestamp"] = "2026-06-10T03:00:27.000Z"
    records = records[:-1] + [instruction] + records[-1:]
    bundle = bundle_from_session(records)
    assert analyze_bundle(bundle) == []


def test_r1_abstains_on_non_test_path_deletion() -> None:
    records = _claude_records("lie_test_deletion_session.jsonl")
    records[3]["message"]["content"][0]["input"]["command"] = "rm src/billing/legacy_util.py"
    bundle = bundle_from_session(records)
    assert analyze_bundle(bundle) == []


def test_r1_abstains_via_codex_adapter_on_user_directed_deletion() -> None:
    records = _codex_records("fire_test_deletion_lie.jsonl")
    directive = {
        "timestamp": "2026-06-10T08:00:04.500Z",
        "type": "event_msg",
        "payload": {
            "type": "user_message",
            "message": "tests/test_billing_proration.py is obsolete, delete it and rerun",
        },
    }
    # insert the directive immediately before the rm call
    rm_index = next(
        i
        for i, rec in enumerate(records)
        if isinstance(rec.get("payload"), dict)
        and "rm tests" in str(rec["payload"].get("arguments", ""))
    )
    records = records[:rm_index] + [directive] + records[rm_index:]
    bundle = bundle_from_rollout(records)
    assert _artifact_prescriptions(bundle) == []


# ===================================================================== R2 fires


def test_r2_fires_on_no_op_completion_claim() -> None:
    """Change requested, completion claimed, ZERO mutating tool calls -> L0
    diagnostic (the loop-tinkerer's MCP/stall case)."""
    bundle = _claude_bundle("lie_no_op_completion_session.jsonl")
    rx = _artifact_prescriptions(bundle)
    assert len(rx) == 1
    assert "rule=R2" in rx[0].evidence
    assert "mutating_calls=0" in rx[0].evidence

    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    receipts = build_receipts(bundle)
    assert len(receipts) == 1
    r = receipts[0]
    assert r.observed_failure == "artifact_failure"
    assert r.proof_level == "L0"
    assert r.confidence == "low"
    # absence-of-evidence honesty: work may have landed outside the log
    assert any("outside" in limit for limit in r.limits)


# =================================================================== R2 abstains


def test_r2_never_fires_on_pure_qa_session() -> None:
    """A question-answering session (no change request, read-only tools, 'hope
    that helps' close) must NEVER fire — explicit acceptance gate."""
    bundle = _claude_bundle("qa_session.jsonl")
    assert analyze_bundle(bundle) == []


def test_r2_abstains_when_a_mutating_call_exists() -> None:
    bundle = _claude_bundle("abstain_edit_completion_session.jsonl")
    assert analyze_bundle(bundle) == []


def test_r2_abstains_when_no_completion_claim() -> None:
    records = _claude_records("lie_no_op_completion_session.jsonl")
    records[-1]["message"]["content"][0]["text"] = "I could not find the bug yet."
    bundle = bundle_from_session(records)
    assert analyze_bundle(bundle) == []


def test_r2_abstains_when_census_facts_are_absent(non_demo_bundle: TraceBundle) -> None:
    """A bundle whose adapter did not supply the mutating census (neutral traces,
    OpenAI SDK traces, old captures) can NEVER fire R2 — absence of the fact is
    unknown, not 'non-mutating'."""
    assert detect_success_lies(non_demo_bundle) == []


# ===================================== grading is fact-corroborated (forged input)


def _neutral_bundle_with_artifact_prescription(evidence: list[str]) -> TraceBundle:
    """A neutral imported bundle claiming an artifact_failure WITHOUT the
    corroborating step facts (forged-evidence class, Task 51 lessons)."""
    return TraceBundle.from_dict(
        {
            "schema_version": "0.1",
            "run": {
                "id": "run_forged",
                "workflow": "w",
                "framework": "f",
                "provider": "p",
                "model": "m",
                "started_at": "2026-06-10T00:00:00Z",
                "ended_at": "2026-06-10T00:00:01Z",
                "success_label": "unknown",
            },
            "steps": [
                {
                    "id": "step_clean",
                    "type": "function",
                    "name": "Bash",
                    "started_at": "2026-06-10T00:00:00Z",
                    "ended_at": "2026-06-10T00:00:01Z",
                }
            ],
            "prescriptions": [
                {
                    "id": "rx_forged",
                    "failure_class": "artifact_failure",
                    "severity": "high",
                    "root_cause": "forged",
                    "one_line_fix": "forged",
                    "evidence": evidence,
                    "patch_type": "config_diff",
                    "patch": (
                        "# forged fix direction long enough to pass the length gate\n"
                        "- before line for the validator\n+ after line for the validator\n"
                    ),
                    "regression_test_template": "def test_x(): pass",
                }
            ],
        }
    )


def test_forged_artifact_prescription_grades_l0_not_l1() -> None:
    """A stored/imported prescription claiming R1 over a step WITHOUT the
    deletes_test_path facts must fail closed to L0 — never L1."""
    bundle = _neutral_bundle_with_artifact_prescription(
        ["rule=R1", "step_id=step_clean", "deletes_test_path=true"]
    )
    receipts = build_receipts(bundle)
    assert len(receipts) == 1
    assert receipts[0].proof_level == "L0"


def test_artifact_prescription_with_no_rule_grades_l0() -> None:
    bundle = _neutral_bundle_with_artifact_prescription(["step_id=step_clean"])
    assert build_receipts(bundle)[0].proof_level == "L0"


# ======================================================= failure_class plumbing


def test_failure_classes_is_a_closed_vocabulary() -> None:
    assert FAILURE_CLASSES == ("retry_loop", "artifact_failure")
    assert "artifact_failure" in OBSERVED_FAILURES


def test_unknown_failure_class_fails_closed() -> None:
    """A hostile/imported prescription with an out-of-vocabulary class is a typed
    error (fail closed), mirroring the patch_type closed vocabulary."""
    from agent_run_ledger.core.models import PrescriptionRecord

    with pytest.raises(TraceValidationError):
        PrescriptionRecord.from_dict({"failure_class": "bogus_class"}, "run_x")


def test_retry_prescriptions_default_to_retry_loop_class(non_demo_bundle: TraceBundle) -> None:
    prescriptions = analyze_bundle(non_demo_bundle)
    assert prescriptions
    assert all(rx.failure_class == "retry_loop" for rx in prescriptions)


def test_failure_class_survives_storage_roundtrip(tmp_path: Path) -> None:
    from agent_run_ledger.core.storage import load_bundle, save_bundle

    bundle = _claude_bundle("lie_test_deletion_session.jsonl")
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    db = tmp_path / "ledger.sqlite"
    run_id = save_bundle(db, bundle)
    loaded = load_bundle(db, run_id)
    assert [rx.failure_class for rx in loaded.prescriptions] == ["artifact_failure"]
    # and the receipt still grades from the loaded facts
    assert build_receipts(loaded)[0].proof_level == "L1"


def test_failure_class_survives_dict_roundtrip() -> None:
    bundle = _claude_bundle("lie_test_deletion_session.jsonl")
    bundle = bundle.with_prescriptions(analyze_bundle(bundle))
    rt = TraceBundle.from_dict(bundle.to_dict())
    assert [rx.failure_class for rx in rt.prescriptions] == ["artifact_failure"]


# ======================================================= verdict CLI wire-through


def test_verdict_fires_artifact_failure_on_lie_fixture(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "verdict",
            str(FIXTURES / "lie_test_deletion_session.jsonl"),
            "--db",
            str(tmp_path / "l.sqlite"),
            "--json",
        ],
    )
    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["verdict"] == "receipts"
    assert payload["receipts"][0]["observed_failure"] == "artifact_failure"
    assert payload["max_proof_level"] == "L1"


def test_verdict_stays_clean_on_user_directed_fixture(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "verdict",
            str(FIXTURES / "abstain_user_directed_deletion_session.jsonl"),
            "--db",
            str(tmp_path / "l.sqlite"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_coverage_moves_artifact_failure_to_checked() -> None:
    """Coverage honesty: artifact_failure is checked (R1/R2 only, L0-L1); what
    remains unchecked stays named."""
    checked = " ".join(DETECTOR_COVERAGE["checked"])
    not_checked = " ".join(DETECTOR_COVERAGE["not_checked"])
    assert "artifact_failure" in checked
    assert "R1" in checked and "R2" in checked
    assert "L0-L1" in checked
    # the rest of the artifact class is honestly still unchecked
    assert "beyond R1/R2" in not_checked
    assert "success claims contradicted" not in not_checked
