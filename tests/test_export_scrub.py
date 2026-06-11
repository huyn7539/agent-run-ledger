"""Task 46 — allowed-key metadata VALUES are scrubbed at the EXPORT boundary.

The metadata sanitizer is a key-name allowlist that keeps VALUES verbatim under
allowed keys (ADR-001 Category 2 — local diffs/diagnostics depend on the exact
text). That is correct LOCALLY and was a leak at the SHARE boundary: the README
tells users to paste `arl export` output, and a sensitive value stuffed under
`before` / `path` / `after` (or embedded in a prescription's unified-diff patch,
which is BUILT from those values) traveled verbatim.

Contract under test (fail-closed, Rule 6):
  * DEFAULT export drops every `_RAW_CONTENT_METADATA_KEYS` value from step
    metadata (key NAMES are disclosed under `_scrubbed_keys` — content-free)
    and replaces a non-empty prescription `patch` with a content-free marker.
  * `raw_local=True` (CLI `--raw-local`) is the explicit LOCAL opt-in that
    keeps the old verbatim behavior with its disclosure note.
  * A scrubbed export re-imports cleanly (graceful absence, no validation error).
  * Local capture/render are untouched — this is strictly an egress projection.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_run_ledger.cli import app
from agent_run_ledger.core.io import load_trace, write_trace
from agent_run_ledger.core.models import (
    _RAW_CONTENT_METADATA_KEYS,
    PrescriptionRecord,
    RunRecord,
    StepRecord,
    TraceBundle,
)
from agent_run_ledger.core.storage import save_bundle

SENTINEL = "SECRET-TASK46-SENTINEL"


def _bundle_with_raw_content() -> TraceBundle:
    """Sentinels under EVERY raw-content carrier the share form must scrub:
    allowed-key step metadata, the patch, run.outcome_json, and every free-text
    prescription field (Codex Rule 8 re-review F-02)."""
    run = RunRecord(
        id="run_t46",
        workflow="w",
        framework="f",
        provider="openai",
        model="gpt-4o-mini",
        started_at="2026-06-11T00:00:00Z",
        ended_at="2026-06-11T00:00:01Z",
        success_label="failed",
        outcome_json=f'{{"note": "{SENTINEL}-outcome"}}',
    )
    metadata = {key: f"{SENTINEL}-{key}" for key in sorted(_RAW_CONTENT_METADATA_KEYS)}
    metadata["model"] = "gpt-4o-mini"  # bounded-fact key: must SURVIVE export
    step = StepRecord(
        id="s1",
        run_id=run.id,
        step_type="function",
        name="crm.lookup",
        started_at="2026-06-11T00:00:00Z",
        ended_at="2026-06-11T00:00:01Z",
        metadata=metadata,
    )
    rx = PrescriptionRecord(
        id="rx1",
        run_id=run.id,
        severity="high",
        root_cause=f"retry loop seen near {SENTINEL}-root",
        one_line_fix=f"Set crm.lookup retry budget ({SENTINEL}-fix).",
        evidence=[
            "step_id=s1",
            "retry_count=2 additional attempts",
            f"free-text note: {SENTINEL}-evidence",
        ],
        patch_type="unified_diff",
        patch=(
            "--- a/secrets/config.py\n+++ b/secrets/config.py\n@@ -1 +1 @@\n"
            f"-API_KEY = '{SENTINEL}'\n+API_KEY = 'rotated'\n"
        ),
        expected_impact={"note": f"{SENTINEL}-impact"},
        regression_test_template=f"def test_x():\n    assert '{SENTINEL}-reg'\n",
    )
    return TraceBundle(run=run, steps=[step], prescriptions=[rx])


def test_default_export_carries_no_raw_content_values(tmp_path: Path) -> None:
    out = tmp_path / "share.json"
    write_trace(_bundle_with_raw_content(), out)
    text = out.read_text(encoding="utf-8")
    assert SENTINEL not in text, "raw-content value survived into the share-form export"
    data = json.loads(text)
    step = data["steps"][0]
    # bounded-fact keys survive; raw-content keys are gone, NAMES disclosed
    assert step["metadata"]["model"] == "gpt-4o-mini"
    assert set(step["metadata"].get("_scrubbed_keys", [])) == set(_RAW_CONTENT_METADATA_KEYS)
    # the patch (built FROM before/path/after values) is replaced by the marker
    rx = data["prescriptions"][0]
    assert "scrubbed at export" in rx["patch"]
    # F-02: free-text prescription fields + run outcome are scrubbed too; the
    # closed-grammar evidence lines receipts grade from SURVIVE (reproducible),
    # the free-text line is dropped and counted.
    assert data["run"].get("outcome_json") in (None, "")
    assert "step_id=s1" in rx["evidence"]
    assert "retry_count=2 additional attempts" in rx["evidence"]
    assert "[1 evidence line(s) scrubbed at export]" in rx["evidence"]


def test_raw_local_export_keeps_values_with_disclosure(tmp_path: Path) -> None:
    out = tmp_path / "local.json"
    write_trace(_bundle_with_raw_content(), out, raw_local=True)
    text = out.read_text(encoding="utf-8")
    assert SENTINEL in text  # explicit LOCAL opt-in keeps full fidelity
    assert "raw local export" in text  # the disclosure note rides along


def test_scrubbed_export_reimports_cleanly(tmp_path: Path) -> None:
    out = tmp_path / "share.json"
    write_trace(_bundle_with_raw_content(), out)
    loaded = load_trace(out)
    assert loaded.run.id == "run_t46"
    assert loaded.steps[0].metadata.get("model") == "gpt-4o-mini"
    assert all(k not in loaded.steps[0].metadata for k in _RAW_CONTENT_METADATA_KEYS)


def test_export_never_mutates_the_in_memory_bundle(tmp_path: Path) -> None:
    """REGRESSION PIN (caught during Task 46 implementation): to_dict shares the
    live step.metadata dict — an in-place scrub mutated the in-memory bundle.
    Exporting must leave the bundle untouched."""
    bundle = _bundle_with_raw_content()
    write_trace(bundle, tmp_path / "share.json")
    md = bundle.steps[0].metadata
    assert all(k in md for k in _RAW_CONTENT_METADATA_KEYS)
    assert "_scrubbed_keys" not in md
    assert SENTINEL in bundle.prescriptions[0].patch


def test_export_is_idempotent_three_times(tmp_path: Path) -> None:
    """Rule 5: three writes of the same bundle are byte-identical."""
    bundle = _bundle_with_raw_content()
    outs = []
    for i in range(3):
        out = tmp_path / f"share{i}.json"
        write_trace(bundle, out)
        outs.append(out.read_bytes())
    assert outs[0] == outs[1] == outs[2]


def test_scrub_marker_satisfies_every_patch_type_and_never_earns_l2() -> None:
    """F-03 lock (the comment-only claim now BITES): the inert marker passes all
    four patch_type validators on re-import AND cannot earn L2 through retry-cap
    grading."""
    from agent_run_ledger.core.io import _PATCH_SCRUB_MARKER
    from agent_run_ledger.core.models import PATCH_TYPES
    from agent_run_ledger.core.receipt import _grade_retry_cap, _is_retry_cap_diff

    base = _bundle_with_raw_content()
    for patch_type in PATCH_TYPES:
        rx = PrescriptionRecord(
            id="rx_m",
            run_id="run_t46",
            severity="low",
            root_cause="x",
            one_line_fix="x",
            evidence=[],
            patch_type=patch_type,
            patch=_PATCH_SCRUB_MARKER,
            expected_impact={},
            regression_test_template="",
        )
        bundle = TraceBundle(run=base.run, steps=base.steps, prescriptions=[rx])
        bundle.validate()  # raises if the marker fails this patch_type's validator

    assert _is_retry_cap_diff(_PATCH_SCRUB_MARKER) is False
    # even with a corroborated observed count, the marker grades L0 (not L2/L1)
    assert _grade_retry_cap("unified_diff", _PATCH_SCRUB_MARKER, 5) == "L0"


def test_cli_export_default_is_scrubbed_and_raw_local_opt_in(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    save_bundle(db, _bundle_with_raw_content())
    runner = CliRunner()

    share = tmp_path / "share.json"
    result = runner.invoke(
        app, ["export", "--run", "run_t46", "--out", str(share), "--db", str(db)]
    )
    assert result.exit_code == 0, result.output
    assert SENTINEL not in share.read_text(encoding="utf-8")

    local = tmp_path / "local.json"
    result = runner.invoke(
        app,
        ["export", "--run", "run_t46", "--out", str(local), "--db", str(db), "--raw-local"],
    )
    assert result.exit_code == 0, result.output
    assert SENTINEL in local.read_text(encoding="utf-8")
