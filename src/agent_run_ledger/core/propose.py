"""Task 60 — deterministic TEMPLATED proposals (the proposer side of the
proposer/verifier split).

Templates only — no free-text generation, no model calls (determinism doctrine
unchanged). The ONLY log-derived slot is the tool name, validated against a
CLOSED charset (Codex spec-review CRITICAL: raw tool names from hostile session
logs reach prescriptions today; a slot that fails validation means ABSTAIN, the
fail-closed default — hostile text can never reach a CLAUDE.md line through a
template slot).

``proposal_id`` is a sha256 over the canonical preimage (domain, template
version, class, tool, line, evidence run ids) — uuid4 is forbidden here
(Codex CRITICAL: ``arl apply <id>`` must be replay-safe; the id re-derives
from the ledger, and apply re-verifies it before mutating anything).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from agent_run_ledger.core.receipt import build_receipts
from agent_run_ledger.core.storage import list_runs, load_bundle

PROPOSAL_DOMAIN = "arl-task60-proposal/v1"
TEMPLATE_VERSION = "retry-budget/v1"
PROPOSAL_CLASS = "retry_loop_budget"  # the ONE class this lane ships — precision before breadth
MIN_RECEIPTS = 3  # N>=3 same-shape receipts before proposing (abstain below)

# Mixed case is DELIBERATE (Codex P2 review F9 rejected with justification):
# the real tool namespace is capitalized (Bash, Read, WebFetch) — a lowercase-
# only slot would abstain on every genuine Claude Code tool, and uppercase
# ASCII letters add zero injection surface to this charset. Matched with
# fullmatch (never match): `$` would let a trailing newline ride through.
_SLOT_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
_STEP_ID_RE = re.compile(r"^step_id=(\S+)$")


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    proposal_class: str
    tool: str
    line: str
    evidence_run_ids: tuple[str, ...]
    receipt_count: int

    def display(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "class": self.proposal_class,
            "tool": self.tool,
            "line": self.line,
            "evidence_run_ids": list(self.evidence_run_ids),
            "receipt_count": self.receipt_count,
        }


def _template_line(tool: str) -> str:
    # Fixed text + ONE validated slot + the template version stamped inline.
    # Charset kept inside claudemd._validate_line's allowance (no backticks,
    # no angle brackets/braces, single line, bounded length).
    return (
        f"- ARL({PROPOSAL_CLASS}/{TEMPLATE_VERSION}) tool={tool}: after 2 identical "
        f"failed attempts of the same {tool} call, stop retrying - change strategy "
        "or ask the user."
    )


def proposal_id_for(tool: str, line: str, run_ids: tuple[str, ...]) -> str:
    # Canonical JSON preimage — unambiguous field boundaries even when a
    # crafted run id contains the old NUL delimiter (Codex P2 review F8:
    # join-based preimages collide across different evidence splits).
    preimage = json.dumps(
        [PROPOSAL_DOMAIN, TEMPLATE_VERSION, PROPOSAL_CLASS, tool, line, sorted(run_ids)],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def mine_proposals(
    db_path: Path, min_receipts: int = MIN_RECEIPTS
) -> tuple[list[Proposal], list[str]]:
    """Mine the ledger for repeated same-shape retry_loop receipts per tool.

    Returns ``(proposals, abstentions)``. Deterministic: the same ledger yields
    the same proposal ids on every machine, forever. A tool name failing the
    closed slot charset is an ABSTENTION (reported, never proposed)."""
    by_tool: dict[str, set[str]] = {}
    abstentions: list[str] = []
    seen_bad: set[str] = set()
    for run in list_runs(db_path):
        bundle = load_bundle(db_path, run.id)
        steps_by_id = {s.id: s for s in bundle.steps}
        for receipt in build_receipts(bundle):
            if receipt.observed_failure != "retry_loop":
                continue
            step_ids = {
                m.group(1)
                for ev in receipt.evidence
                if (m := _STEP_ID_RE.match(ev.strip())) is not None
            }
            if len(step_ids) != 1:
                continue
            step = steps_by_id.get(next(iter(step_ids)))
            if step is None:
                continue
            tool = step.name
            if not _SLOT_RE.fullmatch(tool):
                marker = tool[:80]
                if marker not in seen_bad:
                    seen_bad.add(marker)
                    abstentions.append(
                        "tool name fails the closed slot charset "
                        f"([A-Za-z0-9._-]{{1,64}}); abstaining: {marker!r}"
                    )
                continue
            by_tool.setdefault(tool, set()).add(run.id)
    proposals: list[Proposal] = []
    for tool in sorted(by_tool):
        run_ids = tuple(sorted(by_tool[tool]))
        if len(run_ids) < min_receipts:
            continue
        line = _template_line(tool)
        proposals.append(
            Proposal(
                proposal_id_for(tool, line, run_ids),
                PROPOSAL_CLASS,
                tool,
                line,
                run_ids,
                len(run_ids),
            )
        )
    return proposals, abstentions


def find_proposal(db_path: Path, proposal_id: str) -> Proposal | None:
    """Re-derive proposals from the ledger and return the matching one — apply
    NEVER trusts a caller-supplied proposal body, only the re-verified id."""
    proposals, _ = mine_proposals(db_path)
    for p in proposals:
        if p.proposal_id == proposal_id:
            return p
    return None


def any_failure_counts(db_path: Path, run_ids: list[str]) -> tuple[int, int]:
    """(n, k) over *run_ids*: k = runs whose receipts include ANY failure class
    (recomputed on read — judgments are never stored). This is the guardrail
    metric (Task 61): a rule that suppresses the targeted class while overall
    failures rise must revert, not be kept."""
    n = len(run_ids)
    k = 0
    for run_id in run_ids:
        if build_receipts(load_bundle(db_path, run_id)):
            k += 1
    return n, k


def tool_failure_counts(db_path: Path, tool: str, run_ids: list[str]) -> tuple[int, int]:
    """(n, k) over *run_ids*: k = runs whose receipts include a retry_loop on
    *tool* (recomputed on read — judgments are never stored)."""
    n = len(run_ids)
    k = 0
    for run_id in run_ids:
        bundle = load_bundle(db_path, run_id)
        steps_by_id = {s.id: s for s in bundle.steps}
        for receipt in build_receipts(bundle):
            if receipt.observed_failure != "retry_loop":
                continue
            step_ids = {
                m.group(1)
                for ev in receipt.evidence
                if (m := _STEP_ID_RE.match(ev.strip())) is not None
            }
            if len(step_ids) == 1:
                step = steps_by_id.get(next(iter(step_ids)))
                if step is not None and step.name == tool:
                    k += 1
                    break
    return n, k
