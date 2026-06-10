# Bug report: Fable 5 → Opus content-safety fallback fires silently (no transcript notice) and is a false positive on a benign QA codebase

**Channel:** GitHub issue at https://github.com/anthropics/claude-code/issues (or `/feedback` in-session)

## Summary

On Claude Code **v2.1.170**, with `"model": "claude-fable-5[1m]"` set in user
settings, every session opened inside a benign developer tooling repository
runs entirely on **`claude-opus-4-8`** instead of Fable 5. No `/model` command
is issued. This is the documented Fable-5 content-safety fallback
(cybersecurity/biology classifier) firing on the repository's workspace context
on the first request.

Two things make it report-worthy:

1. **Silent fallback — no transcript notice.** The docs
   ([model-config → Automatic model fallback](https://code.claude.com/docs/en/model-config))
   state the auto-switch "shows a notice in the transcript." In my sessions
   there was **no notice** in the UI and **no record in the saved transcript**;
   58/58 assistant turns simply came back as `claude-opus-4-8` with no
   indication a fallback occurred. The user has no signal that they are no
   longer on the model they configured.

2. **False positive on non-security, non-biology code.** The repo is a local
   CLI that grades AI-agent run logs ("agent run ledger"). The file that almost
   certainly trips the classifier contains, as *string data*, multilingual
   destruction verbs (`nuke`, `purge`, `wipe`, `kill`, `borrar`, `supprimer`,
   `löschen`, …) plus zero-width-character handling and a Cyrillic/Greek
   homoglyph→Latin confusable map. Its actual purpose is the opposite of
   offensive tooling: it *suppresses false accusations* in an honesty grader by
   recognizing user-issued deletion directives in any language/encoding. To a
   content classifier this reads as evasion/offensive-security tooling; it is
   not.

## Environment

- Claude Code: **2.1.170** (native installer, Windows 11)
- Account: Max 20× (`organizationType: claude_max`,
  `organizationRateLimitTier: default_claude_max_20x`, org admin)
- Configured model (user `settings.json`): `claude-fable-5[1m]`
- `additionalModelOptionsCache` contains exactly one entry: `claude-fable-5[1m]`
- `~/.claude.json`: `hasAvailableSubscription: false` (anomalous given the Max
  plan above — flagged for confirmation, not assumed to be the cause)

## Evidence

- Saved transcript for the affected session: **58/58 assistant turns** recorded
  `"model":"claude-opus-4-8"`; **0** `/model` slash-commands in the session.
- No `modelMode` / `fallbackModel` / `isFallback` / fallback-notice records of
  any kind in the transcript JSONL.
- No `fallbackModel` configured in settings (so the *availability* fallback
  chain is not involved — this is the *content* fallback path).

## Expected vs. actual

- **Expected (per docs):** when the Fable classifier flags a request, Claude
  Code re-runs it on Opus **and shows a notice in the transcript**; or, with
  `/config → "switch models when a message is flagged"` off, it pauses and asks.
- **Actual:** session silently ran on Opus for its entire duration with no
  notice and no pause, while configured for Fable 5.

## Asks

1. **Surface the fallback.** When content-safety fallback fires, persist a
   visible, transcript-recorded notice (the docs already promise this) so users
   know they are on Opus, not their configured model. Silent substitution of a
   weaker-for-the-task model with no signal is the core issue.
2. **Tune / allow opt-out for false positives** on benign codebases. A repo
   containing multilingual destruction verbs + Unicode-normalization tables as
   *defensive* string data should not be classified as offensive-security
   content on workspace context alone. Consider a per-project acknowledgment or
   an explicit "I am not doing security/biology work here" opt-out that keeps
   Fable for that workspace.
3. **Clarify `hasAvailableSubscription: false`** on a Max 20× admin account —
   confirm whether this is cosmetic or whether it interacts with model
   resolution.

## Workarounds (for other users hitting this)

- `claude --safe-mode` to confirm whether CLAUDE.md/skills/hooks are the trigger
  (git status + directory names still count).
- `/config` → turn off **"switch models when a message is flagged"** so a
  flagged request pauses and asks instead of silently switching.
- `/model fable` to return to Fable 5 after a fallback.
