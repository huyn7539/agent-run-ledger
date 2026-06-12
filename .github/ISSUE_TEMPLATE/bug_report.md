---
name: Bug report
about: Something broke, misfired, or failed to parse
labels: bug
---

<!-- Everything requested below is content-free by construction (bounded labels,
booleans, hashes, counts) — safe to paste. Do NOT paste raw session .jsonl files;
they contain your actual prompts and tool output. -->

**1. `arl --version` output**

```
paste here
```

**2. Full `--json` output of the verdict/sweep in question**

```json
paste here
```

**3. `arl selftest` output** (proves the pipeline works on your machine)

```
paste here
```

**4. Environment**

- OS:
- Python version:
- Which agent produced the session: Claude Code / Codex CLI / Agents SDK / other

**What happened, and what you expected instead**

<!-- If a session file is needed to reproduce a parser bug:
`arl export --run <id> --out trace.json` produces the content-free neutral form.
Attach that, never the raw .jsonl. -->
