---
name: Receipt report
about: "A receipt was WRONG, a failure was MISSED, or you APPLIED a fix — the three things this project most wants to hear"
labels: receipt-report
---

**Which kind?** (pick one)

- [ ] **Wrong receipt** — ARL accused a run that was actually fine (false
      accusation is the worst bug this tool can have)
- [ ] **Miss** — a failure you know happened, graded clean (describe the
      session shape: unattended loop? CI lane? what failed?)
- [ ] **Applied fix** — you actually applied a receipt's fix direction
      (`arl mark-applied <run-id>`) — this is the project's success metric

**The receipt / verdict JSON** (content-free, safe to paste)

```json
paste `arl verdict ... --json` here
```

**What actually happened in the run, in your words**

**`arl --version`:**
