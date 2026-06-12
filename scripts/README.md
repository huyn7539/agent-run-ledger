# scripts/ — what's in here

Two kinds of files live here; neither ships in the installed package.

**Test/demo harness (load-bearing for the suite):**

- `live_capture_harness.py` — the shared SDK-driving harness reused by both the
  demo and `tests/test_live_capture_receipt.py`, so both exercise the same real
  run path.
- `demo_repair_receipt.py` — end-to-end demo of a receipt firing.

**Research instruments (the lab notebook, in the open):**

- `phase0_*.py`, `git_rework_miner.py`, `xsession_failure_decipher.py`,
  `failure_detectors.py` — the study tooling used to measure detector precision
  and recall against real session corpora before anything ships as a product
  detector. Candidate detectors live here until their precision is measured;
  several that looked plausible were rejected after grading (string-match
  detectors with 0 measured precision stay in the lab, permanently).

If you're evaluating whether to trust ARL's shipped detectors: this directory
is the paper trail of what was tested, what failed, and why only two classes
ship today.
