# End-to-end engineering smoke test — SYNTHETIC DATA ONLY

Execution date: 2026-11-21. A temporary, isolated UCI-shaped fixture was built with **252 train rows (21 people)** and **108 test rows (9 people)**, with 561 numeric attributes and six classes. Only a local test harness overrides the fixed expected row counts; the deliverable never bundles this fixture or its prediction scores.

The following stages completed in that isolated test before the stricter documentary subject-ID and per-class count gates were added: safe ZIP/CRC audit; data validation; quick-mode nested group CV for all six model families; illustrative random-window CV; feature-selection ablation; locked protocol; exactly one test evaluation; person-cluster bootstrap; 14 scientific figure exports; and refusal to execute the test a second time.

Additional full-notebook execution smoke was attempted using a temporary patched notebook and synthetic data, but did **not finish within the runner's execution limit**. Thus **clean-kernel notebook execution is not certified**. Its JSON structure and code conversion were validated.

Engineering results here establish code-path feasibility only. They **must never** be interpreted as results for the real UCI HAR dataset. Full real-data evaluation remains blocked by inability to download the official ZIP.
