# Execution status / 2026-11-21

- Web documentation verification: UCI 240 official catalog, scikit-learn GroupKFold and historical README license conflict checked.
- Local environment: Python 3.13.5 with scientific dependencies installed.
- Official download: **FAILED** due to unavailable DNS/network in the execution sandbox. No alternative mirror was silently substituted.
- Original ZIP integrity audit: **NOT EXECUTED**. No real data in this package.
- Real-data grouped CV, model metrics, test predictions and performance figures: **NOT EXECUTED**. No fictional metrics are included.
- Engineering tests: **9 passed, 1 skipped** (official-dataset integration), after the final source edits.
- Synthetic E2E smoke: **PASS** for pipeline, one-time test guard and all **14 figure generation paths**, conducted before the final addition of strict documentary ID/class-count checks; see `SMOKE_TEST.md`.
- Notebook: 22 cells, `nbformat` JSON validation and nbconvert-to-script conversion passed; full clean-kernel execution **not certified** (smoke attempt exceeded runner's time limit).
- Commit/push/deploy/publication: **NOT PERFORMED**, according to user specification.

Reproduction in an online Linux environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
har-shift acquire --root .
har-shift audit --root .
har-shift train --root . --mode full
har-shift test --root .
har-shift figures --root .
```
