# CLAUDE.md

Portfolio project: ShortlistAI, a LangGraph agent that ranks resumes against a job requirement
with grounded reasons and a recruiter review step. Plan and progress: `docs/PLAN.md`.

## Non-negotiable rules
1. **No fabricated results.** Every number in README/docs comes from a generated file
   (`data/summary.json`, later `results/*.json`). Unknown = "TBD". If the agent loses to a
   baseline, say so plainly.
2. **The data is SYNTHETIC** and every doc says so. Ground truth records only what the resume
   text states; `rubric_fit` labels are the generator's rules, not human judgment.
3. **Never overwrite `data/labels/labels.csv`** - it holds the owner's hand labels. The generator
   leaves it alone if it exists. Changing the generator changes the resumes, which invalidates
   the labels: tell the owner before doing that.
4. **No remote actions**: never create a remote repo, push or deploy. Give the user the commands.
5. Never commit `.env*` (except `.env.example`) or API keys. Code reads keys from `.env`.
6. Dependencies: exact pins in `pyproject.toml`.

## Commands (GNU make)
- `make venv && make install`, `make lint`, `make format`, `make test`
- `make data` (regenerate data/, deterministic with seed 42), `make label` (hand labelling)
- `make eval` (parse/match/rank/fairness evals into `results/`, cached; spends Gemini quota only
  for what is not cached), `make report` (README results from `results/*.json`)
- `make demo` (CLI session), `make api`, `make ui`; sessions and feedback live in `runs/`
  (git-ignored)

## Conventions
- `src/` layout, package `shortlist`. Python 3.11, ruff (line length 100), type hints,
  `from __future__ import annotations`. Always pass `encoding="utf-8"` for text files.
- Canonical names (skills, certificates, cities, languages) live in `shortlist.catalog`.
- "Present" in a resume counts up to `catalog.REFERENCE_DATE` (2026-10-01); a stint lasts
  end month minus start month.
