# ShortlistAI build plan

Each milestone adds one LangGraph idea and leaves something runnable. About 3-4 weeks alongside
coursework. Tick items off as they land.

## 1. Data
- [x] Resume schema (`ResumeFields`) and job schema (`JobRequirement`) in `src/shortlist/schema.py`
- [x] Synthetic generator: 150 resumes, 5 jobs x 30, five messy formats (formal, bullets, prose,
      Hinglish, application form), typos, missing fields, bad PDF extracts
- [x] Ground truth for every resume + the generator's own fit label (`rubric_fit`)
- [x] **Hand-label the 50 sampled resumes** (`make label`), blind to the generator's label
- [x] Report agreement between hand labels and the generator's labels (README, from
      `results/rank_<model>.json`)

## 2. Parsing node
- [x] Pick the LLM API: Gemini (`google-genai`), default `gemini-3.1-flash-lite` (free tier;
      `gemini-2.5-flash` is closed to new keys, `gemini-3.8-flash` was overloaded, and
      `gemini-3.5-flash` allows about 20 free requests a day)
- [x] Resume text -> `ResumeFields` with structured output (`shortlist.parse`); the LLM gives
      each job's months, code adds them up with "Present" = `REFERENCE_DATE`
- [x] Field-extraction accuracy per field against `data/ground_truth.jsonl` (`make parse-eval`
      -> `results/parse_accuracy_<model>.json`)
- [ ] Skills are the weak field (recall 0.908): try giving the catalog's surface forms, then
      re-measure
- [ ] Finish the `gemini-3.5-flash` comparison (39/150 cached; free quota is per day)

## 3. Matching node
- [x] Hard filters in plain code: location, languages, notice, salary, required certificates
      (`shortlist.match`; each check is pass / soft / fail / unknown and names its fields)
- [x] LLM only for fuzzy calls (is "Office Boy" close to "Housekeeping Staff"?): one call per
      job for all its candidates' titles
- [x] `make match-eval` -> `results/match_<model>.json`: title calls vs the role catalog, and
      how often parse errors change a check
- [x] Near-miss slack set per job by the recruiter (`salary_stretch_pct`, `notice_stretch_days`,
      `years_stretch` on `JobRequirement`), not copied from the generator's rule
- [ ] "Bike Rider" called related, not same, for Delivery Executive: decide whether to fix it
      in the prompt (would be tuning on the eval set) or leave it to the recruiter review

## 4. Ranking and explanation node
- [x] Score, then a short reason per candidate (`shortlist.rank`: tier shortlist / call /
      reject, then score; one LLM call per job for the reasons)
- [x] Every reason cites extracted fields; `rank.verify` rejects reasons that cite nothing, cite
      a wrong value, state an unbacked number or guess gender; one retry, then a plain reason

## 5. Graph wiring
- [x] LangGraph state graph: parse -> match -> rank -> review (`shortlist.graph`)
- [x] Conditional edge: parse failure -> retry -> manual-review queue
- [x] SQLite checkpointer so a session can be resumed (`runs/checkpoints.sqlite`; tested across
      a restart)

## 6. Human review
- [x] Interrupt: recruiter approves, edits or rejects the shortlist (CLI `shortlist.cli`, API,
      UI); an invalid decision asks again instead of getting stuck
- [x] Log every edit (feedback data): `runs/feedback.jsonl`, with the tier and score of every
      candidate the recruiter removed or added

## 7. Evaluation
- [x] Ranking quality (precision@5, NDCG) against the hand labels, with a random-order bar
- [x] Ranking quality against the generator's labels (caveat: shares the near-miss idea)
- [x] Ablation: keyword match vs embedding similarity vs full agent (`results/rank_<model>.json`)
- [x] Fairness: identical resumes with only name or college changed (`make fairness-eval`)
- [x] Reason grounding: share of reasons passing the check first time / on retry / fallback

## 8. Ship
- [x] FastAPI backend (`shortlist.api`), Streamlit UI (`shortlist.ui`), LangSmith tracing (env)
- [ ] Docker: `Dockerfile` and `docker-compose.yml` written, not yet built (no Docker on the
      development machine)
- [x] README results table from generated files (`make report`)

## Resume bullet
Section 7 has real numbers against the hand labels now; write the bullet from the README.
