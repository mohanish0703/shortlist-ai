"""Matching node on the synthetic resumes: the LLM's title calls, and how parsing errors carry over.

1. Title calls. Every (job, previous title) pair is classified by the LLM and compared with the
   generator's role catalog (`catalog.ROLES`: `titles` = same, `adjacent_titles` = related,
   anything else = different). The catalog is the generator's opinion, not a recruiter's, so
   every disagreement is listed for reading.
2. Carry-over. Every check runs twice with the same title calls: on the ground-truth fields and
   on the parsed fields. The share of checks that come out the same shows how much parsing
   errors change matching.

Title calls are cached in results/titles/<model>/<prompt id>/. Writes results/match_<model>.json.

    python -m shortlist.eval_match [--model gemini-3.1-flash-lite]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from shortlist import llm
from shortlist.catalog import JOBS, ROLES
from shortlist.eval_parse import RESULTS, cached_fields, load_records
from shortlist.match import TITLE_PROMPT_ID, TitleCall, classify_titles, match, title_key
from shortlist.parse import PROMPT_ID
from shortlist.schema import JobRequirement, ResumeFields
from shortlist.synthetic import ROOT


def catalog_relation(job: JobRequirement, title: str) -> str:
    spec = ROLES[job.role]
    if title_key(title) in {title_key(t) for t in spec["titles"]}:
        return "same"
    if title_key(title) in {title_key(t) for t in spec["adjacent_titles"]}:
        return "related"
    return "different"


def title_calls(model: str, titles_by_job: dict[str, set[str]]) -> dict[str, dict[str, TitleCall]]:
    """One LLM call per job, cached."""
    cache = RESULTS / "titles" / model / TITLE_PROMPT_ID
    cache.mkdir(parents=True, exist_ok=True)
    client = llm.make_client()
    out = {}
    for job in JOBS:
        path = cache / f"{job.job_id}.json"
        titles = sorted(titles_by_job[job.job_id])
        calls = {}
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            calls = {k: TitleCall.model_validate(v) for k, v in saved.items()}
        if any(title_key(t) not in calls for t in titles):
            calls = classify_titles(job, titles, client, model)
            dumped = {k: c.model_dump() for k, c in calls.items()}
            path.write_text(json.dumps(dumped, indent=2) + "\n", encoding="utf-8")
        out[job.job_id] = calls
    return out


def run(model: str) -> dict:
    jobs = {j.job_id: j for j in JOBS}
    records = load_records()
    parsed = cached_fields(model)
    missing = [r["resume_id"] for r in records if r["resume_id"] not in parsed]
    if missing:
        raise SystemExit(f"{len(missing)} resumes not parsed with {model}: run eval_parse first")

    truth = {r["resume_id"]: ResumeFields.model_validate(r["fields"]) for r in records}
    titles_by_job: dict[str, set[str]] = defaultdict(set)
    for r in records:
        rid = r["resume_id"]
        titles_by_job[r["job_id"]].update(truth[rid].job_titles + parsed[rid].job_titles)
    calls = title_calls(model, titles_by_job)

    # 1. title calls, on the ground-truth titles only
    confusion: dict[str, Counter] = defaultdict(Counter)
    disagreements = []
    pairs = {(r["job_id"], t) for r in records for t in truth[r["resume_id"]].job_titles}
    for job_id, title in sorted(pairs):
        expected = catalog_relation(jobs[job_id], title)
        call = calls[job_id].get(title_key(title))
        got = call.relation if call else "missing"
        confusion[expected][got] += 1
        if got != expected:
            disagreements.append(
                {
                    "job": jobs[job_id].title,
                    "title": title,
                    "catalog": expected,
                    "llm": got,
                    "reason": call.reason if call else None,
                }
            )

    # 2. carry-over of parsing errors
    same: Counter = Counter()
    statuses: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        job = jobs[r["job_id"]]
        on_truth = match(truth[r["resume_id"]], job, calls[job.job_id]).checks
        on_parsed = match(parsed[r["resume_id"]], job, calls[job.job_id]).checks
        for a, b in zip(on_truth, on_parsed, strict=True):
            same[a.name] += a.status == b.status
            statuses[b.name][b.status] += 1

    n = len(records)
    return {
        "note": "SYNTHETIC resumes; title calls compared with the generator's role catalog",
        "model": model,
        "title_prompt_id": TITLE_PROMPT_ID,
        "parse_prompt_id": PROMPT_ID,
        "title_calls": {
            "pairs": len(pairs),
            "agree_with_catalog": round(1 - len(disagreements) / len(pairs), 3),
            "confusion_catalog_vs_llm": {k: dict(v) for k, v in sorted(confusion.items())},
            "disagreements": disagreements,
        },
        "checks_same_on_parsed_as_on_truth": {k: round(v / n, 3) for k, v in same.items()},
        "check_status_on_parsed": {k: dict(v) for k, v in statuses.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=llm.model_name())
    args = ap.parse_args()
    summary = run(args.model)
    out = RESULTS / f"match_{args.model}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "check_status_on_parsed"}, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
