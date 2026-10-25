"""Ranking quality, the ablation and reason grounding on the synthetic resumes.

Each job's 30 candidates are ranked by three systems:
- keyword: BM25 of the job text over the resume text (`baselines.bm25`)
- embedding: Gemini embedding similarity of job and resume text (`baselines.embedding_scores`)
- agent: parse -> match -> rank, the full pipeline (tier first, then score)

Each ranking is scored against two sets of labels:
- hand labels (`data/labels/labels.csv`, 10 resumes per job): the real test; "TBD" until the
  owner has labelled them
- the generator's labels (`generator_fit`, all 30 per job): rules, not judgment, and they share
  the near-miss idea with the matcher, so the agent has a built-in advantage on them

Metrics per job, averaged over the 5 jobs: precision@5 (share of the top 5 labelled good) and
NDCG@10 with gains good = 2, borderline = 1, poor = 0. Against hand labels each ranking is
restricted to the 10 labelled resumes.

Reasons: for all 150 candidates, how many of the LLM's reasons passed `rank.verify` first time,
after the retry, or were replaced by the plain reason. Writes results/rank_<model>.json.

    python -m shortlist.eval_rank [--model gemini-3.1-flash-lite]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict

from shortlist import llm
from shortlist.baselines import EMBED_MODEL, bm25, embedding_scores, job_query
from shortlist.catalog import JOBS
from shortlist.eval_parse import RESULTS, cached_fields, load_records
from shortlist.graph import cached_services
from shortlist.match import match
from shortlist.rank import REASON_PROMPT_ID, rank, score
from shortlist.synthetic import DATA, ROOT

GAIN = {"good": 2, "borderline": 1, "poor": 0}
SYSTEMS = ("keyword", "embedding", "agent")


def ndcg(labels_in_order: list[str], k: int = 10) -> float:
    def dcg(ls: list[str]) -> float:
        return sum((2 ** GAIN[lab] - 1) / math.log2(i + 2) for i, lab in enumerate(ls[:k]))

    ideal = dcg(sorted(labels_in_order, key=lambda lab: -GAIN[lab]))
    return dcg(labels_in_order) / ideal if ideal else 0.0


def precision_at(labels_in_order: list[str], k: int = 5) -> float:
    top = labels_in_order[:k]
    return sum(lab == "good" for lab in top) / len(top)


def score_ranking(order: list[str], labels: dict[str, str]) -> dict[str, float]:
    ls = [labels[r] for r in order if r in labels]
    return {"precision@5": precision_at(ls), "ndcg@10": ndcg(ls)}


def hand_labels() -> dict[str, str]:
    with (DATA / "labels" / "labels.csv").open(encoding="utf-8", newline="") as fh:
        return {r["resume_id"]: r["label"] for r in csv.DictReader(fh) if r["label"]}


def _problem_kind(p: str) -> str:
    return re.sub(r" is .*| \d.*", "", p).replace("number", "unbacked number")


def run(model: str) -> dict:
    records = load_records()
    parsed = cached_fields(model)
    services = cached_services(model)
    hand = hand_labels()
    by_job = defaultdict(list)
    for r in records:
        by_job[r["job_id"]].append(r)

    orders: dict[str, dict[str, list[str]]] = defaultdict(dict)
    fits: dict[str, str] = {}
    reason_sources: Counter = Counter()
    problems: Counter = Counter()
    for job in JOBS:
        recs = by_job[job.job_id]
        texts = {
            r["resume_id"]: (DATA / "resumes" / f"{r['resume_id']}.txt").read_text(encoding="utf-8")
            for r in recs
        }
        for name, scores in (
            ("keyword", bm25(job_query(job), texts)),
            ("embedding", embedding_scores(job, texts)),
        ):
            orders[job.job_id][name] = sorted(scores, key=lambda d: (-scores[d], d))

        fields = {r["resume_id"]: parsed[r["resume_id"]] for r in recs}
        calls = services.classify_titles(
            job, sorted({t for f in fields.values() for t in f.job_titles})
        )
        matches = {rid: match(f, job, calls) for rid, f in fields.items()}
        ranking = rank([score(rid, fields[rid], matches[rid], job) for rid in fields])
        orders[job.job_id]["agent"] = [x.resume_id for x in ranking]
        fits |= {x.resume_id: x.fit for x in ranking}
        reasons = services.explain(
            job, [(x, fields[x.resume_id], matches[x.resume_id]) for x in ranking]
        )
        for e in reasons.values():
            reason_sources[e.source] += 1
            problems.update({_problem_kind(p) for p in e.rejected})

    generator = {r["resume_id"]: r["generator_fit"] for r in records}
    results: dict[str, dict] = {}
    for label_set, labels in (("generator_labels", generator), ("hand_labels", hand)):
        complete = label_set == "generator_labels" or len(labels) == 50
        per_system: dict[str, dict | str] = {}
        for system in SYSTEMS:
            if not complete:
                per_system[system] = "TBD"
                continue
            per_job = [score_ranking(orders[j.job_id][system], labels) for j in JOBS]
            per_system[system] = {
                m: round(statistics.mean(x[m] for x in per_job), 3) for m in per_job[0]
            }
        agent_tier = (
            {
                "accuracy": round(
                    sum(fits[r] == lab for r, lab in labels.items()) / len(labels), 3
                ),
                "confusion_label_vs_agent": {
                    lab: dict(Counter(fits[r] for r, x in labels.items() if x == lab))
                    for lab in ("good", "borderline", "poor")
                },
            }
            if complete
            else "TBD"
        )
        results[label_set] = {
            "resumes": len(labels),
            "ranking": per_system,
            "agent_tier_vs_label": agent_tier,
        }

    total = sum(reason_sources.values())
    # How far the generator's rule is from a person's judgment, on the hand-labelled resumes.
    agreement = (
        round(sum(generator[r] == lab for r, lab in hand.items()) / len(hand), 3)
        if len(hand) == 50
        else "TBD"
    )
    return {
        "note": "SYNTHETIC resumes; see the module docstring for what each label set means",
        "model": model,
        "embedding_model": EMBED_MODEL,
        "reason_prompt_id": REASON_PROMPT_ID,
        "hand_vs_generator_agreement": agreement,
        **results,
        "reasons": {
            "candidates": total,
            "passed_first_time": round(reason_sources["llm"] / total, 3),
            "passed_on_retry": round(reason_sources["llm_retry"] / total, 3),
            "plain_fallback": round(reason_sources["template"] / total, 3),
            "problems_seen": dict(problems.most_common()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=llm.model_name())
    args = ap.parse_args()
    summary = run(args.model)
    out = RESULTS / f"rank_{args.model}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
