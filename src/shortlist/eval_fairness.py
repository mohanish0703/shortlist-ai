"""Fairness: does the score move when only the name or only the college changes?

For up to 6 resumes per job (no typos, name and institution in the text), three variants run
through the full pipeline: parse (live LLM), match, score.
- control: the same text again, which measures the LLM's own run-to-run noise
- name: the name replaced by one from another language group in `catalog.NAMES`
- college: the institution replaced by another invented one at the same level and town
Each is compared with the original resume's parse from `make parse-eval`: change in score,
change in tier, change in rank among the job's 30 candidates, and fields (other than name and
institution) that came out different. The embedding baseline gets the same name and college
swaps, which shows how much a text-similarity ranker moves on them.

Parses are cached in results/fairness/<model>/<prompt id>/. Writes results/fairness_<model>.json.

    python -m shortlist.eval_fairness [--model gemini-3.1-flash-lite]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict

from google.genai import errors

from shortlist import llm
from shortlist.baselines import embedding_scores
from shortlist.catalog import INSTITUTIONS, JOBS, NAMES
from shortlist.eval_parse import RESULTS, cached_fields, load_records, norm
from shortlist.graph import cached_services
from shortlist.match import match
from shortlist.parse import PROMPT_ID, Extraction, extract, to_fields
from shortlist.rank import rank, score
from shortlist.schema import ResumeFields
from shortlist.synthetic import DATA, ROOT

PER_JOB = 6
VARIANTS = ("control", "name", "college")


def _group(name: str) -> str:
    first = name.split()[0]
    return next(g for g, (firsts, _) in NAMES.items() if first in firsts)


def swap(text: str, old: str, new: str) -> str:
    """Replace every occurrence, keeping upper case where the text is upper case."""

    def repl(m: re.Match) -> str:
        return new.upper() if m.group(0).isupper() else new

    return re.sub(re.escape(old), repl, text, flags=re.IGNORECASE)


def variants(r: dict, rng: random.Random) -> dict[str, tuple[str, str]] | None:
    """variant -> (text, what changed), or None if the resume cannot be used."""
    text = (DATA / "resumes" / f"{r['resume_id']}.txt").read_text(encoding="utf-8")
    f = r["fields"]
    if r["meta"]["typos"] or not f["institution"]:
        return None
    if f["name"].lower() not in text.lower() or f["institution"].lower() not in text.lower():
        return None
    name_from = f["name"]
    group = rng.choice(sorted(g for g in NAMES if g != _group(name_from)))
    firsts, lasts = NAMES[group]
    name_to = f"{rng.choice(firsts)} {rng.choice(lasts)}"
    head, _, town = f["institution"].rpartition(", ")
    others = [i for i in INSTITUTIONS[f["education"]] if not i.startswith(head)]
    if not others:
        return None
    college_to = rng.choice(others).format(town=town)
    # the email's local part is the first name plus digits: swap it too
    renamed = swap(text, name_from, name_to)
    email_from, email_to = name_from.split()[0].lower(), name_to.split()[0].lower()
    renamed = re.sub(
        rf"\b{re.escape(email_from)}(?=\d+@)",
        lambda m: email_to.upper() if m.group(0).isupper() else email_to,
        renamed,
        flags=re.IGNORECASE,
    )
    return {
        "control": (text, "unchanged"),
        "name": (renamed, f"{name_from} -> {name_to} ({group})"),
        "college": (swap(text, f["institution"], college_to), f"-> {college_to}"),
    }


def agent(job, calls, fields: dict[str, ResumeFields]) -> dict:
    """resume id -> (position, Ranked) for the job's candidates."""
    ranked = rank([score(rid, f, match(f, job, calls), job) for rid, f in fields.items()])
    return {x.resume_id: (i, x) for i, x in enumerate(ranked)}


def embed_rank(job, docs: dict[str, str]) -> dict:
    """resume id -> (position, similarity) under the embedding baseline."""
    s = embedding_scores(job, docs)
    order = sorted(s, key=lambda d: (-s[d], d))
    return {d: (i, s[d]) for i, d in enumerate(order)}


def run(model: str) -> dict:
    records = load_records()
    original = cached_fields(model)
    services = cached_services(model)
    rng = random.Random(0)
    chosen = defaultdict(dict)
    for job in JOBS:
        for r in (r for r in records if r["job_id"] == job.job_id):
            v = variants(r, rng)
            if v and len(chosen[job.job_id]) < PER_JOB:
                chosen[job.job_id][r["resume_id"]] = v

    cache = RESULTS / "fairness" / model / PROMPT_ID
    cache.mkdir(parents=True, exist_ok=True)
    client = llm.make_client()
    parsed: dict[tuple[str, str], ResumeFields] = {}
    for vs in chosen.values():
        for rid, by_variant in vs.items():
            for variant, (text, _) in by_variant.items():
                path = cache / f"{rid}-{variant}.json"
                if not path.exists():
                    try:
                        ex = extract(text, client, model)
                    except errors.APIError as e:
                        raise SystemExit(f"stopped at {rid}-{variant}: {e}\nrerun later") from e
                    path.write_text(json.dumps(ex.model_dump(), indent=2) + "\n", encoding="utf-8")
                saved = json.loads(path.read_text(encoding="utf-8"))
                parsed[rid, variant] = to_fields(Extraction.model_validate(saved))

    rows = []
    for job in JOBS:
        recs = [r for r in records if r["job_id"] == job.job_id]
        base = {r["resume_id"]: original[r["resume_id"]] for r in recs}
        titles = {
            t
            for f in [*base.values(), *(parsed[k] for k in parsed if k[0] in base)]
            for t in f.job_titles
        }
        calls = services.classify_titles(job, sorted(titles))
        texts = {
            r["resume_id"]: (DATA / "resumes" / f"{r['resume_id']}.txt").read_text(encoding="utf-8")
            for r in recs
        }

        before, before_emb = agent(job, calls, base), embed_rank(job, texts)
        for rid, by_variant in chosen[job.job_id].items():
            for variant, (text, change) in by_variant.items():
                after = agent(job, calls, base | {rid: parsed[rid, variant]})
                (i0, a), (i1, b) = before[rid], after[rid]
                f0, f1 = base[rid].model_dump(), parsed[rid, variant].model_dump()
                row = {
                    "resume_id": rid,
                    "job_id": job.job_id,
                    "variant": variant,
                    "change": change,
                    "score_delta": round(b.score - a.score, 1),
                    "tier_changed": a.tier != b.tier,
                    "rank_shift": i1 - i0,
                    "fields_changed": [
                        k
                        for k in f0
                        if k not in ("name", "institution") and norm(f0[k]) != norm(f1[k])
                    ],
                }
                if variant != "control":
                    emb = embed_rank(job, texts | {rid: text})
                    row["embedding_rank_shift"] = emb[rid][0] - before_emb[rid][0]
                    row["embedding_similarity_delta"] = round(emb[rid][1] - before_emb[rid][1], 4)
                rows.append(row)

    def summary(variant: str) -> dict:
        vs = [r for r in rows if r["variant"] == variant]
        out = {
            "resumes": len(vs),
            "mean_abs_score_delta": round(statistics.mean(abs(r["score_delta"]) for r in vs), 2),
            "max_abs_score_delta": max(abs(r["score_delta"]) for r in vs),
            "tier_changed": sum(r["tier_changed"] for r in vs),
            "rank_moved": sum(r["rank_shift"] != 0 for r in vs),
            "fields_changed": sum(bool(r["fields_changed"]) for r in vs),
        }
        if variant != "control":
            out["embedding_rank_moved"] = sum(r["embedding_rank_shift"] != 0 for r in vs)
            out["embedding_mean_abs_rank_shift"] = round(
                statistics.mean(abs(r["embedding_rank_shift"]) for r in vs), 2
            )
        return out

    return {
        "note": "SYNTHETIC resumes; one swap per resume, so small samples",
        "model": model,
        "parse_prompt_id": PROMPT_ID,
        "summary": {v: summary(v) for v in VARIANTS},
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=llm.model_name())
    args = ap.parse_args()
    result = run(args.model)
    out = RESULTS / f"fairness_{args.model}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    for r in result["rows"]:
        if r["score_delta"] or r["tier_changed"] or r["fields_changed"]:
            print(r)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
