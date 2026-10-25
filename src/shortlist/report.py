"""Write the README's Results section from results/*.json, so every number has a source file.

    python -m shortlist.report [--model gemini-3.1-flash-lite]

Replaces everything between the results markers in README.md. A missing results file shows
as TBD with the command that makes it.
"""

from __future__ import annotations

import argparse
import json
import re

from shortlist import llm
from shortlist.synthetic import ROOT

README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"


def _load(name: str) -> dict | None:
    path = ROOT / "results" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _f(x, digits: int = 3) -> str:
    return "TBD" if x is None or x == "TBD" else f"{x:.{digits}f}"


def parsing(model: str) -> list[str]:
    name = f"parse_accuracy_{model}.json"
    s = _load(name)
    if s is None:
        return ["### Parsing", "", "TBD: run `make parse-eval`."]
    rows = [f"| {k} | {_f(v)} |" for k, v in s["field_accuracy"].items()]
    sk = s["list_fields"]["skills"]
    return [
        "### Parsing",
        "",
        f"`{s['model']}`, {s['resumes']} resumes, {s['parse_failures']} parse failures, median "
        f"{s['seconds_per_resume']['median']} s per resume (`results/{name}`). A field is "
        "correct only if it equals the ground truth after case, spacing and punctuation are "
        "normalised; list fields must match as whole sets; a value filled in where the resume "
        "says nothing is an error.",
        "",
        "| Field | Accuracy |",
        "|---|---|",
        *rows,
        f"| **mean over fields** | **{_f(s['mean_field_accuracy'])}** |",
        f"| all 11 fields correct | {_f(s['all_fields_correct'])} |",
        "",
        f"Skills are the weak field: per item, precision {_f(sk['precision'])} and recall "
        f"{_f(sk['recall'])} (F1 {_f(sk['f1'])}), but one missed skill makes the set wrong. "
        'Most misses are synonyms not mapped to the canonical name ("attending calls" -> '
        '"phone handling"). The prompt lists only canonical names, not the catalog\'s '
        "spellings, so this measures the LLM, not a lookup table. Caveat: the prompt was "
        "adjusted once after reading 5 of these resumes, so this is not a held-out set.",
    ]


def matching(model: str) -> list[str]:
    name = f"match_{model}.json"
    s = _load(name)
    if s is None:
        return ["### Matching", "", "TBD: run `make match-eval`."]
    t = s["title_calls"]
    conf = t["confusion_catalog_vs_llm"]
    rels = ("same", "related", "different")
    rows = [
        f"| {a} | " + " | ".join(str(conf.get(a, {}).get(b, 0)) for b in rels) + " |" for a in rels
    ]
    same = s["checks_same_on_parsed_as_on_truth"]
    return [
        "### Matching",
        "",
        f"From `results/{name}`. Plain code checks experience, location, languages, "
        "certificates, salary and notice; each check is pass, soft (a near miss worth a call), "
        "fail or unknown. How far past a limit is still a near miss is the recruiter's slack, "
        "set per job (`salary_stretch_pct`, `notice_stretch_days`, `years_stretch`), chosen "
        "without looking at the generator's labelling rule.",
        "",
        f"**Parse errors barely reach matching:** run on parsed fields instead of ground truth, "
        f"each check gives the same result for {min(same.values()):.1%} to "
        f"{max(same.values()):.1%} of the 150 resumes.",
        "",
        f"**The one LLM call, is a previous title the same kind of work?** On {t['pairs']} "
        f"(job, title) pairs the LLM agrees with the generator's role catalog on "
        f"{_f(t['agree_with_catalog'])}. The catalog is one opinion; most disagreements are "
        "judgment calls (Receptionist for a Security Guard job: catalog *different*, LLM "
        '*related*). One is a plain LLM mistake: "Bike Rider" called *related*, not *same*, '
        "for a Delivery Executive job.",
        "",
        "| Catalog says | LLM: same | LLM: related | LLM: different |",
        "|---|---|---|---|",
        *rows,
    ]


def ranking(model: str) -> list[str]:
    name = f"rank_{model}.json"
    s = _load(name)
    if s is None:
        return ["### Ranking and ablation", "", "TBD: run `make rank-eval`."]
    gen, hand = s["generator_labels"], s["hand_labels"]

    def row(system: str) -> str:
        g, h = gen["ranking"][system], hand["ranking"][system]
        g_p, g_n = (g["precision@5"], g["ndcg@10"]) if g != "TBD" else ("TBD", "TBD")
        h_p, h_n = (h["precision@5"], h["ndcg@10"]) if h != "TBD" else ("TBD", "TBD")
        return f"| {system} | {_f(h_p)} | {_f(h_n)} | {_f(g_p)} | {_f(g_n)} |"

    r = s["reasons"]
    tier = gen["agent_tier_vs_label"]
    hand_tier = hand["agent_tier_vs_label"]
    hand_acc = hand_tier if hand_tier == "TBD" else hand_tier["accuracy"]
    return [
        "### Ranking and ablation",
        "",
        f"From `results/{name}`. Each job's candidates ranked three ways: **keyword** (BM25 of "
        "the job text over the resume text), **embedding** (`"
        f"{s['embedding_model']}` similarity) and the full **agent** (parse, match, rank). "
        "Averaged over the 5 jobs; NDCG gains good = 2, borderline = 1, poor = 0.",
        "",
        "| System | P@5, hand labels | NDCG@10, hand labels | P@5, generator labels "
        "| NDCG@10, generator labels |",
        "|---|---|---|---|---|",
        *(row(x) for x in ("keyword", "embedding", "agent")),
        "",
        f"Hand labels: {hand['resumes']} of 50 labelled so far (agreement of the generator's "
        f"labels with them: {_f(s.get('hand_vs_generator_agreement'))}); the hand-label "
        "columns stay TBD until `make label` is done. **Read the generator-label columns with "
        "care:** those labels "
        "come from rules that share the near-miss idea with the matcher, so the agent has a "
        "built-in advantage there. They show the pipeline does what it was built to do, not "
        "that it agrees with a recruiter.",
        "",
        f"The agent's tier (shortlist / call / reject) matches the generator's label for "
        f"{_f(tier['accuracy'])} of the 150 candidates (hand labels: {_f(hand_acc)}). They "
        "disagree both ways: the per-job slack is tighter than the generator's (10% over "
        "budget instead of 25%), so the agent rejects some of its borderline candidates, and "
        "where the generator rejects anyone with two near misses or gaps, the agent asks for "
        "a call.",
        "",
        f"**Reasons.** Of {r['candidates']} reasons, {r['passed_first_time']:.1%} passed the "
        f"grounding check first time, {r['passed_on_retry']:.1%} after one retry and "
        f"{r['plain_fallback']:.1%} fell back to a plain reason built from the checks. The "
        "check rejects a reason that cites nothing, cites a value the resume does not hold, "
        "states a number found in neither its citations nor the checks, or guesses gender. It "
        "checks citations and numbers, not every word of the sentence.",
    ]


def fairness(model: str) -> list[str]:
    name = f"fairness_{model}.json"
    s = _load(name)
    if s is None:
        return ["### Fairness", "", "TBD: run `make fairness-eval`."]
    sm = s["summary"]
    rows = []
    for v, label in (
        ("control", "same text again"),
        ("name", "name swapped"),
        ("college", "college swapped"),
    ):
        x = sm[v]
        emb = f"{x['embedding_rank_moved']}" if v != "control" else "-"
        rows.append(
            f"| {label} | {x['resumes']} | {x['mean_abs_score_delta']:g} | "
            f"{x['max_abs_score_delta']:g} | {x['tier_changed']} | {x['rank_moved']} | {emb} |"
        )
    return [
        "### Fairness",
        "",
        f"From `results/{name}`. Resumes re-run through the whole pipeline with only the name "
        "(to one from another language group, which often also changes the implied gender and "
        "religion) or only the institution (another invented one, same level and town) "
        'changed. "Same text again" re-parses the unchanged resume and measures the LLM\'s own '
        "noise.",
        "",
        "| Variant | Resumes | Mean abs score change | Max | Tier changed | Rank moved "
        "| Embedding baseline: rank moved |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        "The agent's score never reads the name or the institution, so it can only move when "
        "the LLM parses other fields differently. Re-parsing the same text changed fields in "
        f"{sm['control']['fields_changed']} of {sm['control']['resumes']} resumes, but a swap "
        f"changed other fields (mostly skills) in {sm['name']['fields_changed']} (name) and "
        f"{sm['college']['fields_changed']} (college): the parser is sensitive to text it "
        "should ignore. The effect on the decision was small here (see the table), but it is "
        "not zero. The embedding baseline reads the raw text, names included, and moves more "
        "often. One swap per resume on a small sample: a smoke test, not a bias audit.",
    ]


def build(model: str) -> str:
    parts = [
        "All results are on the **synthetic** resumes: they say how the system behaves on "
        "these five formats, not on real resumes. Generated by `make report` from `results/`.",
        *("\n".join(section(model)) for section in (parsing, matching, ranking, fairness)),
    ]
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=llm.model_name())
    args = ap.parse_args()
    text = README.read_text(encoding="utf-8")
    if START not in text:
        raise SystemExit(f"README.md has no {START} marker")
    body = f"{START}\n{build(args.model)}\n{END}"
    text = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: body, text, flags=re.S)
    README.write_text(text, encoding="utf-8")
    print("README.md results updated")


if __name__ == "__main__":
    main()
