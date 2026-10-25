"""Field-extraction accuracy of the parsing node against `data/ground_truth.jsonl`.

Each parse is cached in results/parses/<model>/<prompt id>/, so a run stopped by a rate limit
picks up where it left off. Writes results/parse_accuracy_<model>.json and prints a table.

    python -m shortlist.eval_parse [--model gemini-3.1-flash-lite] [--limit 20]

A field is correct when it equals the ground truth after case, spacing and punctuation are
normalised; list fields compare as sets; years compare at one decimal. "Not stated" is a value
too: a parser that fills in a field the resume leaves out is wrong.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from google.genai import errors
from pydantic import ValidationError

from shortlist import llm
from shortlist.parse import PROMPT_ID, Extraction, extract, to_fields
from shortlist.schema import ResumeFields
from shortlist.synthetic import DATA, ROOT

RESULTS = ROOT / "results"
FIELDS = list(ResumeFields.model_fields)
LIST_FIELDS = ("job_titles", "skills", "languages", "certifications")


def norm(value):
    """Comparable form of a field value; anything empty becomes None."""
    if isinstance(value, str):
        return " ".join(re.sub(r"\W+", " ", value.casefold()).split()) or None
    if isinstance(value, list):
        return sorted({v for v in map(norm, value) if v is not None}) or None
    if isinstance(value, float):
        return round(value, 1)
    return value


def compare(pred: dict, true: dict) -> dict[str, str]:
    """Per field: "ok", "missed" (not extracted), "invented" (not stated) or "wrong"."""
    out = {}
    for f in FIELDS:
        p, t = norm(pred[f]), norm(true[f])
        if p == t:
            out[f] = "ok"
        elif p is None:
            out[f] = "missed"
        elif t is None:
            out[f] = "invented"
        else:
            out[f] = "wrong"
    return out


def list_counts(pred: dict, true: dict, field: str) -> tuple[int, int, int]:
    """True positives, predicted items, true items for one list field."""
    p, t = set(norm(pred[field]) or []), set(norm(true[field]) or [])
    return len(p & t), len(p), len(t)


def _rate(num: float, den: float) -> float | None:
    return round(num / den, 3) if den else None


def summarize(records: list[dict], parsed: dict[str, dict], model: str) -> dict:
    """`parsed` maps resume id -> cached result; a parse failure counts every field as wrong."""
    ok = {f: 0 for f in FIELDS}
    errs = {f: defaultdict(int) for f in FIELDS}
    lists = {f: [0, 0, 0] for f in LIST_FIELDS}
    groups: dict[str, list[float]] = defaultdict(list)
    all_ok = failures = 0
    scored = [r for r in records if r["resume_id"] in parsed]
    for r in scored:
        cached = parsed[r["resume_id"]]
        meta = r["meta"]
        noise = (
            "bad_pdf_extract" if meta["bad_pdf_extract"] else "typos" if meta["typos"] else "clean"
        )
        if "extraction" not in cached:
            failures += 1
            share = 0.0
        else:
            pred = to_fields(Extraction.model_validate(cached["extraction"])).model_dump()
            result = compare(pred, r["fields"])
            for f, outcome in result.items():
                ok[f] += outcome == "ok"
                if outcome != "ok":
                    errs[f][outcome] += 1
            for f in LIST_FIELDS:
                for i, n in enumerate(list_counts(pred, r["fields"], f)):
                    lists[f][i] += n
            share = sum(o == "ok" for o in result.values()) / len(FIELDS)
            all_ok += share == 1
        groups[f"template:{meta['template']}"].append(share)
        groups[f"noise:{noise}"].append(share)

    n = len(scored)
    seconds = [c["seconds"] for c in parsed.values()]
    return {
        "note": "SYNTHETIC resumes; accuracy against the generator's ground truth",
        "model": model,
        "prompt_id": PROMPT_ID,
        "date": date.today().isoformat(),
        "resumes": n,
        "complete": n == len(records),
        "parse_failures": failures,
        "field_accuracy": {f: _rate(ok[f], n) for f in FIELDS},
        "field_errors": {f: dict(errs[f]) for f in FIELDS},
        "list_fields": {
            f: {
                "precision": _rate(tp, npred),
                "recall": _rate(tp, ntrue),
                "f1": _rate(2 * tp, npred + ntrue),
            }
            for f, (tp, npred, ntrue) in lists.items()
        },
        "mean_field_accuracy": _rate(sum(ok.values()), n * len(FIELDS)),
        "all_fields_correct": _rate(all_ok, n),
        "by_group": {
            g: {"resumes": len(v), "mean_field_accuracy": round(statistics.mean(v), 3)}
            for g, v in sorted(groups.items())
        },
        "seconds_per_resume": {
            "median": round(statistics.median(seconds), 2) if seconds else None,
            "max": max(seconds, default=None),
        },
    }


def load_records() -> list[dict]:
    lines = (DATA / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def cache_dir(model: str) -> Path:
    return RESULTS / "parses" / model / PROMPT_ID


def cached_fields(model: str) -> dict[str, ResumeFields]:
    """Parsed fields per resume id from the cache of the current prompt; failures left out."""
    out = {}
    for p in cache_dir(model).glob("R*.json"):
        cached = json.loads(p.read_text(encoding="utf-8"))
        if "extraction" in cached:
            out[p.stem] = to_fields(Extraction.model_validate(cached["extraction"]))
    return out


def run(model: str, limit: int | None) -> dict:
    records = load_records()[:limit]
    cache = cache_dir(model)
    cache.mkdir(parents=True, exist_ok=True)
    client = llm.make_client()
    for i, r in enumerate(records, 1):
        rid = r["resume_id"]
        path = cache / f"{rid}.json"
        if path.exists():
            continue
        text = (DATA / "resumes" / f"{rid}.txt").read_text(encoding="utf-8")
        start = time.perf_counter()
        try:
            out: dict = {"extraction": extract(text, client, model).model_dump()}
        except ValidationError as e:  # the model answered, but not with a valid Extraction
            out = {"error": str(e)[:1000]}
        except errors.APIError as e:  # quota or outage after retries: stop, keep the cache
            print(f"stopped at {rid}: {e}\nrerun the same command later to continue")
            break
        out["seconds"] = round(time.perf_counter() - start, 2)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[{i}/{len(records)}] {rid} {out['seconds']}s {'FAILED' if 'error' in out else ''}")

    parsed = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in cache.glob("R*.json")
        if p.stem in {r["resume_id"] for r in records}
    }
    summary = summarize(records, parsed, model)
    # Only a full run is a result; a partial one is printed but not saved.
    if limit is None and summary["complete"]:
        out_path = RESULTS / f"parse_accuracy_{model}.json"
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_path.relative_to(ROOT)}")
    return summary


def print_table(s: dict) -> None:
    print(f"\n{s['model']} (prompt {s['prompt_id']}): {s['resumes']} resumes, ", end="")
    print(f"{s['parse_failures']} parse failures")
    print(f"{'field':<30}{'accuracy':>9}  errors")
    for f in FIELDS:
        acc = s["field_accuracy"][f]
        detail = ", ".join(f"{k} {v}" for k, v in sorted(s["field_errors"][f].items()))
        print(f"{f:<30}{acc if acc is not None else '-':>9}  {detail}")
    print(f"{'mean':<30}{s['mean_field_accuracy']:>9}")
    print(f"{'all fields correct':<30}{s['all_fields_correct']:>9}")
    for g, v in s["by_group"].items():
        print(f"  {g:<28}{v['mean_field_accuracy']:>9}  ({v['resumes']} resumes)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=llm.model_name())
    ap.add_argument("--limit", type=int, help="first N resumes only (no results file written)")
    args = ap.parse_args()
    print_table(run(args.model, args.limit))


if __name__ == "__main__":
    main()
