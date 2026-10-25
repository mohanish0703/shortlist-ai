"""Hand-label resumes in the terminal, blind to the generator's label.

Shows the job and one resume at a time and saves after every answer, so you can stop and
resume. Answers: g = good, b = borderline, p = poor, s = skip, q = quit. Anything typed after
the letter is saved as a note, e.g. "b knows Kannada but lives in Mysore".

    python -m shortlist.label
"""

from __future__ import annotations

import csv
import json

from shortlist.synthetic import DATA

LABELS = DATA / "labels" / "labels.csv"
CHOICES = {"g": "good", "b": "borderline", "p": "poor"}


def _job_card(job: dict) -> str:
    langs = []
    if job["languages_required"]:
        langs.append("all of " + ", ".join(job["languages_required"]))
    if job["languages_any"]:
        langs.append("one of " + ", ".join(job["languages_any"]))
    lines = [
        f"{job['job_id']}  {job['title']} x{job['openings']}  |  {job['location']}",
        f"  {job['description']}",
        f"  min experience: {job['min_years']:g} yrs   languages: {'; '.join(langs) or 'any'}",
        f"  budget: {job['max_salary_monthly_inr']}/month   "
        f"join within: {job['max_notice_days']} days",
    ]
    if job["required_certifications"]:
        lines.append("  must have: " + ", ".join(job["required_certifications"]))
    return "\n".join(lines)


def main() -> None:
    jobs = {j["job_id"]: j for j in json.loads((DATA / "jobs.json").read_text(encoding="utf-8"))}
    with LABELS.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    def save() -> None:
        with LABELS.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["resume_id", "job_id", "label", "notes"])
            w.writeheader()
            w.writerows(rows)

    todo = [r for r in rows if not r["label"]]
    print(f"{len(rows) - len(todo)}/{len(rows)} labelled. Ctrl+C or q to stop.\n")
    for n, row in enumerate(todo, 1):
        text = (DATA / "resumes" / f"{row['resume_id']}.txt").read_text(encoding="utf-8")
        print("=" * 78)
        print(_job_card(jobs[row["job_id"]]))
        print("-" * 78)
        print(f"{row['resume_id']}  ({n}/{len(todo)})\n")
        print(text)
        print("-" * 78)
        while True:
            answer = input("[g]ood [b]orderline [p]oor [s]kip [q]uit > ").strip()
            key = answer[:1].lower()
            if key in CHOICES:
                row["label"], row["notes"] = CHOICES[key], answer[1:].strip()
                save()
                break
            if key == "s":
                break
            if key == "q":
                return
    print("All done.")


if __name__ == "__main__":
    main()
