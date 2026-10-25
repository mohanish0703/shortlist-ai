"""Parsing node: resume text -> `ResumeFields`.

The LLM copies what the resume says into `Extraction`: every job with its start and end month,
and skills, certificates and languages as the canonical names in `shortlist.catalog` (the JSON
schema restricts them to those names). Code then does what code is better at: it adds up the
months, counting "Present" up to `catalog.REFERENCE_DATE`, and maps city spellings to the
canonical city.

    python -m shortlist.parse data/resumes/R003.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from google import genai
from pydantic import BaseModel, ConfigDict, Field

from shortlist import llm
from shortlist.catalog import CERTIFICATIONS, CITIES, REFERENCE_DATE, SKILL_VOCAB
from shortlist.schema import Education, Language, ResumeFields

SkillName = Literal[tuple(SKILL_VOCAB)]
CertificateName = Literal[tuple(CERTIFICATIONS)]

MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
REF_MONTH = REFERENCE_DATE.year * 12 + REFERENCE_DATE.month - 1
# every way a catalog city is written -> canonical name
CITY_FORMS = {
    form.casefold(): city for city, (_, forms) in CITIES.items() for form in [city, *forms]
}


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Job title")
    start: str | None = Field(default=None, description="Start month as YYYY-MM, null if not given")
    end: str | None = Field(
        default=None,
        description='End month as YYYY-MM; "present" if still working there; null if not given',
    )


class Extraction(BaseModel):
    """What the LLM returns. `to_fields` turns it into `ResumeFields`."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, description="Full name in normal capitalisation")
    jobs: list[Job] = Field(default_factory=list, description="Paid jobs, most recent first")
    stated_total_years: float | None = Field(
        default=None,
        description="Total experience in years only if the resume states a total; 0 for a fresher",
    )
    skills: list[SkillName] = Field(default_factory=list)
    location: str | None = Field(
        default=None, description="City the candidate lives in now, as a standard city name"
    )
    languages: list[Language] = Field(default_factory=list)
    certifications: list[CertificateName] = Field(default_factory=list)
    education: Education | None = Field(default=None, description="Highest education level")
    institution: str | None = Field(default=None, description="School or college of that education")
    expected_salary_monthly_inr: int | None = Field(
        default=None, description="Expected salary per month in rupees"
    )
    notice_period_days: int | None = Field(
        default=None, description="Days until the candidate can join; 0 for immediate"
    )


SYSTEM = f"""You extract facts from one job applicant's resume into JSON.

Rules:
- Use only what the resume states. If something is not stated, use null or an empty list. Never
  guess. Resumes may be in Hinglish, have typos, or be upper-case text run together from a PDF.
- Names, job titles and institutions: copy them as written, but fix obvious typos and use
  normal capitalisation ("SECURTY GAURD" -> "Security Guard"). Keep the town that follows an
  institution's name ("Govt. College, Satara").
- Skills, certifications and languages: map each one the resume mentions to the closest allowed
  name in the schema (e.g. "Bike licence" -> "Two-Wheeler Driving Licence"). Leave out anything
  with no close match.
- Education: the highest level stated. SSLC/SSC/Matriculation = 10th; PUC/HSC/Intermediate =
  12th; B.Com/BA/B.Sc/BBA = Graduate; M.Com/MA/MBA = Postgraduate.
- Dates: write months as YYYY-MM. Two-digit years are 20xx. Today is {REFERENCE_DATE:%Y-%m-%d}.
- Salary: monthly rupees. "15k" = 15000; an annual figure such as "2.4 LPA" = 240000 / 12.
- Notice: in days. 1 week = 7, 1 month = 30; "immediate joiner" = 0.
"""
# Changes whenever the prompt or the output schema does; cached parses are kept per prompt.
PROMPT_ID = hashlib.sha256(
    (SYSTEM + json.dumps(Extraction.model_json_schema(), sort_keys=True)).encode()
).hexdigest()[:8]


def to_fields(ex: Extraction) -> ResumeFields:
    """Apply the deterministic steps and validate against the shared schema."""
    location = None
    if ex.location:
        location = CITY_FORMS.get(ex.location.strip().casefold(), ex.location.strip())
    return ResumeFields(
        name=ex.name,
        job_titles=[j.title for j in ex.jobs],
        skills=list(ex.skills),
        years_experience=years_experience(ex),
        location=location,
        languages=list(ex.languages),
        certifications=list(ex.certifications),
        education=ex.education,
        institution=ex.institution,
        expected_salary_monthly_inr=ex.expected_salary_monthly_inr,
        notice_period_days=ex.notice_period_days,
    )


def _month(value: str | None) -> int | None:
    if value is not None and value.strip().lower() == "present":
        return REF_MONTH
    m = MONTH_RE.match(value.strip()) if value else None
    if not m or not 1 <= int(m.group(2)) <= 12:
        return None
    return int(m.group(1)) * 12 + int(m.group(2)) - 1


def years_experience(ex: Extraction) -> float | None:
    """Sum of (end month - start month) over all jobs; the stated total if any date is missing."""
    spans = [(_month(j.start), _month(j.end)) for j in ex.jobs]
    if spans and all(s is not None and e is not None and e >= s for s, e in spans):
        return round(sum(e - s for s, e in spans) / 12, 1)
    if ex.stated_total_years is not None:
        return round(ex.stated_total_years, 1)
    return None


def extract(text: str, client: genai.Client | None = None, model: str | None = None) -> Extraction:
    return llm.generate_structured(
        client or llm.make_client(), SYSTEM, f"Resume:\n\n{text}", Extraction, model
    )


def parse_resume(
    text: str, client: genai.Client | None = None, model: str | None = None
) -> ResumeFields:
    return to_fields(extract(text, client, model))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("resume", type=Path)
    args = ap.parse_args()
    fields = parse_resume(args.resume.read_text(encoding="utf-8"))
    print(json.dumps(fields.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
