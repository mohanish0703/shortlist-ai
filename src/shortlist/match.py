"""Matching node: check one candidate's `ResumeFields` against a `JobRequirement`.

Hard requirements are plain code: experience, location, languages, certificates, salary and
notice. The one fuzzy call, whether a previous job title is the same kind of work as the job,
goes to the LLM: one call per job for every title its candidates have held.

Each check is "pass", "soft" (misses, but a recruiter might still call), "fail" or "unknown"
(the resume does not say), and names the fields it read so a reason can cite them later. How
far past a limit still counts as "soft" is the recruiter's slack, set per job
(`JobRequirement.salary_stretch_pct` and friends), not the generator's labelling rule.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from google import genai
from pydantic import BaseModel, ConfigDict, Field

from shortlist import llm
from shortlist.catalog import NEARBY
from shortlist.schema import JobRequirement, ResumeFields

Status = Literal["pass", "soft", "fail", "unknown"]
Relation = Literal["same", "related", "different"]


class Check(BaseModel):
    name: str
    status: Status
    detail: str
    fields: list[str]  # the `ResumeFields` keys this check read


class TitleCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    relation: Relation
    reason: str = Field(description="One short sentence")


class TitleCalls(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calls: list[TitleCall]


class Match(BaseModel):
    job_id: str
    checks: list[Check]
    # The job's preferred skills that the candidate lists.
    preferred_skills: list[str]

    def with_status(self, status: Status) -> list[Check]:
        return [c for c in self.checks if c.status == status]


TITLE_SYSTEM = """You help a recruiter screen candidates for jobs in India.
For each previous job title of a candidate, say how it relates to the job being filled:
- same: the same work under another name ("Line Cook" for a "Commis Chef" job)
- related: different work, but the skills carry over and the person could start after brief
  training ("Waiter" for a "Bartender" job)
- different: unrelated work ("Electrician" for an "Accountant" job)
Judge from the title and the job description only. Return one entry per title, spelled exactly
as given."""

TITLE_PROMPT_ID = hashlib.sha256(
    (TITLE_SYSTEM + json.dumps(TitleCalls.model_json_schema(), sort_keys=True)).encode()
).hexdigest()[:8]


def title_key(title: str) -> str:
    return " ".join(title.casefold().split())


def classify_titles(
    job: JobRequirement,
    titles: list[str],
    client: genai.Client | None = None,
    model: str | None = None,
) -> dict[str, TitleCall]:
    """LLM call: how each title relates to the job. Keyed by normalised title."""
    unique = sorted({title_key(t): t for t in titles}.values())
    if not unique:
        return {}
    prompt = (
        f"Job: {job.title}\nDescription: {job.description}\n\nPrevious job titles:\n"
        + "\n".join(f"- {t}" for t in unique)
    )
    result = llm.generate_structured(
        client or llm.make_client(), TITLE_SYSTEM, prompt, TitleCalls, model
    )
    wanted = {title_key(t) for t in unique}
    return {title_key(c.title): c for c in result.calls if title_key(c.title) in wanted}


def _role(f: ResumeFields, job: JobRequirement, calls: dict[str, TitleCall]) -> Check:
    fields = ["job_titles"]
    if not f.job_titles:
        if job.min_years == 0:
            return Check(
                name="role",
                status="pass",
                detail="no previous job; freshers accepted",
                fields=fields,
            )
        return Check(name="role", status="unknown", detail="no previous job listed", fields=fields)
    known = [calls[title_key(t)] for t in f.job_titles if title_key(t) in calls]
    if not known:
        return Check(name="role", status="unknown", detail="titles not classified", fields=fields)
    rank = {"same": 0, "related": 1, "different": 2}
    best = min(known, key=lambda c: rank[c.relation])
    status: Status = {"same": "pass", "related": "soft", "different": "fail"}[best.relation]
    return Check(
        name="role",
        status=status,
        detail=f'worked as "{best.title}" ({best.relation} to {job.title}: {best.reason})',
        fields=fields,
    )


def _experience(f: ResumeFields, job: JobRequirement) -> Check:
    fields = ["years_experience"]
    years = f.years_experience
    if years is None:
        return Check(
            name="experience", status="unknown", detail="experience not stated", fields=fields
        )
    detail = f"{years:g} years vs {job.min_years:g} required"
    if years >= job.min_years:
        status: Status = "pass"
    elif years >= job.min_years - job.years_stretch:
        status = "soft"
    else:
        status = "fail"
    return Check(name="experience", status=status, detail=detail, fields=fields)


def _location(f: ResumeFields, job: JobRequirement) -> Check:
    fields = ["location"]
    if f.location is None:
        return Check(name="location", status="unknown", detail="location not stated", fields=fields)
    if f.location == job.location:
        status: Status = "pass"
        detail = f"lives in {f.location}"
    elif f.location in NEARBY.get(job.location, []):
        status, detail = "soft", f"lives in {f.location}, near {job.location}"
    else:
        status, detail = "fail", f"lives in {f.location}, job is in {job.location}"
    return Check(name="location", status=status, detail=detail, fields=fields)


def _languages(f: ResumeFields, job: JobRequirement) -> Check:
    fields = ["languages"]
    if not f.languages:
        return Check(
            name="languages", status="unknown", detail="languages not stated", fields=fields
        )
    missing = [lang for lang in job.languages_required if lang not in f.languages]
    if job.languages_any and not set(job.languages_any) & set(f.languages):
        missing.append(" or ".join(job.languages_any))
    speaks = ", ".join(f.languages)
    if missing:
        detail = f"speaks {speaks}; needs {', '.join(missing)}"
        return Check(name="languages", status="fail", detail=detail, fields=fields)
    return Check(name="languages", status="pass", detail=f"speaks {speaks}", fields=fields)


def _certifications(f: ResumeFields, job: JobRequirement) -> Check:
    fields = ["certifications"]
    missing = [c for c in job.required_certifications if c not in f.certifications]
    if not job.required_certifications:
        return Check(name="certifications", status="pass", detail="none required", fields=fields)
    if missing:
        # Not listing a licence is not proof of not having one: worth a call.
        detail = f"{', '.join(missing)} not listed"
        return Check(name="certifications", status="soft", detail=detail, fields=fields)
    detail = f"has {', '.join(job.required_certifications)}"
    return Check(name="certifications", status="pass", detail=detail, fields=fields)


def _limit(name: str, field: str, value: int | None, cap: int, soft_cap: float, unit: str) -> Check:
    if value is None:
        return Check(name=name, status="unknown", detail=f"{name} not stated", fields=[field])
    detail = f"{value} {unit} vs {cap} allowed"
    if value <= cap:
        status: Status = "pass"
    elif value <= soft_cap:
        status = "soft"
    else:
        status = "fail"
    return Check(name=name, status=status, detail=detail, fields=[field])


def match(f: ResumeFields, job: JobRequirement, calls: dict[str, TitleCall]) -> Match:
    """All checks for one candidate. Pure code: `calls` comes from `classify_titles`."""
    return Match(
        job_id=job.job_id,
        checks=[
            _role(f, job, calls),
            _experience(f, job),
            _location(f, job),
            _languages(f, job),
            _certifications(f, job),
            _limit(
                "salary",
                "expected_salary_monthly_inr",
                f.expected_salary_monthly_inr,
                job.max_salary_monthly_inr,
                job.max_salary_monthly_inr * (1 + job.salary_stretch_pct / 100),
                "INR/month",
            ),
            _limit(
                "notice",
                "notice_period_days",
                f.notice_period_days,
                job.max_notice_days,
                job.max_notice_days + job.notice_stretch_days,
                "days notice",
            ),
        ],
        preferred_skills=[s for s in job.preferred_skills if s in f.skills],
    )
