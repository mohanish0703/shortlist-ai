"""Pydantic schemas: what the parser must extract from a resume, and what a job asks for.

`ResumeFields` is the contract between the parsing node and everything after it. The synthetic
generator writes the same schema as ground truth, so field-extraction accuracy is a direct
comparison.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Language = Literal[
    "Hindi",
    "English",
    "Kannada",
    "Tamil",
    "Telugu",
    "Marathi",
    "Bengali",
    "Odia",
    "Malayalam",
    "Urdu",
    "Assamese",
]

Education = Literal["Below 10th", "10th", "12th", "ITI", "Diploma", "Graduate", "Postgraduate"]

FitLabel = Literal["good", "borderline", "poor"]


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = item.strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


class ResumeFields(BaseModel):
    """Structured fields extracted from one resume. None / [] means "not stated"."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    # Titles as written in the resume, most recent first.
    job_titles: list[str] = Field(default_factory=list)
    # Canonical skill names from `catalog.SKILL_VOCAB`.
    skills: list[str] = Field(default_factory=list)
    # Total paid work experience in years, counted up to `catalog.REFERENCE_DATE`.
    years_experience: float | None = Field(default=None, ge=0, le=50)
    # Canonical city name, e.g. "Bengaluru" for "B'lore" or "Whitefield, Bangalore".
    location: str | None = None
    languages: list[Language] = Field(default_factory=list)
    # Canonical certificate names from `catalog.CERTIFICATIONS`.
    certifications: list[str] = Field(default_factory=list)
    education: Education | None = None
    institution: str | None = None
    expected_salary_monthly_inr: int | None = Field(default=None, ge=0)
    # 0 means an immediate joiner.
    notice_period_days: int | None = Field(default=None, ge=0)

    @field_validator("job_titles", "skills", "certifications")
    @classmethod
    def _clean_list(cls, v: list[str]) -> list[str]:
        return _dedupe(v)

    @field_validator("languages")
    @classmethod
    def _clean_languages(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


class JobRequirement(BaseModel):
    """One job posting. Hard requirements are checked in code; fuzzy ones go to the LLM."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    role: str  # key into `catalog.ROLES`
    title: str
    openings: int = Field(ge=1)
    location: str  # canonical city
    min_years: float = Field(ge=0)
    # The candidate must speak every language in `languages_required`
    # and at least one in `languages_any` (if it is non-empty).
    languages_required: list[Language] = Field(default_factory=list)
    languages_any: list[Language] = Field(default_factory=list)
    required_certifications: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    max_salary_monthly_inr: int = Field(ge=0)
    max_notice_days: int = Field(ge=0)
    # The recruiter's slack: a candidate this far past a limit is a near miss worth a call,
    # not a rejection. Set per job by the recruiter; the generator does not read them.
    salary_stretch_pct: float = Field(default=10, ge=0)
    notice_stretch_days: int = Field(default=15, ge=0)
    years_stretch: float = Field(default=0.5, ge=0)
    # The requirement as a recruiter would type it.
    description: str
