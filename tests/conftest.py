"""Shared fakes: the LLM steps replaced by catalog lookups and plain reasons."""

from __future__ import annotations

from shortlist.catalog import JOBS
from shortlist.eval_match import catalog_relation
from shortlist.graph import Services
from shortlist.match import TitleCall, title_key
from shortlist.rank import template_reason
from shortlist.schema import ResumeFields

J1 = next(j for j in JOBS if j.job_id == "J1")


def _fields(name: str, **kw) -> ResumeFields:
    base = dict(
        name=name,
        job_titles=["Housekeeping Staff"],
        skills=["room cleaning"],
        years_experience=2.0,
        location="Bengaluru",
        languages=["Kannada"],
        expected_salary_monthly_inr=16_000,
        notice_period_days=0,
    )
    return ResumeFields(**(base | kw))


PEOPLE = {
    "R1": _fields("Asha"),
    "R2": _fields("Ravi", location="Patna"),  # rejected: far city
    "R3": _fields("Meena", location="Mysuru"),  # near miss
    "R4": _fields("Flaky"),  # fails once, then parses
    "R5": _fields("Broken"),  # never parses
}


def fake_services(tmp_path, calls: list[str]) -> Services:
    def parse(rid: str, text: str) -> ResumeFields:
        calls.append(rid)
        if rid == "R5" or (rid == "R4" and calls.count("R4") == 1):
            raise ValueError("bad output")
        return PEOPLE[rid]

    def titles(job, wanted):
        return {
            title_key(t): TitleCall(title=t, relation=catalog_relation(job, t), reason="fake")
            for t in wanted
        }

    def explain(job, items):
        return {r.resume_id: template_reason(r, f, m) for r, f, m in items}

    return Services(parse, titles, explain, feedback_log=tmp_path / "feedback.jsonl")
