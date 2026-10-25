"""Scoring, ordering and the grounding check for reasons; no API calls."""

from __future__ import annotations

import pytest

from shortlist.catalog import JOBS
from shortlist.eval_match import catalog_relation
from shortlist.match import TitleCall, match, title_key
from shortlist.rank import Cite, Reason, rank, score, template_reason, verify
from shortlist.schema import ResumeFields
from shortlist.synthetic import generate

JOB = {j.job_id: j for j in JOBS}


def _calls(job, titles):
    return {
        title_key(t): TitleCall(title=t, relation=catalog_relation(job, t), reason="catalog")
        for t in titles
    }


def _fields(**kw) -> ResumeFields:
    base = dict(
        name="Asha Rao",
        job_titles=["Housekeeping Staff"],
        skills=["room cleaning", "bathroom sanitation"],
        years_experience=3.0,
        location="Bengaluru",
        languages=["Kannada"],
        expected_salary_monthly_inr=16_000,
        notice_period_days=0,
    )
    return ResumeFields(**(base | kw))


def _scored(rid="R1", **kw):
    f = _fields(**kw)
    m = match(f, JOB["J1"], _calls(JOB["J1"], f.job_titles))
    return score(rid, f, m, JOB["J1"]), f, m


def test_tiers():
    assert _scored()[0].tier == "shortlist"
    assert _scored(expected_salary_monthly_inr=None)[0].tier == "shortlist"  # cheap to ask
    assert _scored(location=None)[0].tier == "call"  # must know where they live
    assert _scored(location="Mysuru")[0].tier == "call"
    assert _scored(location="Patna")[0].tier == "reject"


def test_score_rewards_skills_and_experience():
    base = _scored()[0].score
    assert _scored(skills=["room cleaning"])[0].score == base - 5
    assert _scored(years_experience=2.0)[0].score == base - 2


def test_rank_orders_by_tier_then_score_then_id():
    a = _scored("R3", location="Patna")[0]  # reject, however good otherwise
    b = _scored("R2", skills=[])[0]
    c = _scored("R1")[0]
    d = _scored("R0", skills=[])[0]
    assert [r.resume_id for r in rank([a, b, c, d])] == ["R1", "R0", "R2", "R3"]


def _reason(text, *cites):
    return Reason(resume_id="R1", text=text, cites=[Cite(field=f, value=v) for f, v in cites])


def test_verify_accepts_a_grounded_reason():
    _, f, m = _scored()
    r = _reason(
        "3 years as Housekeeping Staff in Bengaluru; asks 16,000 against an 18000 budget.",
        ("job_titles", "Housekeeping Staff"),
        ("years_experience", "3"),
        ("location", "Bengaluru"),
        ("expected_salary_monthly_inr", "16000"),
    )
    assert verify(r, f, m) == []  # 18000 comes from the salary check


@pytest.mark.parametrize(
    ("reason", "problem"),
    [
        (_reason("Strong candidate."), "cites nothing"),
        (_reason("Lives in Mysuru.", ("location", "Mysuru")), "location is 'Bengaluru'"),
        (_reason("5 years of work.", ("job_titles", "Housekeeping Staff")), "number 5"),
        (_reason("Fine.", ("institution", "Govt School")), "institution is None"),
        (_reason("Her Kannada is good.", ("languages", "Kannada")), "guesses gender"),
    ],
)
def test_verify_rejects(reason, problem):
    _, f, m = _scored()
    assert any(problem in p for p in verify(reason, f, m))


def test_citation_formats():
    _, f, m = _scored(languages=["Kannada", "Hindi"])
    ok = _reason("Speaks both.", ("languages", '["Kannada", "Hindi"]'), ("years_experience", "3.0"))
    assert verify(ok, f, m) == []
    bad = _reason("Speaks Tamil.", ("languages", '["Kannada", "Tamil"]'))
    assert verify(bad, f, m)


def test_not_stated_is_a_valid_citation():
    _, f, m = _scored(location=None)
    assert verify(_reason("Location unknown.", ("location", "not stated")), f, m) == []


def test_template_reason_is_always_grounded():
    """The fallback must pass the same check, for every synthetic candidate."""
    for r in generate(per_job=30, seed=42):
        job, f = JOB[r["job_id"]], ResumeFields.model_validate(r["fields"])
        m = match(f, job, _calls(job, f.job_titles))
        ranked = score(r["resume_id"], f, m, job)
        expl = template_reason(ranked, f, m)
        reason = Reason(resume_id=r["resume_id"], text=expl.text, cites=expl.cites)
        assert verify(reason, f, m) == [], (r["resume_id"], expl.text)
