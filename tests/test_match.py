"""The plain-code checks of the matching node; the title calls are given, no API calls."""

from __future__ import annotations

import pytest

from shortlist.catalog import JOBS
from shortlist.match import TitleCall, match, title_key
from shortlist.schema import ResumeFields

JOB = {j.job_id: j for j in JOBS}
CALLS = {
    title_key(t): TitleCall(title=t, relation=rel, reason="test")
    for t, rel in [
        ("Housekeeping Staff", "same"),
        ("Office Boy", "related"),
        ("Receptionist", "different"),
    ]
}


def _good(**kw) -> ResumeFields:
    """Meets every requirement of J1 (housekeeping, Bengaluru)."""
    base = dict(
        name="Asha Rao",
        job_titles=["Housekeeping Staff"],
        skills=["room cleaning", "teamwork"],
        years_experience=2.0,
        location="Bengaluru",
        languages=["Kannada"],
        expected_salary_monthly_inr=16_000,
        notice_period_days=0,
    )
    return ResumeFields(**(base | kw))


def _status(f: ResumeFields, job_id: str = "J1") -> dict[str, str]:
    return {c.name: c.status for c in match(f, JOB[job_id], CALLS).checks}


def test_good_candidate_passes_everything():
    m = match(_good(), JOB["J1"], CALLS)
    assert {c.status for c in m.checks} == {"pass"}
    assert m.preferred_skills == ["room cleaning"]


@pytest.mark.parametrize(
    ("change", "check", "status"),
    [
        ({"job_titles": ["Office Boy"]}, "role", "soft"),
        ({"job_titles": ["Receptionist"]}, "role", "fail"),
        ({"job_titles": ["Office Boy", "Housekeeping Staff"]}, "role", "pass"),  # best one counts
        ({"job_titles": ["Astronaut"]}, "role", "unknown"),  # not classified
        ({"years_experience": 0.5}, "experience", "soft"),  # J1 allows half a year short
        ({"years_experience": 0.4}, "experience", "fail"),
        ({"location": "Mysuru"}, "location", "soft"),
        ({"location": "Patna"}, "location", "fail"),
        ({"location": None}, "location", "unknown"),
        ({"languages": ["Tamil"]}, "languages", "fail"),
        ({"languages": []}, "languages", "unknown"),
        ({"expected_salary_monthly_inr": 19_800}, "salary", "soft"),  # 10% over 18k budget
        ({"expected_salary_monthly_inr": 19_801}, "salary", "fail"),
        ({"expected_salary_monthly_inr": None}, "salary", "unknown"),
        ({"notice_period_days": 30}, "notice", "soft"),  # 15 days + 15 days of slack
        ({"notice_period_days": 31}, "notice", "fail"),
    ],
)
def test_one_miss(change, check, status):
    result = _status(_good(**change))
    assert result[check] == status
    assert all(s == "pass" for name, s in result.items() if name != check)


def test_languages_required_and_any():
    # J5 needs both English and Hindi
    assert _status(_good(languages=["Hindi"]), "J5")["languages"] == "fail"
    assert _status(_good(languages=["Hindi", "English"]), "J5")["languages"] == "pass"


def test_fresher_and_licence_for_delivery():
    fresher = _good(job_titles=[], years_experience=0.0, location="Chennai", languages=["Tamil"])
    result = _status(fresher, "J4")
    assert result["role"] == "pass"  # J4 takes freshers
    assert result["experience"] == "pass"
    assert result["certifications"] == "soft"  # licence not listed: worth a call
    licensed = fresher.model_copy(update={"certifications": ["Two-Wheeler Driving Licence"]})
    assert _status(licensed, "J4")["certifications"] == "pass"


def test_checks_name_the_fields_they_read():
    fields = set(ResumeFields.model_fields)
    for c in match(_good(), JOB["J1"], CALLS).checks:
        assert c.fields
        assert set(c.fields) <= fields


def test_slack_is_set_per_job():
    # J3 wants guards within a week and allows one more week, not the default 15 days
    j3 = {"location": "Hyderabad", "languages": ["Telugu"], "job_titles": []}
    assert _status(_good(**j3, notice_period_days=14), "J3")["notice"] == "soft"
    assert _status(_good(**j3, notice_period_days=15), "J3")["notice"] == "fail"
