"""The deterministic half of the parsing node and the scoring; no API calls."""

from __future__ import annotations

from shortlist.eval_parse import compare, list_counts
from shortlist.parse import Extraction, Job, to_fields, years_experience


def _ex(**kw) -> Extraction:
    return Extraction(name="Asha Rao", **kw)


def test_years_sum_months_of_every_job():
    ex = _ex(
        jobs=[
            Job(title="A", start="2024-08", end="present"),
            Job(title="B", start="2022-10", end="2024-07"),
        ]
    )
    # Aug 2024 -> Oct 2026 is 26 months, Oct 2022 -> Jul 2024 is 21: 47 months
    assert years_experience(ex) == 3.9


def test_years_fall_back_to_stated_total_when_a_date_is_missing():
    ex = _ex(jobs=[Job(title="A", start=None, end="present")], stated_total_years=2.5)
    assert years_experience(ex) == 2.5


def test_years_ignore_bad_dates():
    ex = _ex(jobs=[Job(title="A", start="2024-13", end="2025-01")], stated_total_years=None)
    assert years_experience(ex) is None
    backwards = _ex(jobs=[Job(title="A", start="2025-01", end="2024-01")], stated_total_years=1.0)
    assert years_experience(backwards) == 1.0


def test_fresher_is_zero_only_when_stated():
    assert years_experience(_ex(stated_total_years=0)) == 0.0
    assert years_experience(_ex()) is None


def test_city_spellings_map_to_canonical():
    assert to_fields(_ex(location="Madras")).location == "Chennai"
    assert to_fields(_ex(location="whitefield, bangalore")).location == "Bengaluru"
    assert to_fields(_ex(location="Nagpur")).location == "Nagpur"  # outside the catalog: kept


def _fields(**kw) -> dict:
    base = {
        "name": "Asha Rao",
        "job_titles": ["Security Guard"],
        "skills": ["patrolling"],
        "years_experience": 2.2,
        "location": "Hyderabad",
        "languages": ["Telugu"],
        "certifications": [],
        "education": "10th",
        "institution": "Sri Saraswati Vidya Mandir, Hyderabad",
        "expected_salary_monthly_inr": None,
        "notice_period_days": 0,
    }
    return base | kw


def test_compare_ignores_case_spacing_and_punctuation():
    pred = _fields(
        name="ASHA  RAO",
        job_titles=["security guard"],
        institution="Sri Saraswati Vidya Mandir Hyderabad",
    )
    assert set(compare(pred, _fields()).values()) == {"ok"}


def test_compare_tells_missed_invented_and_wrong_apart():
    pred = _fields(location=None, expected_salary_monthly_inr=20000, education="12th")
    result = compare(pred, _fields())
    assert result["location"] == "missed"
    assert result["expected_salary_monthly_inr"] == "invented"
    assert result["education"] == "wrong"
    assert result["notice_period_days"] == "ok"  # 0 is a value, not "not stated"


def test_list_counts():
    pred = _fields(skills=["patrolling", "CCTV monitoring"])
    true = _fields(skills=["Patrolling", "access control"])
    assert list_counts(pred, true, "skills") == (1, 2, 2)
