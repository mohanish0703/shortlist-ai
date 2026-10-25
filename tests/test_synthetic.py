from __future__ import annotations

import re
from collections import Counter

import pytest

from shortlist.catalog import CITIES, JOBS, SKILL_VOCAB
from shortlist.schema import ResumeFields
from shortlist.synthetic import generate, rubric_fit, sample_for_labelling

JOB = {j.job_id: j for j in JOBS}


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return generate(per_job=30, seed=42)


def test_same_seed_same_data(records):
    assert generate(per_job=30, seed=42) == records


def test_ground_truth_validates(records):
    for r in records:
        f = ResumeFields.model_validate(r["fields"])
        assert set(f.skills) <= set(SKILL_VOCAB)


def test_thirty_resumes_per_job(records):
    assert Counter(r["job_id"] for r in records) == {j.job_id: 30 for j in JOBS}


def test_every_job_has_every_label(records):
    for job in JOBS:
        fits = {r["generator_fit"] for r in records if r["job_id"] == job.job_id}
        assert fits == {"good", "borderline", "poor"}, job.job_id


def test_stated_fields_are_in_the_text(records):
    """Ground truth may only contain what the resume actually says."""
    for r in records:
        f, text = r["fields"], r["text"].lower()
        if r["meta"]["typos"]:
            continue  # a typo can break the literal match
        assert f["name"].lower() in text
        for title in f["job_titles"]:
            assert title.lower() in text
        if f["location"]:
            assert any(form.lower() in text for form in CITIES[f["location"]][1])
        else:
            assert not any(form.lower() in text for form in CITIES[r["meta"]["city"]][1])
        if f["expected_salary_monthly_inr"] is None:
            assert not re.search(r"salary|ctc", text)


def test_no_real_contact_details(records):
    for r in records:
        assert not re.search(r"\b\d{10}\b", r["text"])
        for email in re.findall(r"\S+@\S+", r["text"]):
            assert email.lower().endswith("@example.com")


def test_rubric_matches_recorded_label(records):
    for r in records:
        fit, _ = rubric_fit(
            ResumeFields.model_validate(r["fields"]), JOB[r["job_id"]], r["meta"]["title_relation"]
        )
        assert fit == r["generator_fit"]


def test_rubric_rules():
    job = JOB["J1"]  # Bengaluru, 1+ yr, Kannada or Hindi, budget 18000, 15 days
    good = ResumeFields(
        years_experience=2,
        location="Bengaluru",
        languages=["Kannada"],
        expected_salary_monthly_inr=16000,
        notice_period_days=0,
    )
    assert rubric_fit(good, job, "same")[0] == "good"
    near = good.model_copy(update={"location": "Mysuru"})
    assert rubric_fit(near, job, "same")[0] == "borderline"
    assert rubric_fit(near, job, "adjacent")[0] == "poor"
    assert rubric_fit(good.model_copy(update={"location": "Patna"}), job, "same")[0] == "poor"
    assert rubric_fit(good.model_copy(update={"languages": ["Tamil"]}), job, "same")[0] == "poor"
    assert (
        rubric_fit(good.model_copy(update={"years_experience": 0.5}), job, "same")[0]
        == "borderline"
    )
    assert rubric_fit(good.model_copy(update={"years_experience": 0}), job, "same")[0] == "poor"
    unstated = good.model_copy(
        update={"expected_salary_monthly_inr": None, "notice_period_days": None}
    )
    assert rubric_fit(unstated, job, "same")[0] == "good"


def test_labelling_sample(records):
    picked = sample_for_labelling(records)
    assert len(picked) == 50
    assert len({r["resume_id"] for r in picked}) == 50
    for job in JOBS:
        fits = Counter(r["generator_fit"] for r in picked if r["job_id"] == job.job_id)
        assert sum(fits.values()) == 10
        assert min(fits.values()) >= 2, (job.job_id, fits)
