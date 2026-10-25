"""Ranking and explanation node.

Scoring is plain code over the matcher's checks:
- tier: "shortlist" when nothing fails, nothing is a near miss and role, experience, location
  and languages are all known; "call" when nothing fails but a near miss or a gap needs a phone
  call; "reject" when any check fails
- score: 100, minus 15 per near miss, minus 5 per unknown, minus 40 per fail, plus 5 per
  preferred skill (at most 3) and 2 per year past the minimum (at most 10)
Candidates sort by tier, then score, then resume id.

Reasons come from the LLM, one call per job for all its candidates. Each reason must cite the
fields it uses with their values. `verify` rejects a reason that cites nothing, cites a value
the candidate's fields do not hold, states a number found in neither its citations nor the
checks, or uses a gendered pronoun (resumes do not state gender). Rejected reasons are
retried once, then replaced by a plain one built from the checks.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from google import genai
from pydantic import BaseModel, ConfigDict, Field

from shortlist import llm
from shortlist.match import Match
from shortlist.schema import FitLabel, JobRequirement, ResumeFields

Tier = Literal["shortlist", "call", "reject"]
FieldName = Literal[tuple(ResumeFields.model_fields)]

TIER_ORDER = {"shortlist": 0, "call": 1, "reject": 2}
TIER_FIT: dict[str, FitLabel] = {"shortlist": "good", "call": "borderline", "reject": "poor"}
# An unstated salary or notice period is common and cheap to ask about; these four are not.
MUST_KNOW = {"role", "experience", "location", "languages"}
PENALTY = {"pass": 0, "unknown": 5, "soft": 15, "fail": 40}


class Ranked(BaseModel):
    resume_id: str
    tier: Tier
    score: float
    fit: FitLabel  # the tier as a fit label, for comparing with labels


def score(resume_id: str, f: ResumeFields, m: Match, job: JobRequirement) -> Ranked:
    statuses = {c.name: c.status for c in m.checks}
    if "fail" in statuses.values():
        tier: Tier = "reject"
    elif "soft" in statuses.values() or any(statuses[n] == "unknown" for n in MUST_KNOW):
        tier = "call"
    else:
        tier = "shortlist"
    extra_years = max(0.0, (f.years_experience or 0.0) - job.min_years)
    points = (
        100
        - sum(PENALTY[s] for s in statuses.values())
        + 5 * min(3, len(m.preferred_skills))
        + min(10.0, 2 * extra_years)
    )
    return Ranked(resume_id=resume_id, tier=tier, score=round(points, 1), fit=TIER_FIT[tier])


def rank(scored: list[Ranked]) -> list[Ranked]:
    return sorted(scored, key=lambda r: (TIER_ORDER[r.tier], -r.score, r.resume_id))


# ---------------------------------------------------------------- reasons


class Cite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: FieldName
    value: str = Field(description='The value exactly as in the fields JSON, or "not stated"')


class Reason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resume_id: str
    text: str = Field(description="One or two sentences, at most 40 words")
    cites: list[Cite]


class Reasons(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasons: list[Reason]


class Explanation(BaseModel):
    text: str
    cites: list[Cite]
    source: Literal["llm", "llm_retry", "template"]
    # what `verify` found wrong with the LLM's attempts, if anything
    rejected: list[str] = Field(default_factory=list)


REASON_SYSTEM = """You write short reasons for a recruiter's shortlist.
For each candidate you get the decision (tier and score), the checks behind it and the
candidate's extracted fields. Write one or two plain sentences, at most 40 words: the main
reason for the decision first, then the main concern if there is one.
Use only facts from the fields and the checks. Cite every field you use, with its value copied
exactly from the fields JSON (one list item per citation), or "not stated" if it is empty.
Every number you write must come from a cited value or from the checks.
Refer to the candidate by name or as "the candidate"; never guess gender (no he/she/his/her)."""

REASON_PROMPT_ID = hashlib.sha256(
    (REASON_SYSTEM + json.dumps(Reasons.model_json_schema(), sort_keys=True)).encode()
).hexdigest()[:8]

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# The resume does not state gender; a reason must not guess it from the name.
GENDERED_RE = re.compile(r"\b(he|she|him|his|her|hers|himself|herself)\b", re.IGNORECASE)


def _norm(value) -> str:
    """Comparable text: case, spacing and thousands commas ignored; 1.0, "1.0" and 1 alike."""
    text = " ".join(str(value).replace(",", "").casefold().split())
    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def _items(value: str) -> list[str]:
    """A citation of a list field may be one item or the whole list as JSON."""
    try:
        parsed = json.loads(value)
    except ValueError:
        return [value]
    return [str(v) for v in parsed] if isinstance(parsed, list) and parsed else [value]


def _numbers(text: str) -> set[str]:
    return {_norm(n).rstrip(".") for n in NUMBER_RE.findall(text)}


def verify(reason: Reason, f: ResumeFields, m: Match) -> list[str]:
    """Problems with a reason; empty means it is grounded in the fields and checks."""
    problems = []
    if not reason.cites:
        problems.append("cites nothing")
    values = f.model_dump()
    for c in reason.cites:
        held = values[c.field]
        if held in (None, []):
            ok = _norm(c.value) == "not stated"
        elif isinstance(held, list):
            ok = {_norm(v) for v in _items(c.value)} <= {_norm(v) for v in held}
        else:
            ok = _norm(c.value) == _norm(held)
        if not ok:
            problems.append(f"{c.field} is {held!r}, not {c.value!r}")
    grounded = set()
    for source in [*(c.value for c in reason.cites), *(c.detail for c in m.checks)]:
        grounded |= _numbers(source)
    for n in sorted(_numbers(reason.text) - grounded):
        problems.append(f"number {n} is not in the citations or checks")
    if GENDERED_RE.search(reason.text):
        problems.append("guesses gender")
    return problems


def template_reason(r: Ranked, f: ResumeFields, m: Match) -> Explanation:
    """A plain reason from the checks: the fallback when the LLM's reasons fail `verify`."""
    misses = [c for c in m.checks if c.status in ("fail", "soft", "unknown")]
    lead = {"shortlist": "Meets every requirement", "call": "Worth a call", "reject": "Rejected"}
    parts = [c.detail for c in misses if c.status != "unknown"]
    parts += [c.detail for c in misses if c.status == "unknown" and c.name in MUST_KNOW]
    if not parts:
        parts = [c.detail for c in m.checks if c.name in ("role", "experience")]
    values = f.model_dump()
    cites = [
        cite
        for c in misses or m.checks[:2]
        for name in c.fields
        for cite in _cites(name, values[name])
    ]
    text = f"{lead[r.tier]}: {'; '.join(parts[:3])}."
    return Explanation(text=text, cites=cites, source="template")


def _cites(name: str, held) -> list[Cite]:
    if held in (None, []):
        return [Cite(field=name, value="not stated")]
    items = held if isinstance(held, list) else [held]
    return [Cite(field=name, value=f"{v:g}" if isinstance(v, float) else str(v)) for v in items]


def _candidate_block(r: Ranked, f: ResumeFields, m: Match) -> str:
    checks = "; ".join(f"{c.name} {c.status}: {c.detail}" for c in m.checks)
    return (
        f"### {r.resume_id}\ntier: {r.tier}, score: {r.score:g}\nchecks: {checks}\n"
        f"preferred skills: {', '.join(m.preferred_skills) or 'none'}\n"
        f"fields: {json.dumps(f.model_dump(), ensure_ascii=False)}"
    )


def _ask(job, items, client, model) -> dict[str, Reason]:
    prompt = f"Job: {job.title}\n{job.description}\n\n" + "\n\n".join(
        _candidate_block(r, f, m) for r, f, m in items
    )
    result = llm.generate_structured(client, REASON_SYSTEM, prompt, Reasons, model)
    return {x.resume_id: x for x in result.reasons}


def explain(
    job: JobRequirement,
    items: list[tuple[Ranked, ResumeFields, Match]],
    client: genai.Client | None = None,
    model: str | None = None,
) -> dict[str, Explanation]:
    """One LLM call for all candidates, one retry for the rejected ones, then the template."""
    client = client or llm.make_client()
    out: dict[str, Explanation] = {}
    problems: dict[str, list[str]] = {}
    todo = items
    for source in ("llm", "llm_retry"):
        if not todo:
            break
        reasons = _ask(job, todo, client, model)
        failed = []
        for r, f, m in todo:
            reason = reasons.get(r.resume_id)
            found = ["no reason returned"] if reason is None else verify(reason, f, m)
            if found:
                problems.setdefault(r.resume_id, []).extend(found)
                failed.append((r, f, m))
            else:
                out[r.resume_id] = Explanation(
                    text=reason.text,
                    cites=reason.cites,
                    source=source,
                    rejected=problems.get(r.resume_id, []),
                )
        todo = failed
    for r, f, m in todo:
        fallback = template_reason(r, f, m)
        fallback.rejected = problems[r.resume_id]
        out[r.resume_id] = fallback
    return out
