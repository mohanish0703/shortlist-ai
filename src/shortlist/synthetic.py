"""SYNTHETIC resumes with ground-truth fields, for building and evaluating ShortlistAI.

Each resume is generated for one job and aimed at a fit tier (good / borderline / poor) by
applying "misses" to an otherwise good candidate. The profile is rendered into one of five messy
formats, the ground truth (`ResumeFields`) records only what the text states, and the fit label
is re-derived from that visible information by `rubric_fit`. These are the generator's rules,
not a recruiter's judgment: the hand labels in data/labels/labels.csv are the independent check.

    python -m shortlist.synthetic            # writes data/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from shortlist.catalog import (
    CERTIFICATIONS,
    CITIES,
    EDUCATION_FORMS,
    FAR_CITIES,
    GENERIC_SKILLS,
    INSTITUTIONS,
    JOBS,
    LANGUAGE_FORMS,
    NAMES,
    NEARBY,
    REFERENCE_DATE,
    ROLES,
    SKILL_FORMS,
)
from shortlist.schema import JobRequirement, ResumeFields

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Months are integers (year * 12 + month - 1). A stint lasts end - start months.
REF = REFERENCE_DATE.year * 12 + REFERENCE_DATE.month - 1

SOFT_MISSES = (
    "adjacent_title",
    "years_short",
    "nearby_city",
    "salary_high",
    "notice_long",
    "cert_missing",
    "languages_unstated",
    "location_unstated",
)
HARD_FAILS = (
    "off_role",
    "far_city",
    "too_junior",
    "language_mismatch",
    "salary_way_high",
    "notice_way_long",
)
# A miss shows up in one field, which random omission must leave visible.
MISS_FIELD = {
    "nearby_city": "location",
    "far_city": "location",
    "salary_high": "salary",
    "salary_way_high": "salary",
    "notice_long": "notice",
    "notice_way_long": "notice",
    "language_mismatch": "languages",
}
# Chance that a resume leaves a field out.
OMIT_P = {
    "salary": 0.35,
    "notice": 0.30,
    "education": 0.20,
    "certifications": 0.15,
    "skills": 0.08,
    "languages": 0.06,
    "location": 0.05,
}
TEMPLATES = ("formal", "bullets", "prose", "hinglish", "form")
TEMPLATE_WEIGHTS = (3, 2, 2, 2, 2)

EDUCATION_BY_ROLE = {
    "housekeeping": ["Below 10th", "10th", "10th", "12th", "ITI"],
    "security": ["10th", "12th", "12th", "ITI", "Graduate"],
    "delivery": ["10th", "12th", "12th", "ITI", "Graduate"],
    "front_desk": ["12th", "Diploma", "Graduate", "Graduate"],
    "customer_support": ["12th", "Graduate", "Graduate", "Postgraduate"],
}
HOMETOWNS = {
    lang: [c for c, (state_lang, _) in CITIES.items() if state_lang == lang] for lang in NAMES
}
HOMETOWNS["Urdu"] = ["Hyderabad", "Lucknow"]

MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip
MONTH_STYLES = (
    lambda y, m: f"{MON[m]} {y}",
    lambda y, m: f"{m + 1:02d}/{y}",
    lambda y, m: f"{MONTH[m]} {y}",
    lambda y, m: f"{MON[m]}-{y % 100:02d}",
    lambda y, m: f"{m + 1}.{y}",
)
PRESENT = ("Present", "Till date", "Now", "Current")
PROFICIENCY = ("", "", " (speak)", " (read/write/speak)", " - fluent", " (basic)")
OBJECTIVES = (
    "To work sincerely in a reputed organisation and grow with it.",
    "Seeking a job where I can use my experience and work hard.",
    "To get a stable job and serve the company with honesty.",
)
HOBBIES = ("Cricket", "Listening music", "Reading newspaper", "Cooking", "Watching movies")


@dataclass
class Stint:
    title: str
    employer: str
    start: int
    end: int | None  # None = still working there

    @property
    def months(self) -> int:
        return (REF if self.end is None else self.end) - self.start


@dataclass
class Profile:
    """Everything about a candidate; `hidden` lists what their resume does not state."""

    name: str
    home_lang: str
    hometown: str
    city: str
    languages: list[str]
    role: str
    title_relation: str  # "same", "adjacent" (related role) or "off" (different role)
    history: list[Stint]  # most recent first
    skills: list[str]
    certifications: list[str]
    education: str
    institution: str
    salary: int
    notice: int
    misses: list[str]
    hidden: set[str] = field(default_factory=set)

    @property
    def months(self) -> int:
        return sum(s.months for s in self.history)


# ---------------------------------------------------------------- profile


def _applies(miss: str, job: JobRequirement) -> bool:
    if miss in ("years_short", "too_junior"):
        return job.min_years >= 1
    if miss == "cert_missing":
        return bool(job.required_certifications)
    return True


def _pick_misses(job: JobRequirement, tier: str, rng: random.Random) -> list[str]:
    if tier == "good":
        return []
    soft = [m for m in SOFT_MISSES if _applies(m, job)]
    if tier == "borderline":
        return [rng.choice(soft)]
    if rng.random() < 0.2:
        while True:
            pair = rng.sample(soft, 2)
            # A hidden location makes "nearby city" invisible, leaving one miss.
            if set(pair) != {"nearby_city", "location_unstated"}:
                return pair
    return [rng.choice([m for m in HARD_FAILS if _applies(m, job)])]


def _languages(
    job: JobRequirement, city: str, role: str, misses: list[str], rng: random.Random
) -> tuple[list[str], str]:
    local = CITIES[city][0]
    if "language_mismatch" in misses:
        banned = set(job.languages_required) | set(job.languages_any)
        if "Hindi" in banned:
            banned.add("Urdu")
        home = rng.choice([lang for lang in NAMES if lang not in banned])
        langs = [home]
        extra = [lang for lang in ("English", "Hindi", local) if lang not in banned | {home}]
        if extra and rng.random() < 0.5:
            langs.append(rng.choice(extra))
        return langs, home

    home = local if rng.random() < 0.65 else rng.choice([lang for lang in NAMES if lang != local])
    langs = [home]
    if home == "Urdu":
        langs.append("Hindi")
    if home != local and rng.random() < 0.5:
        langs.append(local)
    if rng.random() < 0.5:
        langs.append("Hindi")
    if rng.random() < (0.9 if role in ("front_desk", "customer_support") else 0.35):
        langs.append("English")
    langs += job.languages_required
    if job.languages_any and not set(langs) & set(job.languages_any):
        langs.append(rng.choice(job.languages_any))
    return list(dict.fromkeys(langs)), home


def _history(
    job: JobRequirement, spec: dict, relation: str, misses: list[str], rng: random.Random
) -> list[Stint]:
    min_m = round(job.min_years * 12)
    if "too_junior" in misses:
        months = 0
    elif "years_short" in misses:
        months = rng.randint(max(min_m - 9, 1), min_m - 3)
    else:
        months = rng.randint(min_m, min_m + 60)
        if min_m == 0 and rng.random() < 0.3:
            months = 0
    if months == 0:
        return []

    titles = spec["adjacent_titles"] if relation == "adjacent" else spec["titles"]
    k = 1 if months < 18 else rng.choice([1, 2, 2, 3])
    cuts = sorted(rng.sample(range(1, months), k - 1))
    parts = [b - a for a, b in zip([0, *cuts], [*cuts, months], strict=True)]
    end = REF if rng.random() < 0.6 else REF - rng.randint(1, 6)
    employers = rng.sample(spec["employers"], k)  # no employer twice in one history
    stints = []
    for length, employer in zip(parts, employers, strict=True):
        start = end - length
        stints.append(
            Stint(
                rng.choice(titles),
                employer,
                start,
                None if end == REF else end,
            )
        )
        end = start - rng.randint(0, 3)
    return stints


def make_profile(job: JobRequirement, tier: str, rng: random.Random) -> Profile:
    misses = _pick_misses(job, tier, rng)

    role, relation = job.role, "same"
    if "off_role" in misses:
        role, relation = rng.choice([r for r in ROLES if r != job.role]), "off"
    elif "adjacent_title" in misses:
        relation = "adjacent"
    spec = ROLES[role]

    city = job.location
    if "nearby_city" in misses:
        city = rng.choice(NEARBY[job.location])
    elif "far_city" in misses:
        city = rng.choice(FAR_CITIES)

    hidden = set()
    if "languages_unstated" in misses:
        hidden.add("languages")
    if "location_unstated" in misses:
        hidden.add("location")
    protected = {MISS_FIELD[m] for m in misses if m in MISS_FIELD}
    for fld, p in OMIT_P.items():
        if fld not in protected and rng.random() < p:
            hidden.add(fld)

    languages, home = _languages(job, city, role, misses, rng)
    hometown = city if CITIES[city][0] == home else rng.choice(HOMETOWNS[home])
    if "location" in hidden and hometown == city:
        # The school's town must not give away a location the resume leaves out.
        hometown = rng.choice([c for c in HOMETOWNS[home] if c != city] or [hometown])
    firsts, lasts = NAMES[home]

    role_skills = list(spec["skills"])
    if relation == "adjacent":
        skills = rng.sample(role_skills, rng.randint(1, 2))
    else:
        skills = rng.sample(role_skills, rng.randint(3, min(6, len(role_skills))))
    skills += rng.sample(list(GENERIC_SKILLS), rng.randint(0, 2))

    certs = rng.sample(spec["certifications"], rng.randint(0, min(2, len(spec["certifications"]))))
    if relation != "off":
        certs += job.required_certifications
    if "cert_missing" in misses:
        certs = [c for c in certs if c not in job.required_certifications]

    education = rng.choice(EDUCATION_BY_ROLE[role])
    town = CITIES[hometown][1][0]
    institution = rng.choice(INSTITUTIONS[education]).format(town=town)

    cap = job.max_salary_monthly_inr
    low = spec["salary"][0]
    if "salary_high" in misses:
        salary = cap * rng.uniform(1.05, 1.22)
    elif "salary_way_high" in misses:
        salary = cap * rng.uniform(1.35, 1.8)
    else:
        salary = rng.uniform(min(low, cap * 0.7), cap)
    salary = round(salary / 500) * 500

    max_n = job.max_notice_days
    if "notice_long" in misses:
        notice = rng.choice([n for n in (10, 14, 20, 30, 45, 60) if max_n < n <= 2 * max_n])
    elif "notice_way_long" in misses:
        notice = rng.choice([n for n in (45, 60, 90) if n > 2 * max_n])
    else:
        notice = rng.choice([n for n in (0, 0, 7, 15, 30) if n <= max_n])

    return Profile(
        name=f"{rng.choice(firsts)} {rng.choice(lasts)}",
        home_lang=home,
        hometown=hometown,
        city=city,
        languages=languages,
        role=role,
        title_relation=relation,
        history=_history(job, spec, relation, misses, rng),
        skills=skills,
        certifications=list(dict.fromkeys(certs)),
        education=education,
        institution=institution,
        salary=salary,
        notice=notice,
        misses=misses,
        hidden=hidden,
    )


# ---------------------------------------------------------------- surface forms


def _month(idx: int, style) -> str:
    y, m = divmod(idx, 12)
    return style(y, m)


def _span(s: Stint, style, rng: random.Random) -> str:
    end = rng.choice(PRESENT) if s.end is None else _month(s.end, style)
    return f"{_month(s.start, style)} {rng.choice(['-', 'to', '–'])} {end}"


def _years_text(months: int) -> str:
    return f"{round(months / 12, 1):g}"


def _skill_forms(p: Profile, rng: random.Random) -> list[str]:
    return [rng.choice(SKILL_FORMS[s]) for s in p.skills]


def _lang_forms(p: Profile, rng: random.Random, proficiency: bool = True) -> list[str]:
    return [
        rng.choice(LANGUAGE_FORMS[lang]) + (rng.choice(PROFICIENCY) if proficiency else "")
        for lang in p.languages
    ]


def _cert_forms(p: Profile, rng: random.Random) -> list[str]:
    return [rng.choice(CERTIFICATIONS[c]) for c in p.certifications]


def _salary(s: int, rng: random.Random) -> str:
    forms = [f"{s}", f"Rs. {s:,}/-", f"{s} per month", f"INR {s:,} pm"]
    if s % 1000 == 0:
        forms.append(f"{s // 1000}k")
    forms.append(f"{s * 12 / 100_000:g} LPA")
    return rng.choice(forms)


def _notice(n: int, rng: random.Random) -> str:
    if n == 0:
        return rng.choice(["Immediate joiner", "Can join immediately", "Available immediately"])
    named = {7: "1 week", 14: "2 weeks", 30: "1 month", 60: "2 months", 90: "3 months"}
    forms = [f"{n} days", f"Can join within {n} days"]
    if n in named:
        forms.append(f"{named[n]} notice")
    return rng.choice(forms)


def _contact(p: Profile, rng: random.Random) -> tuple[str, str]:
    phone = f"9{rng.randint(0, 9)}XXXXX{rng.randint(100, 999)}"
    email = f"{p.name.split()[0].lower()}{rng.randint(10, 99)}@example.com"
    return phone, email


def _join_and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _shows(p: Profile, fld: str) -> bool:
    if fld == "skills":
        return "skills" not in p.hidden and bool(p.skills)
    if fld == "certifications":
        return "certifications" not in p.hidden and bool(p.certifications)
    return fld not in p.hidden


# ---------------------------------------------------------------- templates


def render_formal(p: Profile, rng: random.Random) -> str:
    style = rng.choice(MONTH_STYLES)
    phone, email = _contact(p, rng)
    city = rng.choice(CITIES[p.city][1])
    out = [rng.choice(["CURRICULUM VITAE", "RESUME", "BIO-DATA"]), "", f"Name: {p.name}"]
    if _shows(p, "location"):
        out.append(f"Address: #{rng.randint(1, 250)}, {rng.randint(1, 12)}th Cross, {city}")
    out += [f"Mobile: {phone}    Email: {email}", "", "CAREER OBJECTIVE", rng.choice(OBJECTIVES)]
    out += ["", "WORK EXPERIENCE"]
    if p.history:
        out += [
            f"{i}. {s.title} - {s.employer} ({_span(s, style, rng)})"
            for i, s in enumerate(p.history, 1)
        ]
    else:
        out.append("Fresher")
    if _shows(p, "skills"):
        out += ["", "SKILLS", *(f"- {s}" for s in _skill_forms(p, rng))]
    if _shows(p, "languages"):
        out += ["", "LANGUAGES KNOWN: " + ", ".join(_lang_forms(p, rng))]
    if _shows(p, "education"):
        edu = rng.choice(EDUCATION_FORMS[p.education])
        out += ["", "EDUCATIONAL QUALIFICATION", f"{edu} - {p.institution}"]
    if _shows(p, "certifications"):
        out += ["", "CERTIFICATIONS", *(f"* {c}" for c in _cert_forms(p, rng))]
    out += [
        "",
        "PERSONAL DETAILS",
        f"Date of Birth: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/"
        f"{REFERENCE_DATE.year - rng.randint(19, 45)}",
        f"Marital Status: {rng.choice(['Married', 'Unmarried', 'Single'])}",
        f"Hobbies: {', '.join(rng.sample(HOBBIES, 2))}",
    ]
    if _shows(p, "salary"):
        out.append(f"Expected Salary: {_salary(p.salary, rng)}")
    if _shows(p, "notice"):
        out.append(f"Notice Period: {_notice(p.notice, rng)}")
    out += [
        "",
        "DECLARATION",
        "I hereby declare that the above information is true to the best of my knowledge.",
    ]
    if _shows(p, "location"):
        out.append(f"Place: {city}")
    return "\n".join(out)


def render_bullets(p: Profile, rng: random.Random) -> str:
    style = rng.choice(MONTH_STYLES)
    phone, email = _contact(p, rng)
    head = p.name.upper()
    if _shows(p, "location"):
        head += f" | {rng.choice(CITIES[p.city][1])}"
    out = [head, f"{phone} | {email}", ""]
    out.append(f"Total exp: {_years_text(p.months)} yrs" if p.history else "Fresher")
    out += [f"• {s.title} @ {s.employer}, {_span(s, style, rng)}" for s in p.history]
    if _shows(p, "skills"):
        out.append("Skills — " + ", ".join(_skill_forms(p, rng)))
    if _shows(p, "languages"):
        out.append("Lang: " + " / ".join(_lang_forms(p, rng, proficiency=False)))
    if _shows(p, "education"):
        out.append(f"Edu: {rng.choice(EDUCATION_FORMS[p.education])}, {p.institution}")
    if _shows(p, "certifications"):
        out.append("Certs: " + "; ".join(_cert_forms(p, rng)))
    tail = []
    if _shows(p, "salary"):
        tail.append(f"Expected CTC: {_salary(p.salary, rng)}")
    if _shows(p, "notice"):
        tail.append(f"Notice: {_notice(p.notice, rng)}")
    if tail:
        out.append(" | ".join(tail))
    return "\n".join(out)


def render_prose(p: Profile, rng: random.Random) -> str:
    style = rng.choice(MONTH_STYLES)
    phone, email = _contact(p, rng)
    s = [f"My name is {p.name}."]
    if _shows(p, "location"):
        s.append(f"I live in {rng.choice(CITIES[p.city][1])}.")
    for i, st in enumerate(p.history):
        lead = "Before that, I" if i else "I"
        if st.end is None:
            s.append(
                f"{lead} am working as {st.title} at {st.employer} since {_month(st.start, style)}."
            )
        else:
            s.append(
                f"{lead} worked as {st.title} at {st.employer} from {_month(st.start, style)} "
                f"to {_month(st.end, style)}."
            )
    if not p.history:
        s.append("I am a fresher looking for my first job.")
    if _shows(p, "skills"):
        s.append(f"I am good at {_join_and(_skill_forms(p, rng))}.")
    if _shows(p, "languages"):
        s.append(f"I can speak {_join_and(_lang_forms(p, rng, proficiency=False))}.")
    if _shows(p, "education"):
        edu = rng.choice(EDUCATION_FORMS[p.education])
        s.append(f"I have completed {edu} from {p.institution}.")
    if _shows(p, "certifications"):
        s.append(f"I also have {_join_and(_cert_forms(p, rng))}.")
    if _shows(p, "salary"):
        s.append(f"My salary expectation is {_salary(p.salary, rng)}.")
    if _shows(p, "notice"):
        s.append(f"Notice period: {_notice(p.notice, rng)}.")
    return f"To the Hiring Manager,\n\n{' '.join(s)}\n\nThank you.\n{p.name}\n{phone}, {email}"


def render_hinglish(p: Profile, rng: random.Random) -> str:
    p.hidden.add("institution")  # this format gives only the education level
    phone, _ = _contact(p, rng)
    out = [
        rng.choice(["Namaste Sir/Madam,", "Respected Sir,", "Hello sir"]),
        f"Mera naam {p.name} hai.",
    ]
    if _shows(p, "location"):
        out.append(f"Abhi {rng.choice(CITIES[p.city][1])} mein hoon.")
    if p.history:
        years, months = divmod(p.months, 12)
        dur = " ".join(
            part
            for part in (f"{years} saal" if years else "", f"{months} mahine" if months else "")
            if part
        )
        out.append(f"Mujhe total {dur} ka experience hai.")
        out += [f"{s.employer} mein {s.title} ka kaam kiya." for s in p.history]
    else:
        out.append("Abhi fresher hoon, kaam ki zaroorat hai.")
    if _shows(p, "skills"):
        out.append("Kaam aata hai: " + ", ".join(_skill_forms(p, rng)))
    if _shows(p, "languages"):
        out.append("Bhasha: " + ", ".join(_lang_forms(p, rng, proficiency=False)) + " aati hai.")
    if _shows(p, "education"):
        out.append(f"Padhai: {rng.choice(EDUCATION_FORMS[p.education])}.")
    if _shows(p, "certifications"):
        out.append("Certificate: " + ", ".join(_cert_forms(p, rng)) + " hai.")
    if _shows(p, "salary"):
        out.append(f"Salary {_salary(p.salary, rng)} chahiye.")
    if _shows(p, "notice"):
        if p.notice == 0:
            out.append("Turant join kar sakte hain.")
        else:
            out.append(f"{p.notice} din mein join kar sakte hain.")
    out += [f"Call/WhatsApp: {phone}", "Dhanyavaad"]
    return "\n".join(out)


def render_form(p: Profile, rng: random.Random) -> str:
    style = rng.choice(MONTH_STYLES)
    phone, email = _contact(p, rng)

    def row(key: str, value: str) -> str:
        return f"{key:<20}: {value}"

    out = ["APPLICATION FORM", "", row("Candidate Name", p.name)]
    if _shows(p, "location"):
        out.append(row("Current City", rng.choice(CITIES[p.city][1])))
    out.append(row("Contact", f"{phone} / {email}"))
    if p.history:
        out += [
            row("Total Experience", f"{_years_text(p.months)} years"),
            "",
            "| Organisation | Designation | From | To |",
        ]
        for s in p.history:
            end = rng.choice(PRESENT) if s.end is None else _month(s.end, style)
            out.append(f"| {s.employer} | {s.title} | {_month(s.start, style)} | {end} |")
        out.append("")
    else:
        out.append(row("Total Experience", "0 (Fresher)"))
    if _shows(p, "skills"):
        out.append(row("Key Skills", " / ".join(_skill_forms(p, rng))))
    if _shows(p, "languages"):
        out.append(row("Languages", ", ".join(_lang_forms(p, rng))))
    if _shows(p, "education"):
        out.append(
            row("Qualification", f"{rng.choice(EDUCATION_FORMS[p.education])} ({p.institution})")
        )
    if _shows(p, "certifications"):
        out.append(row("Certificates", ", ".join(_cert_forms(p, rng))))
    if _shows(p, "salary"):
        out.append(row("Expected Salary", _salary(p.salary, rng)))
    if _shows(p, "notice"):
        out.append(row("Notice Period", _notice(p.notice, rng)))
    return "\n".join(out)


RENDERERS = {
    "formal": render_formal,
    "bullets": render_bullets,
    "prose": render_prose,
    "hinglish": render_hinglish,
    "form": render_form,
}


# ---------------------------------------------------------------- noise


def add_typos(text: str, protect: set[str], rng: random.Random, rate: float = 0.05) -> str:
    """Swap, drop or double one letter in some longer words. Names are left alone."""

    def typo(m: re.Match) -> str:
        w = m.group(0)
        if w in protect or rng.random() >= rate:
            return w
        i = rng.randint(1, len(w) - 2)
        op = rng.choice(("swap", "drop", "double"))
        if op == "swap":
            return w[:i] + w[i + 1] + w[i] + w[i + 2 :]
        if op == "drop":
            return w[:i] + w[i + 1 :]
        return w[:i] + w[i] + w[i:]

    return re.sub(r"[A-Za-z]{5,}", typo, text)


def as_bad_pdf_extract(text: str, rng: random.Random) -> str:
    """Upper-case everything and run some lines together, like a poor PDF-to-text dump."""
    lines = text.upper().split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if line and out[-1] and rng.random() < 0.25:
            out[-1] += "   " + line
        else:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------- truth and labels


def ground_truth(p: Profile) -> ResumeFields:
    """What the rendered resume states. Call after rendering (templates may hide fields)."""
    return ResumeFields(
        name=p.name,
        job_titles=[s.title for s in p.history],
        skills=p.skills if _shows(p, "skills") else [],
        years_experience=round(p.months / 12, 1),
        location=p.city if _shows(p, "location") else None,
        languages=p.languages if _shows(p, "languages") else [],
        certifications=p.certifications if _shows(p, "certifications") else [],
        education=p.education if _shows(p, "education") else None,
        institution=p.institution if _shows(p, "education") and _shows(p, "institution") else None,
        expected_salary_monthly_inr=p.salary if _shows(p, "salary") else None,
        notice_period_days=p.notice if _shows(p, "notice") else None,
    )


def rubric_fit(f: ResumeFields, job: JobRequirement, title_relation: str) -> tuple[str, list[str]]:
    """The generator's labelling rule: any hard fail or two soft misses = poor,
    one soft miss = borderline, none = good. Unstated salary or notice is not held against
    the candidate; unstated location or languages is (a recruiter would have to call)."""
    hard: list[str] = []
    soft: list[str] = []

    if title_relation == "off":
        hard.append("experience is in a different role")
    elif title_relation == "adjacent":
        soft.append("related role, not the same one")

    years = f.years_experience or 0.0
    if years < job.min_years:
        short = f"{years:g} years vs {job.min_years:g} required"
        (soft if years > 0 and job.min_years - years < 1 else hard).append(short)

    if f.location is None:
        soft.append("location not stated")
    elif f.location in NEARBY[job.location]:
        soft.append(f"lives in {f.location}, near {job.location}")
    elif f.location != job.location:
        hard.append(f"lives in {f.location}")

    if not f.languages:
        soft.append("languages not stated")
    elif any(lang not in f.languages for lang in job.languages_required) or (
        job.languages_any and not set(job.languages_any) & set(f.languages)
    ):
        hard.append("does not list the required languages")

    for cert in job.required_certifications:
        if cert not in f.certifications:
            soft.append(f"{cert} not listed")

    cap = job.max_salary_monthly_inr
    if f.expected_salary_monthly_inr is not None and f.expected_salary_monthly_inr > cap:
        over = f"expects {f.expected_salary_monthly_inr} vs budget {cap}"
        (hard if f.expected_salary_monthly_inr > 1.25 * cap else soft).append(over)

    max_n = job.max_notice_days
    if f.notice_period_days is not None and f.notice_period_days > max_n:
        late = f"{f.notice_period_days} days notice vs {max_n} allowed"
        (hard if f.notice_period_days > 2 * max_n else soft).append(late)

    label = "poor" if hard or len(soft) >= 2 else "borderline" if soft else "good"
    return label, hard + soft


def generate(per_job: int = 30, seed: int = 42) -> list[dict]:
    """Return one record per resume: id, job, text, ground truth, generator label, metadata."""
    rng = random.Random(seed)
    plan = [
        (job, tier)
        for job in JOBS
        for tier in ("good", "borderline", "poor")
        for _ in range(per_job // 3)
    ]
    rng.shuffle(plan)  # so resume ids say nothing about job or tier

    records = []
    for i, (job, tier) in enumerate(plan, 1):
        p = make_profile(job, tier, rng)
        template = rng.choices(TEMPLATES, weights=TEMPLATE_WEIGHTS)[0]
        text = RENDERERS[template](p, rng)
        typos = rng.random() < 0.35
        if typos:
            # Names stay exact; "example" keeps emails on the reserved example.com domain.
            text = add_typos(text, {*p.name.split(), "example"}, rng)
        bad_pdf = template in ("formal", "form") and rng.random() < 0.3
        if bad_pdf:
            text = as_bad_pdf_extract(text, rng)
        truth = ground_truth(p)
        fit, reasons = rubric_fit(truth, job, p.title_relation)
        records.append(
            {
                "resume_id": f"R{i:03d}",
                "job_id": job.job_id,
                "text": text,
                "fields": truth.model_dump(),
                "generator_fit": fit,
                "fit_reasons": reasons,
                "meta": {
                    "target_tier": tier,
                    "misses": p.misses,
                    "title_relation": p.title_relation,
                    "written_role": p.role,
                    "city": p.city,
                    "template": template,
                    "typos": typos,
                    "bad_pdf_extract": bad_pdf,
                },
            }
        )
    return records


def sample_for_labelling(records: list[dict], per_job: int = 10, seed: int = 7) -> list[dict]:
    """Pick resumes for hand labelling: per job, spread evenly over the generator's labels."""
    rng = random.Random(seed)
    picked = []
    for job in JOBS:
        by_fit: dict[str, list[dict]] = {}
        for r in records:
            if r["job_id"] == job.job_id:
                by_fit.setdefault(r["generator_fit"], []).append(r)
        for group in by_fit.values():
            rng.shuffle(group)
        chosen: list[dict] = []
        while len(chosen) < per_job and any(by_fit.values()):
            for fit in ("good", "borderline", "poor"):
                if by_fit.get(fit) and len(chosen) < per_job:
                    chosen.append(by_fit[fit].pop())
        picked += sorted(chosen, key=lambda r: r["resume_id"])
    return picked


def write(records: list[dict], out: Path = DATA) -> None:
    resumes = out / "resumes"
    resumes.mkdir(parents=True, exist_ok=True)
    for old in resumes.glob("R*.txt"):
        old.unlink()
    for r in records:
        (resumes / f"{r['resume_id']}.txt").write_text(r["text"] + "\n", encoding="utf-8")

    with (out / "ground_truth.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(
                json.dumps({k: v for k, v in r.items() if k != "text"}, ensure_ascii=False) + "\n"
            )

    (out / "jobs.json").write_text(
        json.dumps([j.model_dump() for j in JOBS], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "note": "SYNTHETIC data generated by shortlist.synthetic",
        "resumes": len(records),
        "generator_fit_by_job": {
            job.job_id: dict(
                Counter(r["generator_fit"] for r in records if r["job_id"] == job.job_id)
            )
            for job in JOBS
        },
        "templates": dict(Counter(r["meta"]["template"] for r in records)),
        "with_typos": sum(r["meta"]["typos"] for r in records),
        "bad_pdf_extract": sum(r["meta"]["bad_pdf_extract"] for r in records),
        "field_missing": {
            fld: sum(r["fields"][fld] in (None, []) for r in records)
            for fld in ResumeFields.model_fields
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    labels = out / "labels" / "labels.csv"
    if labels.exists():
        print(f"kept existing {labels.relative_to(ROOT)} (delete it to resample)")
        return
    labels.parent.mkdir(parents=True, exist_ok=True)
    with labels.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["resume_id", "job_id", "label", "notes"])
        for r in sample_for_labelling(records):
            w.writerow([r["resume_id"], r["job_id"], "", ""])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-job", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    records = generate(args.per_job, args.seed)
    write(records)
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
