"""The ShortlistAI graph: parse -> match -> rank -> recruiter review.

    START -> parse -+-> parse            (a resume failed and has attempts left)
                    +-> manual_review    (a resume failed MAX_PARSE_ATTEMPTS times) -> match
                    +-> match -> rank -> review -> END

- parse retries a resume whose parse failed (invalid output or an API error) and, after
  MAX_PARSE_ATTEMPTS, puts it in the manual-review queue instead of dropping it silently.
- review pauses the graph with `interrupt` until the recruiter approves, edits or rejects the
  proposed shortlist. Every decision is appended to the feedback log.
- With a SQLite checkpointer the state is saved after every step, so a paused session survives
  a restart and is resumed by its thread id.

The LLM-backed steps come in through `Services`, the run's context, which is not checkpointed:
tests pass fakes, demos pass `cached_services` (evaluation caches first, then the API).
State values are plain dicts so the checkpointer can store them.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Literal, TypedDict

from google.genai import errors
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from shortlist import llm
from shortlist.match import TITLE_PROMPT_ID, Match, TitleCall, classify_titles, match, title_key
from shortlist.parse import PROMPT_ID, Extraction, parse_resume, to_fields
from shortlist.rank import REASON_PROMPT_ID, Explanation, Ranked, explain, rank, score
from shortlist.schema import JobRequirement, ResumeFields
from shortlist.synthetic import DATA, ROOT

MAX_PARSE_ATTEMPTS = 2
RUNS = ROOT / "runs"  # git-ignored: checkpoints and the feedback log
RESULTS = ROOT / "results"

Items = list[tuple[Ranked, ResumeFields, Match]]


@dataclass
class Services:
    parse: Callable[[str, str], ResumeFields]  # (resume id, text) -> fields
    classify_titles: Callable[[JobRequirement, list[str]], dict[str, TitleCall]]
    explain: Callable[[JobRequirement, Items], dict[str, Explanation]]
    feedback_log: Path = RUNS / "feedback.jsonl"


class State(TypedDict, total=False):
    job: dict
    resumes: dict[str, str]  # resume id -> text
    parsed: dict[str, dict]  # resume id -> ResumeFields
    attempts: dict[str, int]
    errors: dict[str, str]  # last parse error per resume
    manual_review: list[str]  # resumes a person has to read
    titles: dict[str, dict]  # title key -> TitleCall
    matches: dict[str, dict]  # resume id -> Match
    ranking: list[dict]  # Ranked, best first
    reasons: dict[str, dict]  # resume id -> Explanation
    proposed: list[str]  # the shortlist put to the recruiter
    decision: dict  # the recruiter's Decision
    final: list[str]  # the shortlist after review


class Decision(BaseModel):
    action: Literal["approve", "edit", "reject"]
    remove: list[str] = Field(default_factory=list)
    add: list[str] = Field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------- nodes


def parse_node(state: State, runtime: Runtime[Services]) -> State:
    parsed = dict(state.get("parsed", {}))
    attempts = dict(state.get("attempts", {}))
    errs = dict(state.get("errors", {}))
    queued = set(state.get("manual_review", []))
    for rid, text in state["resumes"].items():
        if rid in parsed or rid in queued:
            continue
        attempts[rid] = attempts.get(rid, 0) + 1
        try:
            parsed[rid] = runtime.context.parse(rid, text).model_dump()
            errs.pop(rid, None)
        except (ValidationError, errors.APIError, ValueError) as e:
            errs[rid] = f"{type(e).__name__}: {e}"[:300]
    return {"parsed": parsed, "attempts": attempts, "errors": errs}


def _failed(state: State) -> list[str]:
    queued = set(state.get("manual_review", []))
    return [rid for rid in state.get("errors", {}) if rid not in queued]


def after_parse(state: State) -> str:
    failed = _failed(state)
    if any(state["attempts"][rid] < MAX_PARSE_ATTEMPTS for rid in failed):
        return "parse"
    return "manual_review" if failed else "match"


def manual_review_node(state: State) -> State:
    return {"manual_review": sorted({*state.get("manual_review", []), *_failed(state)})}


def match_node(state: State, runtime: Runtime[Services]) -> State:
    job = JobRequirement.model_validate(state["job"])
    fields = {rid: ResumeFields.model_validate(d) for rid, d in state["parsed"].items()}
    titles = sorted({t for f in fields.values() for t in f.job_titles})
    calls = runtime.context.classify_titles(job, titles)
    return {
        "titles": {k: c.model_dump() for k, c in calls.items()},
        "matches": {rid: match(f, job, calls).model_dump() for rid, f in fields.items()},
    }


def rank_node(state: State, runtime: Runtime[Services]) -> State:
    job = JobRequirement.model_validate(state["job"])
    fields = {rid: ResumeFields.model_validate(d) for rid, d in state["parsed"].items()}
    matches = {rid: Match.model_validate(d) for rid, d in state["matches"].items()}
    ranking = rank([score(rid, fields[rid], matches[rid], job) for rid in fields])
    items = [(r, fields[r.resume_id], matches[r.resume_id]) for r in ranking]
    reasons = runtime.context.explain(job, items) if items else {}
    proposed = [r.resume_id for r in ranking if r.tier != "reject"][: job.openings]
    return {
        "ranking": [r.model_dump() for r in ranking],
        "reasons": {rid: e.model_dump() for rid, e in reasons.items()},
        "proposed": proposed,
    }


def check_decision(state: State, answer) -> tuple[Decision | None, str | None]:
    """The decision, or why it cannot be applied to this session."""
    try:
        decision = Decision.model_validate(answer)
    except ValidationError as e:
        return None, str(e)
    unknown = [rid for rid in decision.add if rid not in state["resumes"]]
    if unknown:
        return None, f"not in this session: {', '.join(unknown)}"
    return decision, None


def review_node(state: State, config: RunnableConfig, runtime: Runtime[Services]) -> State:
    # Pauses here; the value passed to Command(resume=...) comes back from `interrupt`. An
    # invalid answer asks again (raising would make LangGraph replay the same bad answer).
    question = {
        "job_id": state["job"]["job_id"],
        "proposed": state["proposed"],
        "manual_review": state.get("manual_review", []),
    }
    while True:
        decision, problem = check_decision(state, interrupt(question))
        if decision is not None:
            break
        question = {**question, "error": problem}
    if decision.action == "reject":
        final: list[str] = []
    else:
        final = [rid for rid in state["proposed"] if rid not in decision.remove]
        final += [rid for rid in decision.add if rid not in final]
    _log_feedback(state, decision, final, config, runtime.context.feedback_log)
    return {"decision": decision.model_dump(), "final": final}


def _log_feedback(state: State, d: Decision, final: list[str], config, path: Path) -> None:
    """One line per decision: what was proposed, what the recruiter changed and why."""
    ranked = {r["resume_id"]: r for r in state["ranking"]}
    entry = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "thread_id": config["configurable"].get("thread_id"),
        "job_id": state["job"]["job_id"],
        "action": d.action,
        "proposed": state["proposed"],
        "final": final,
        "removed": [ranked[r] for r in d.remove if r in ranked],
        "added": [ranked.get(r, {"resume_id": r, "tier": "unparsed"}) for r in d.add],
        "note": d.note,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_graph(checkpointer=None):
    g = StateGraph(State, context_schema=Services)
    g.add_node("parse", parse_node)
    g.add_node("manual_review", manual_review_node)
    g.add_node("match", match_node)
    g.add_node("rank", rank_node)
    g.add_node("review", review_node)
    g.add_edge(START, "parse")
    g.add_conditional_edges("parse", after_parse, ["parse", "manual_review", "match"])
    g.add_edge("manual_review", "match")
    g.add_edge("match", "rank")
    g.add_edge("rank", "review")
    g.add_edge("review", END)
    return g.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path: Path = RUNS / "checkpoints.sqlite") -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


# ---------------------------------------------------------------- services


def live_services(model: str | None = None) -> Services:
    # The client is made on first use, so cached runs work without an API key.
    client = cache(llm.make_client)
    return Services(
        parse=lambda rid, text: parse_resume(text, client(), model),
        classify_titles=lambda job, titles: classify_titles(job, titles, client(), model),
        explain=lambda job, items: explain(job, items, client(), model),
    )


def cached_services(model: str | None = None) -> Services:
    """Live services that first look in results/: parses of the synthetic resumes (only when
    the text is unchanged), title calls per job, and reasons per job and input."""
    model = model or llm.model_name()
    live = live_services(model)

    def parse(rid: str, text: str) -> ResumeFields:
        cached = RESULTS / "parses" / model / PROMPT_ID / f"{rid}.json"
        source = DATA / "resumes" / f"{rid}.txt"
        if cached.exists() and source.exists() and source.read_text(encoding="utf-8") == text:
            saved = json.loads(cached.read_text(encoding="utf-8"))
            if "extraction" in saved:
                return to_fields(Extraction.model_validate(saved["extraction"]))
        return live.parse(rid, text)

    def titles(job: JobRequirement, wanted: list[str]) -> dict[str, TitleCall]:
        path = RESULTS / "titles" / model / TITLE_PROMPT_ID / f"{job.job_id}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            calls = {k: TitleCall.model_validate(v) for k, v in saved.items()}
            if all(title_key(t) in calls for t in wanted):
                return calls
        return live.classify_titles(job, wanted)

    def reasons(job: JobRequirement, items: Items) -> dict[str, Explanation]:
        dumped_items = [(r.model_dump(), f.model_dump(), m.model_dump()) for r, f, m in items]
        key = json.dumps([job.model_dump(), dumped_items], sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        path = RESULTS / "reasons" / model / REASON_PROMPT_ID / f"{job.job_id}-{digest}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return {rid: Explanation.model_validate(e) for rid, e in saved.items()}
        out = live.explain(job, items)
        path.parent.mkdir(parents=True, exist_ok=True)
        dumped = {rid: e.model_dump() for rid, e in out.items()}
        path.write_text(json.dumps(dumped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return out

    return Services(parse=parse, classify_titles=titles, explain=reasons)
