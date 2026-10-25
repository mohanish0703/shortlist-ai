"""Review sessions on top of the graph, shared by the CLI and the API.

A session is one graph thread: start it with a job and resumes, it runs until the recruiter
review and waits there (saved in the SQLite checkpointer), then `decide` resumes it.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from langgraph.types import Command

from shortlist.catalog import JOBS
from shortlist.graph import RUNS, Services, build_graph, check_decision, sqlite_checkpointer
from shortlist.schema import JobRequirement
from shortlist.synthetic import DATA

JOB_BY_ID = {j.job_id: j for j in JOBS}


@cache
def graph(db: Path = RUNS / "checkpoints.sqlite"):
    return build_graph(sqlite_checkpointer(db))


def synthetic_resumes(job_id: str) -> dict[str, str]:
    """The synthetic resumes written for one job."""
    lines = (DATA / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines()
    ids = [r["resume_id"] for r in map(json.loads, lines) if r["job_id"] == job_id]
    return {rid: (DATA / "resumes" / f"{rid}.txt").read_text(encoding="utf-8") for rid in ids}


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def start(thread_id: str, job: JobRequirement, resumes: dict[str, str], services: Services) -> dict:
    if graph().get_state(_config(thread_id)).values:
        raise ValueError(f"session {thread_id!r} already exists")
    state = {"job": job.model_dump(), "resumes": resumes}
    graph().invoke(state, _config(thread_id), context=services)
    return view(thread_id)


def decide(thread_id: str, decision: dict, services: Services) -> dict:
    if status(thread_id) != "waiting_for_review":
        raise ValueError(f"session {thread_id!r} is not waiting for a review")
    _, problem = check_decision(graph().get_state(_config(thread_id)).values, decision)
    if problem:
        raise ValueError(problem)
    graph().invoke(Command(resume=decision), _config(thread_id), context=services)
    return view(thread_id)


def status(thread_id: str) -> str:
    snap = graph().get_state(_config(thread_id))
    if not snap.values:
        return "missing"
    if "review" in snap.next:
        return "waiting_for_review"
    return "done" if "final" in snap.values else "running"


def view(thread_id: str) -> dict:
    """What a recruiter needs to see: the ranked candidates with reasons, and the decision."""
    values = graph().get_state(_config(thread_id)).values
    if not values:
        raise KeyError(thread_id)
    reasons, parsed = values.get("reasons", {}), values.get("parsed", {})
    rows = [
        {
            "rank": i,
            "resume_id": r["resume_id"],
            "name": parsed[r["resume_id"]]["name"],
            "tier": r["tier"],
            "score": r["score"],
            "proposed": r["resume_id"] in values.get("proposed", []),
            "reason": reasons.get(r["resume_id"], {}).get("text", ""),
            "reason_source": reasons.get(r["resume_id"], {}).get("source"),
            "checks": values["matches"][r["resume_id"]]["checks"],
        }
        for i, r in enumerate(values.get("ranking", []), 1)
    ]
    return {
        "thread_id": thread_id,
        "status": status(thread_id),
        "job": values["job"],
        "candidates": rows,
        "proposed": values.get("proposed", []),
        "manual_review": [
            {"resume_id": rid, "error": values.get("errors", {}).get(rid, "")}
            for rid in values.get("manual_review", [])
        ],
        "decision": values.get("decision"),
        "final": values.get("final"),
    }


def threads() -> list[str]:
    saver = graph().checkpointer
    return sorted({c.config["configurable"]["thread_id"] for c in saver.list(None)})
