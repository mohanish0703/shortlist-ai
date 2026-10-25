"""HTTP API over review sessions.

    uvicorn shortlist.api:app --reload        # docs at http://127.0.0.1:8000/docs

POST /sessions starts a session (the job's synthetic resumes unless `resumes` is given) and
returns once it waits for the recruiter; POST /sessions/{id}/decision resumes it. Sessions
live in the SQLite checkpointer, so they survive a restart of the server.

Set SHORTLIST_LIVE=1 to skip the evaluation caches and always call Gemini.
"""

from __future__ import annotations

import os
import uuid
from functools import cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shortlist import session
from shortlist.graph import Decision, Services, cached_services, live_services
from shortlist.schema import JobRequirement

app = FastAPI(
    title="ShortlistAI",
    description="Ranks resumes against a job with grounded reasons and a recruiter review. "
    "The bundled resumes are SYNTHETIC.",
)


@cache
def services() -> Services:
    return live_services() if os.environ.get("SHORTLIST_LIVE") == "1" else cached_services()


class NewSession(BaseModel):
    job_id: str | None = Field(default=None, description="One of GET /jobs")
    job: JobRequirement | None = Field(default=None, description="Or a job of your own")
    resumes: dict[str, str] | None = Field(
        default=None, description="resume id -> text; default: the job's synthetic resumes"
    )
    thread_id: str | None = None


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/jobs")
def jobs() -> list[JobRequirement]:
    return list(session.JOB_BY_ID.values())


@app.get("/sessions")
def list_sessions() -> list[dict]:
    return [{"thread_id": t, "status": session.status(t)} for t in session.threads()]


@app.post("/sessions")
def new_session(body: NewSession) -> dict:
    if body.job is not None:
        job = body.job
    elif body.job_id in session.JOB_BY_ID:
        job = session.JOB_BY_ID[body.job_id]
    else:
        raise HTTPException(422, "give a known job_id or a job")
    if body.resumes is None and body.job is not None:
        raise HTTPException(422, "a custom job needs resumes")
    resumes = body.resumes if body.resumes is not None else session.synthetic_resumes(job.job_id)
    if not resumes:
        raise HTTPException(422, "no resumes")
    thread_id = body.thread_id or f"{job.job_id}-{uuid.uuid4().hex[:6]}"
    try:
        return session.start(thread_id, job, resumes, services())
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@app.get("/sessions/{thread_id}")
def get_session(thread_id: str) -> dict:
    try:
        return session.view(thread_id)
    except KeyError as e:
        raise HTTPException(404, f"no session {thread_id}") from e


@app.post("/sessions/{thread_id}/decision")
def decide(thread_id: str, decision: Decision) -> dict:
    status = session.status(thread_id)
    if status == "missing":
        raise HTTPException(404, f"no session {thread_id}")
    if status != "waiting_for_review":
        raise HTTPException(409, f"session {thread_id} is {status}")
    try:
        return session.decide(thread_id, decision.model_dump(), services())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
