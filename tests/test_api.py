"""The HTTP API with fake LLM services and a temporary checkpoint database."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import J1, PEOPLE, fake_services
from shortlist import api, session
from shortlist.graph import build_graph, sqlite_checkpointer


@pytest.fixture
def client(tmp_path, monkeypatch):
    graph = build_graph(sqlite_checkpointer(tmp_path / "cp.sqlite"))
    monkeypatch.setattr(session, "graph", lambda: graph)
    monkeypatch.setattr(api, "services", lambda: fake_services(tmp_path, []))
    return TestClient(api.app)


def _new(client, thread="s1"):
    resumes = {rid: f"resume text of {rid}" for rid in PEOPLE}
    body = {"job": J1.model_dump(), "resumes": resumes, "thread_id": thread}
    return client.post("/sessions", json=body)


def test_session_waits_for_review_then_takes_a_decision(client):
    r = _new(client)
    assert r.status_code == 200
    v = r.json()
    assert v["status"] == "waiting_for_review"
    assert v["proposed"] == ["R1", "R4", "R3"]
    assert [c["resume_id"] for c in v["candidates"]] == ["R1", "R4", "R3", "R2"]
    assert v["manual_review"][0]["resume_id"] == "R5"

    r = client.post("/sessions/s1/decision", json={"action": "edit", "remove": ["R3"]})
    assert r.status_code == 200
    assert r.json()["status"] == "done"
    assert r.json()["final"] == ["R1", "R4"]
    assert client.get("/sessions").json() == [{"thread_id": "s1", "status": "done"}]


def test_errors(client):
    assert client.get("/sessions/nope").status_code == 404
    assert client.post("/sessions", json={"job_id": "J9"}).status_code == 422
    _new(client)
    assert _new(client).status_code == 409  # same thread id again
    bad = client.post("/sessions/s1/decision", json={"action": "edit", "add": ["R99"]})
    assert bad.status_code == 422
    assert client.get("/sessions/s1").json()["status"] == "waiting_for_review"  # still open
    client.post("/sessions/s1/decision", json={"action": "approve"})
    assert client.post("/sessions/s1/decision", json={"action": "approve"}).status_code == 409
