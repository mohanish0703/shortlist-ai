"""The graph end to end with fake LLM services and a real SQLite checkpointer."""

from __future__ import annotations

import json

from langgraph.types import Command

from conftest import J1, PEOPLE, fake_services
from shortlist.graph import build_graph, sqlite_checkpointer


def _start(tmp_path, thread="t1"):
    calls: list[str] = []
    services = fake_services(tmp_path, calls)
    graph = build_graph(sqlite_checkpointer(tmp_path / "cp.sqlite"))
    config = {"configurable": {"thread_id": thread}}
    resumes = {rid: f"resume text of {rid}" for rid in PEOPLE}
    result = graph.invoke({"job": J1.model_dump(), "resumes": resumes}, config, context=services)
    return graph, config, services, result, calls


def test_runs_to_review_and_waits(tmp_path):
    graph, config, _, result, calls = _start(tmp_path)
    assert graph.get_state(config).next == ("review",)
    assert result["__interrupt__"][0].value["proposed"] == ["R1", "R4", "R3"]
    assert result["manual_review"] == ["R5"]
    assert calls.count("R4") == 2
    assert calls.count("R5") == 2
    assert [r["resume_id"] for r in result["ranking"]] == ["R1", "R4", "R3", "R2"]
    assert result["reasons"]["R2"]["text"].startswith("Rejected")


def test_approve_logs_feedback(tmp_path):
    graph, config, services, _, _ = _start(tmp_path)
    out = graph.invoke(Command(resume={"action": "approve"}), config, context=services)
    assert out["final"] == ["R1", "R4", "R3"]
    log = [json.loads(line) for line in services.feedback_log.read_text().splitlines()]
    assert log[0]["action"] == "approve"
    assert log[0]["thread_id"] == "t1"


def test_resume_after_restart_with_an_edit(tmp_path):
    _start(tmp_path, thread="t2")
    # a new process: new graph, new connection to the same SQLite file
    services = fake_services(tmp_path, [])
    graph = build_graph(sqlite_checkpointer(tmp_path / "cp.sqlite"))
    config = {"configurable": {"thread_id": "t2"}}
    assert graph.get_state(config).next == ("review",)
    edit = {"action": "edit", "remove": ["R3"], "add": ["R2"], "note": "Ravi is relocating"}
    out = graph.invoke(Command(resume=edit), config, context=services)
    assert out["final"] == ["R1", "R4", "R2"]
    entry = json.loads(services.feedback_log.read_text().splitlines()[-1])
    assert [r["resume_id"] for r in entry["removed"]] == ["R3"]
    assert entry["added"][0]["tier"] == "reject"  # the recruiter overrode the ranker
    assert entry["note"] == "Ravi is relocating"


def test_reject(tmp_path):
    graph, config, services, _, _ = _start(tmp_path)
    out = graph.invoke(Command(resume={"action": "reject"}), config, context=services)
    assert out["final"] == []


def test_an_invalid_decision_asks_again(tmp_path):
    graph, config, services, _, _ = _start(tmp_path)
    bad = graph.invoke(Command(resume={"action": "edit", "add": ["R99"]}), config, context=services)
    assert "R99" in bad["__interrupt__"][0].value["error"]
    assert graph.get_state(config).next == ("review",)  # still waiting, not stuck
    out = graph.invoke(Command(resume={"action": "approve"}), config, context=services)
    assert out["final"] == ["R1", "R4", "R3"]
