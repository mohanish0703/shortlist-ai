"""Run a shortlist session in the terminal and review it as the recruiter.

    python -m shortlist.cli run J1             # rank the synthetic resumes for job J1, review
    python -m shortlist.cli run J1 --later     # stop at the review; come back with `review`
    python -m shortlist.cli review J1-1        # review a paused session, even after a restart
    python -m shortlist.cli list

Uses the evaluation caches in results/ where they apply (no API calls for the synthetic set
after `make parse-eval`), otherwise Gemini. `--live` skips the caches.
"""

from __future__ import annotations

import argparse
import textwrap

from shortlist import session
from shortlist.graph import cached_services, live_services


def show(v: dict) -> None:
    job = v["job"]
    print(f"\n{v['thread_id']}: {job['title']} x{job['openings']} in {job['location']}")
    print(f"  {job['description']}\n")
    for c in v["candidates"]:
        mark = "*" if c["proposed"] else " "
        head = f"{mark}{c['rank']:>3}. {c['resume_id']} {c['name'] or '?':<22}"
        print(f"{head} {c['tier']:<9} {c['score']:>6g}")
        source = "" if c["reason_source"] != "template" else " [plain reason: LLM's failed check]"
        for line in textwrap.wrap(c["reason"] + source, 70):
            print(f"        {line}")
    if v["manual_review"]:
        print("\nNeeds a person to read (could not be parsed):")
        for m in v["manual_review"]:
            print(f"  {m['resume_id']}: {m['error'][:80]}")
    print(f"\n* proposed shortlist: {', '.join(v['proposed']) or 'nobody'}")
    if v["status"] == "done":
        d = v["decision"]
        print(f"decision: {d['action']}; final shortlist: {', '.join(v['final']) or 'nobody'}")


def ask_decision() -> dict | None:
    while True:
        answer = input("\n[a]pprove  [e]dit  [r]eject  [q]uit (stays paused) > ").strip().lower()
        if answer[:1] == "a":
            return {"action": "approve"}
        if answer[:1] == "r":
            return {"action": "reject", "note": input("why? > ").strip()}
        if answer[:1] == "e":
            remove = input("remove (ids, space separated) > ").upper().split()
            add = input("add (ids, space separated) > ").upper().split()
            note = input("note > ").strip()
            return {"action": "edit", "remove": remove, "add": add, "note": note}
        if answer[:1] == "q":
            return None


def review(thread_id: str, services) -> None:
    v = session.view(thread_id)
    show(v)
    if v["status"] != "waiting_for_review":
        return
    while True:
        decision = ask_decision()
        if decision is None:
            print(f"paused; continue with: python -m shortlist.cli review {thread_id}")
            return
        try:
            show(session.decide(thread_id, decision, services))
            return
        except ValueError as e:
            print(f"cannot apply that: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("job_id", choices=sorted(session.JOB_BY_ID))
    run.add_argument("--thread", help="session name (default: job id and a number)")
    run.add_argument("--later", action="store_true", help="stop at the review")
    rev = sub.add_parser("review")
    rev.add_argument("thread")
    sub.add_parser("list")
    for p in (run, rev):
        p.add_argument("--live", action="store_true", help="always call the API")
    args = ap.parse_args()

    if args.cmd == "list":
        for t in session.threads():
            print(f"{t:<20} {session.status(t)}")
        return
    services = live_services() if args.live else cached_services()
    if args.cmd == "review":
        review(args.thread, services)
        return
    existing = set(session.threads())
    n = 1
    while f"{args.job_id}-{n}" in existing:
        n += 1
    thread_id = args.thread or f"{args.job_id}-{n}"
    job = session.JOB_BY_ID[args.job_id]
    session.start(thread_id, job, session.synthetic_resumes(args.job_id), services)
    if args.later:
        show(session.view(thread_id))
        print(f"\npaused for review; continue with: python -m shortlist.cli review {thread_id}")
    else:
        review(thread_id, services)


if __name__ == "__main__":
    main()
