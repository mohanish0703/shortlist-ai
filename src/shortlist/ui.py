"""Streamlit UI for the recruiter; talks to the API over HTTP.

uvicorn shortlist.api:app            # first, the API
streamlit run src/shortlist/ui.py    # then the UI (SHORTLIST_API overrides the API URL)
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.environ.get("SHORTLIST_API", "http://127.0.0.1:8000")
TIER_LABEL = {"shortlist": "Shortlist", "call": "Call to check", "reject": "Reject"}
STATUS_ICON = {"pass": "✅", "soft": "🟡", "fail": "❌", "unknown": "❔"}


def api(method: str, path: str, **kw):
    r = httpx.request(method, f"{API}{path}", timeout=600, **kw)
    if r.status_code >= 400:
        st.error(f"{r.status_code}: {r.json().get('detail', r.text)}")
        st.stop()
    return r.json()


st.set_page_config(page_title="ShortlistAI", layout="wide")
st.title("ShortlistAI")
st.caption("All bundled resumes are synthetic: invented people, masked phone numbers.")

with st.sidebar:
    st.header("New session")
    jobs = {j["job_id"]: j for j in api("GET", "/jobs")}
    job_id = st.selectbox(
        "Job", list(jobs), format_func=lambda j: f"{j}: {jobs[j]['title']}, {jobs[j]['location']}"
    )
    st.write(jobs[job_id]["description"])
    if st.button("Rank the synthetic resumes", type="primary"):
        with st.spinner("Parsing, matching and ranking..."):
            st.session_state.thread = api("POST", "/sessions", json={"job_id": job_id})["thread_id"]
    st.header("Sessions")
    for s in api("GET", "/sessions"):
        if st.button(f"{s['thread_id']} ({s['status'].replace('_', ' ')})", key=s["thread_id"]):
            st.session_state.thread = s["thread_id"]

thread = st.session_state.get("thread")
if not thread:
    st.info("Pick a job and rank its resumes, or open a session on the left.")
    st.stop()

v = api("GET", f"/sessions/{thread}")
job = v["job"]
st.subheader(f"{job['title']} x{job['openings']}, {job['location']}  ·  {thread}")

waiting = v["status"] == "waiting_for_review"
keep: list[str] = []
for c in v["candidates"]:
    cols = st.columns([0.6, 3, 1.3, 0.8, 8])
    picked = cols[0].checkbox(
        "keep",
        value=c["proposed"] if waiting else c["resume_id"] in (v["final"] or []),
        key=f"{thread}-{c['resume_id']}",
        disabled=not waiting,
        label_visibility="collapsed",
    )
    if picked:
        keep.append(c["resume_id"])
    cols[1].markdown(f"**{c['rank']}. {c['name'] or '?'}**  \n`{c['resume_id']}`")
    cols[2].write(TIER_LABEL[c["tier"]])
    cols[3].write(f"{c['score']:g}")
    with cols[4]:
        note = " *(plain reason: the LLM's failed the grounding check)*"
        st.write(c["reason"] + (note if c["reason_source"] == "template" else ""))
        with st.expander("checks"):
            for check in c["checks"]:
                st.write(f"{STATUS_ICON[check['status']]} **{check['name']}**: {check['detail']}")

if v["manual_review"]:
    st.warning(
        "Could not be parsed, needs a person: "
        + ", ".join(m["resume_id"] for m in v["manual_review"])
    )

if waiting:
    st.divider()
    note = st.text_input("Note (why you changed it)")
    a, b = st.columns(2)
    if a.button("Approve the ticked candidates", type="primary"):
        proposed = set(v["proposed"])
        decision = {
            "action": "approve" if set(keep) == proposed else "edit",
            "remove": sorted(proposed - set(keep)),
            "add": [rid for rid in keep if rid not in proposed],
            "note": note,
        }
        api("POST", f"/sessions/{thread}/decision", json=decision)
        st.rerun()
    if b.button("Reject the whole shortlist"):
        api("POST", f"/sessions/{thread}/decision", json={"action": "reject", "note": note})
        st.rerun()
else:
    d = v["decision"]
    st.success(f"Decision: {d['action']}. Final shortlist: {', '.join(v['final']) or 'nobody'}")
