"""Baselines for the ablation: rank raw resume text against the job, without the agent.

- keyword: BM25 of the job's words (title, description, location, languages, preferred skills)
  over the resume texts
- embedding: cosine similarity of Gemini embeddings of the job and of each resume
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

from google.genai import types

from shortlist import llm
from shortlist.schema import JobRequirement
from shortlist.synthetic import ROOT

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
EMBED_CACHE = ROOT / "results" / "embeddings" / f"{EMBED_MODEL}-{EMBED_DIM}.json"
STOPWORDS = {
    *("a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "i"),
    *("in", "is", "it", "my", "of", "on", "or", "the", "to", "with", "year", "years", "yrs"),
}


def job_query(job: JobRequirement) -> str:
    parts = [job.title, job.description, job.location, *job.languages_required]
    return " ".join([*parts, *job.languages_any, *job.preferred_skills])


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) > 1 and t not in STOPWORDS]


def bm25(query: str, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    toks = {d: tokens(t) for d, t in docs.items()}
    avg = sum(map(len, toks.values())) / len(toks)
    df = Counter(w for ts in toks.values() for w in set(ts))
    n = len(toks)
    out = {}
    for d, ts in toks.items():
        tf = Counter(ts)
        s = 0.0
        for w in set(tokens(query)):
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(ts) / avg))
        out[d] = s
    return out


def _key(text: str, task: str) -> str:
    return hashlib.sha256(f"{task}\n{text}".encode()).hexdigest()[:16]


def embed(texts: list[str], task: str) -> list[list[float]]:
    """Embeddings, cached by text in results/embeddings/."""
    cache = json.loads(EMBED_CACHE.read_text(encoding="utf-8")) if EMBED_CACHE.exists() else {}
    todo = list(dict.fromkeys(t for t in texts if _key(t, task) not in cache))
    if todo:
        client = llm.make_client()
        config = types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM)
        for i in range(0, len(todo), 100):
            batch = todo[i : i + 100]
            r = client.models.embed_content(model=EMBED_MODEL, contents=batch, config=config)
            for text, e in zip(batch, r.embeddings or [], strict=True):
                cache[_key(text, task)] = [round(v, 6) for v in e.values or []]
        EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        EMBED_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return [cache[_key(t, task)] for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def embedding_scores(job: JobRequirement, docs: dict[str, str]) -> dict[str, float]:
    (q,) = embed([job_query(job)], "RETRIEVAL_QUERY")
    ids = list(docs)
    vecs = embed([docs[d] for d in ids], "RETRIEVAL_DOCUMENT")
    return {d: cosine(q, v) for d, v in zip(ids, vecs, strict=True)}
