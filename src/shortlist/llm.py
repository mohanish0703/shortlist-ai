"""Gemini client. The key and model name come from `.env` (see `.env.example`)."""

from __future__ import annotations

import os
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langsmith import traceable
from pydantic import BaseModel

from shortlist.synthetic import ROOT

# A fixed model version, not a "-latest" alias, so results stay reproducible.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

T = TypeVar("T", bound=BaseModel)

load_dotenv(ROOT / ".env")


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def make_client() -> genai.Client:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set: copy .env.example to .env and add your key")
    # Rate limits (429) and "high demand" (503) are common on the free tier: back off and retry.
    retry = types.HttpRetryOptions(attempts=6, initial_delay=5, max_delay=60)
    return genai.Client(
        api_key=key, http_options=types.HttpOptions(timeout=180_000, retry_options=retry)
    )


# Shows each call inside the LangGraph trace when LANGSMITH_TRACING=true; a no-op otherwise.
@traceable(run_type="llm", name="gemini")
def generate_structured(
    client: genai.Client, system: str, prompt: str, schema: type[T], model: str | None = None
) -> T:
    """One call with JSON output constrained to `schema`, validated by pydantic."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        response_mime_type="application/json",
        response_json_schema=schema.model_json_schema(),
        # Extraction needs little reasoning; low thinking is several times faster.
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )
    response = client.models.generate_content(
        model=model or model_name(), contents=prompt, config=config
    )
    return schema.model_validate_json(response.text or "")
