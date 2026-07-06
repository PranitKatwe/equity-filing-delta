"""narrate/memo.py — grounded per-event and verdict memos.

The narrator is handed pre-computed numbers and turns them into plain English.
It is forbidden from computing new figures or giving advice — the honesty rule
that lets an LLM sit in a research pipeline without undermining its credibility:

  * Every number in the memo must be one that was passed in (no new arithmetic).
  * Descriptive only — no buy/sell/hold, no recommendation, no prediction.
  * Cite the source (accession number) so the memo is traceable.

Model: GLM 5.2 via NVIDIA's OpenAI-compatible endpoint (free on build.nvidia.com).
Configurable via EQD_NARRATE_MODEL / EQD_LLM_BASE_URL. Requires NVIDIA_API_KEY.
"""

from __future__ import annotations

import json
import os
import re

from openai import OpenAI

MODEL = os.getenv("EQD_NARRATE_MODEL", "z-ai/glm-5.2")
BASE_URL = os.getenv("EQD_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")

_EVENT_SYSTEM = (
    "You write a short, factual memo about one SEC 10-K filing: what changed in its "
    "Risk Factors year over year, and how the stock's abnormal return behaved around "
    "the filing. You are given ALL the numbers pre-computed. Rules, strictly:\n"
    "1. Use ONLY the numbers provided. Never invent, estimate, or compute a new figure.\n"
    "2. Purely descriptive. No buy/sell/hold, no recommendation, no forecast, no advice.\n"
    "3. Cite the accession number as the source.\n"
    "4. Neutral and concise: 3-4 sentences. Abnormal returns are versus an "
    "expected-return model; say 'abnormal return', not 'return'."
)

_VERDICT_SYSTEM = (
    "You write a short, honest summary of an event-study result. You are given the "
    "pre-computed statistics. Rules, strictly:\n"
    "1. Use ONLY the numbers provided. Never invent or compute a new figure.\n"
    "2. Descriptive and calibrated — state clearly whether the effect is significant "
    "and whether it survives costs. No advice, no recommendation, no forecast.\n"
    "3. 3-5 sentences. Do not overstate: a weak or null result should be described as such."
)

# Result-row fields the narrator is allowed to see (keeps it grounded).
_EVENT_FIELDS = [
    "ticker", "sector", "accession", "filing_date", "t0",
    "n_added", "n_removed", "net_added", "doc_similarity",
    "d_tone_negative", "d_tone_uncertainty",
    "market_model__ar_0_1", "market_model__car_0_5", "market_model__car_0_21",
    "market_model__placebo_pre", "beta",
]


def _client(client: OpenAI | None) -> OpenAI:
    if client is not None:
        return client
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not set — the narrator runs on your "
                           "free build.nvidia.com key.")
    return OpenAI(base_url=BASE_URL, api_key=key)


def _text(resp) -> str:
    """Return the assistant text, stripping any reasoning-trace <think> block."""
    content = resp.choices[0].message.content or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()


def _chat(client: OpenAI, system: str, user: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.4,          # low: these are factual restatements, not creative writing
        top_p=1,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return _text(resp)


def event_memo(row: dict, client: OpenAI | None = None) -> str:
    """A grounded, cited, no-advice memo for one filing event (a panel row)."""
    facts = {k: row[k] for k in _EVENT_FIELDS if k in row and row[k] is not None}
    return _chat(_client(client), _EVENT_SYSTEM,
                 "Write the memo from these computed values:\n" + json.dumps(facts, default=str),
                 max_tokens=800)


def verdict_memo(stats: dict, client: OpenAI | None = None) -> str:
    """A grounded, honest prose summary of the overall study result."""
    return _chat(_client(client), _VERDICT_SYSTEM,
                 "Summarize this event-study result:\n" + json.dumps(stats, default=str),
                 max_tokens=900)
