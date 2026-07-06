"""narrate/memo.py — grounded per-event and verdict memos.

The narrator is handed pre-computed numbers and turns them into plain English.
It is forbidden from computing new figures or giving advice — the honesty rule
that lets an LLM sit in a research pipeline without undermining its credibility:

  * Every number in the memo must be one that was passed in (no new arithmetic).
  * Descriptive only — no buy/sell/hold, no recommendation, no prediction.
  * Cite the source (accession number) so the memo is traceable.

Model: configurable via EQD_NARRATE_MODEL; defaults to claude-sonnet-5
(DESIGN §3: Sonnet for routine summarization). Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import os

import anthropic

MODEL = os.getenv("EQD_NARRATE_MODEL", "claude-sonnet-5")

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


def _client(client: anthropic.Anthropic | None) -> anthropic.Anthropic:
    if client is not None:
        return client
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set — the narrator runs on your key.")
    return anthropic.Anthropic()


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def event_memo(row: dict, client: anthropic.Anthropic | None = None) -> str:
    """A grounded, cited, no-advice memo for one filing event (a panel row)."""
    facts = {k: row[k] for k in _EVENT_FIELDS if k in row and row[k] is not None}
    resp = _client(client).messages.create(
        model=MODEL,
        max_tokens=500,
        system=_EVENT_SYSTEM,
        messages=[{"role": "user", "content":
                   "Write the memo from these computed values:\n" + json.dumps(facts, default=str)}],
    )
    return _text(resp)


def verdict_memo(stats: dict, client: anthropic.Anthropic | None = None) -> str:
    """A grounded, honest prose summary of the overall study result."""
    resp = _client(client).messages.create(
        model=MODEL,
        max_tokens=600,
        system=_VERDICT_SYSTEM,
        messages=[{"role": "user", "content":
                   "Summarize this event-study result:\n" + json.dumps(stats, default=str)}],
    )
    return _text(resp)
