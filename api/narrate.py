"""Vercel serverless function: GET /api/narrate?event=<KEY>

Turns ONE pre-computed filing event into a grounded, cited, no-advice memo.

Why this is safe to expose publicly:
  * It narrates ONLY the fixed events in api/data/events.json (a closed allowlist).
    An unknown `event` is rejected before any model call — no arbitrary text can
    reach the LLM, so a public URL cannot be coerced into narrating made-up data.
  * Every number in the prompt is a value that was pre-computed by the study
    harness; the model is told to invent nothing and to give no advice.
  * A per-instance rate limiter + a warm-instance cache bound how often the key
    is actually spent (there are only ~10 possible inputs).

Env: ANTHROPIC_API_KEY (set in the Vercel project, not committed).
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import anthropic

MODEL = os.getenv("EQD_NARRATE_MODEL", "claude-sonnet-5")

# Same grounding contract as src/eqd/narrate/memo.py (duplicated so the serverless
# bundle stays self-contained and does not import pandas / the whole package).
_SYSTEM = (
    "You write a short, factual memo about one SEC 10-K filing: what changed in its "
    "Risk Factors year over year, and how the stock's abnormal return behaved around "
    "the filing. You are given ALL the numbers pre-computed. Rules, strictly:\n"
    "1. Use ONLY the numbers provided. Never invent, estimate, or compute a new figure.\n"
    "2. Purely descriptive. No buy/sell/hold, no recommendation, no forecast, no advice.\n"
    "3. Cite the accession number as the source.\n"
    "4. Neutral and concise: 3-4 sentences. Abnormal returns are versus an "
    "expected-return model; say 'abnormal return', not 'return'."
)

_EVENTS = json.loads((Path(__file__).parent / "data" / "events.json").read_text("utf-8"))

# --- Per-instance guards (module state persists across warm invocations) ---
_CACHE: dict[str, str] = {}          # event key -> generated memo (free on repeat)
_HITS: list[float] = []              # request timestamps for the token bucket
_MAX_PER_MIN = 20                    # cap model calls per warm instance per minute


def _rate_limited() -> bool:
    now = time.time()
    _HITS[:] = [t for t in _HITS if now - t < 60.0]
    if len(_HITS) >= _MAX_PER_MIN:
        return True
    _HITS.append(now)
    return False


def _memo(key: str, client=None) -> str:
    if key in _CACHE:
        return _CACHE[key]
    facts = _EVENTS[key]
    client = client or anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=_SYSTEM,
        messages=[{"role": "user", "content":
                   "Write the memo from these computed values:\n" + json.dumps(facts, default=str)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    _CACHE[key] = text
    return text


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        params = parse_qs(urlparse(self.path).query)
        key = (params.get("event") or [""])[0]

        if key == "__list__":                       # lets the page discover the allowlist
            self._send(200, {"events": {
                k: {"ticker": v["ticker"], "sector": v["sector"],
                    "filing_date": v["filing_date"], "net_added": v["net_added"],
                    "car_0_5": v.get("market_model__car_0_5")}
                for k, v in _EVENTS.items()}})
            return

        if key not in _EVENTS:                      # the allowlist guard
            self._send(404, {"error": "unknown event",
                             "allowed": sorted(_EVENTS.keys())})
            return

        if not os.getenv("ANTHROPIC_API_KEY"):
            self._send(503, {"error": "narrator not configured "
                             "(ANTHROPIC_API_KEY unset on the server)"})
            return

        if _rate_limited():
            self._send(429, {"error": "rate limit — try again shortly"})
            return

        try:
            memo = _memo(key)
        except Exception as exc:                    # never leak a stack trace
            self._send(502, {"error": f"narrator failed: {type(exc).__name__}"})
            return

        facts = _EVENTS[key]
        self._send(200, {
            "event": key,
            "memo": memo,
            "facts": {"ticker": facts["ticker"], "sector": facts["sector"],
                      "accession": facts["accession"], "filing_date": facts["filing_date"],
                      "net_added": facts["net_added"],
                      "car_0_5": facts.get("market_model__car_0_5"),
                      "car_0_21": facts.get("market_model__car_0_21")},
            "cached": key in _CACHE,
        })
