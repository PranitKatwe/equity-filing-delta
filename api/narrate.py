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

Model: GLM 5.2 via NVIDIA's OpenAI-compatible endpoint (free on build.nvidia.com).
Env: NVIDIA_API_KEY (set in the Vercel project, not committed).
"""

from __future__ import annotations

import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from openai import OpenAI

MODEL = os.getenv("EQD_NARRATE_MODEL", "z-ai/glm-5.2")
BASE_URL = os.getenv("EQD_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")

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

# The fixed event allowlist, embedded (a mirror of api/data/events.json produced
# by scripts/bundle_events.py). Embedded rather than read from disk because
# Vercel's Python bundler ships traced code, not arbitrary data files opened at
# runtime — inlining guarantees the function always has its data.
_EVENTS = {
  "AAPL-2025": {"ticker": "AAPL", "sector": "Information Technology", "accession": "0000320193-25-000079", "filing_date": "2025-10-31", "t0": "2025-10-31", "n_added": 51, "n_removed": 50, "net_added": 1, "doc_similarity": 0.4229, "d_tone_negative": 0.001121, "d_tone_uncertainty": 0.002128, "market_model__ar_0_1": -0.0163085822449843, "market_model__car_0_5": 0.0051126519764815, "market_model__car_0_21": 0.0439553053726464, "market_model__placebo_pre": 0.0268769336406199, "beta": 1.3647},
  "MSFT-2022": {"ticker": "MSFT", "sector": "Information Technology", "accession": "0001564590-22-026876", "filing_date": "2022-07-28", "t0": "2022-07-29", "n_added": 17, "n_removed": 15, "net_added": 2, "doc_similarity": 0.7969, "d_tone_negative": 0.000314, "d_tone_uncertainty": -0.001457, "market_model__ar_0_1": -0.0093305712305713, "market_model__car_0_5": -0.0017428880535869, "market_model__car_0_21": -0.0401241796148169, "market_model__placebo_pre": 0.0203857899186348, "beta": 1.2213},
  "JPM-2026": {"ticker": "JPM", "sector": "Financials", "accession": "0001628280-26-008131", "filing_date": "2026-02-13", "t0": "2026-02-17", "n_added": 79, "n_removed": 129, "net_added": -50, "doc_similarity": 0.0942, "d_tone_negative": 0.003853, "d_tone_uncertainty": 0.007591, "market_model__ar_0_1": 0.0144101950059432, "market_model__car_0_5": -0.0235703424880089, "market_model__car_0_21": -0.016888307075471, "market_model__placebo_pre": -0.0499463650027324, "beta": 0.9568},
  "XOM-2026": {"ticker": "XOM", "sector": "Energy", "accession": "0000034088-26-000045", "filing_date": "2026-02-18", "t0": "2026-02-19", "n_added": 9, "n_removed": 4, "net_added": 5, "doc_similarity": 0.5344, "d_tone_negative": 0.001125, "d_tone_uncertainty": -0.000435, "market_model__ar_0_1": -0.0261615005050279, "market_model__car_0_5": -0.0241844012537613, "market_model__car_0_21": 0.0245761070437523, "market_model__placebo_pre": -0.0061816276412195, "beta": 0.0449},
  "JNJ-2026": {"ticker": "JNJ", "sector": "Health Care", "accession": "0000200406-26-000016", "filing_date": "2026-02-11", "t0": "2026-02-12", "n_added": 12, "n_removed": 1, "net_added": 11, "doc_similarity": 0.855, "d_tone_negative": -0.001418, "d_tone_uncertainty": 0.000868, "market_model__ar_0_1": 0.0056390117380748, "market_model__car_0_5": -0.0053701806876364, "market_model__car_0_21": -0.0311664984090124, "market_model__placebo_pre": 0.0175986510891843, "beta": -0.0759},
  "WMT-2026": {"ticker": "WMT", "sector": "Consumer Staples", "accession": "0000104169-26-000055", "filing_date": "2026-03-13", "t0": "2026-03-16", "n_added": 61, "n_removed": 82, "net_added": -21, "doc_similarity": 0.5694, "d_tone_negative": -0.000863, "d_tone_uncertainty": 0.0009, "market_model__ar_0_1": -0.0149983971408656, "market_model__car_0_5": -0.0626401398748735, "market_model__car_0_21": -0.0622677897011976, "market_model__placebo_pre": 0.0052602698010022, "beta": -0.1656},
  "NVDA-2026": {"ticker": "NVDA", "sector": "Information Technology", "accession": "0001045810-26-000021", "filing_date": "2026-02-25", "t0": "2026-02-26", "n_added": 65, "n_removed": 88, "net_added": -23, "doc_similarity": 0.6482, "d_tone_negative": 0.003139, "d_tone_uncertainty": 0.00163, "market_model__ar_0_1": -0.0758425470223322, "market_model__car_0_5": -0.0267052789241036, "market_model__car_0_21": 0.0217454118398588, "market_model__placebo_pre": 0.0226097795710197, "beta": 1.8912},
  "KO-2026": {"ticker": "KO", "sector": "Consumer Staples", "accession": "0001628280-26-010047", "filing_date": "2026-02-20", "t0": "2026-02-23", "n_added": 12, "n_removed": 4, "net_added": 8, "doc_similarity": 0.8003, "d_tone_negative": 2.4e-05, "d_tone_uncertainty": -6.2e-05, "market_model__ar_0_1": 0.0082796414988827, "market_model__car_0_5": -0.0017525522239883, "market_model__car_0_21": -0.0947168421801206, "market_model__placebo_pre": 0.0099046989277091, "beta": -0.3094},
  "BA-2026": {"ticker": "BA", "sector": "Industrials", "accession": "0001628280-26-004357", "filing_date": "2026-01-30", "t0": "2026-02-02", "n_added": 31, "n_removed": 47, "net_added": -16, "doc_similarity": 0.6496, "d_tone_negative": 0.000258, "d_tone_uncertainty": 0.000443, "market_model__ar_0_1": 0.0035285314056013, "market_model__car_0_5": 0.0499334900679297, "market_model__car_0_21": 0.0094121194640909, "market_model__placebo_pre": -0.073540987120346, "beta": 1.0516},
  "PFE-2026": {"ticker": "PFE", "sector": "Health Care", "accession": "0000078003-26-000026", "filing_date": "2026-02-26", "t0": "2026-02-27", "n_added": 143, "n_removed": 101, "net_added": 42, "doc_similarity": 0.6, "d_tone_negative": -0.000842, "d_tone_uncertainty": 0.000515, "market_model__ar_0_1": 0.0078539911346609, "market_model__car_0_5": 0.0114968876294091, "market_model__car_0_21": 0.0683405865230474, "market_model__placebo_pre": 0.0019449437722362, "beta": 0.6343},
}

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
    client = client or OpenAI(base_url=BASE_URL, api_key=os.getenv("NVIDIA_API_KEY"))
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.4,
        top_p=1,
        max_tokens=800,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content":
                   "Write the memo from these computed values:\n" + json.dumps(facts, default=str)}],
    )
    content = resp.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
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

        if not os.getenv("NVIDIA_API_KEY"):
            self._send(503, {"error": "narrator not configured "
                             "(NVIDIA_API_KEY unset on the server)"})
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
