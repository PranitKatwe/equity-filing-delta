"""Vercel serverless function: GET /api/ask?company=<TICKER>&q=<question>

Grounded Q&A over ONE company's most recent Item 1A (Risk Factors) diff passages.

Why this is safe to expose publicly:
  * `company` must be a key in the pre-computed passage store (api/_passages.py).
    Unknown -> 404, so the model is only ever grounded in real, extracted text.
  * The model is instructed to answer ONLY from the provided added/removed/reworded
    sentences — descriptive, never advice or prediction. The question is free text
    but length-capped, rate-limited, and cached; it can steer *what* is summarized,
    not *what data* the model sees (only that company's real filing changes).

Model: GLM 5.2 via NVIDIA's OpenAI-compatible endpoint (free on build.nvidia.com).
Env: NVIDIA_API_KEY (set in the Vercel project, not committed).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _passages import PASSAGES  # noqa: E402

MODEL = os.getenv("EQD_NARRATE_MODEL", "z-ai/glm-5.2")
BASE_URL = os.getenv("EQD_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
MAX_Q = 300                     # question length cap (chars)

_SYSTEM = (
    "You answer questions about how a company's SEC 10-K Risk Factors changed year "
    "over year. You are given the ACTUAL added, removed, and reworded sentences from "
    "the filing — nothing else. Rules, strictly:\n"
    "1. Answer ONLY from the provided passages. If they don't address the question, "
    "say the filing changes don't cover it. Never use outside knowledge or invent text.\n"
    "2. Purely descriptive. No advice, no buy/sell/hold, no recommendation, no forecast, "
    "no price target, no view on the stock. If asked for any of those, reply that you can "
    "only describe what changed in the filing.\n"
    "3. Quote or closely paraphrase the relevant sentences; be concise (3-6 sentences).\n"
    "4. Cite the accession number as the source."
)

# --- Per-instance guards ---
_CACHE: dict[str, str] = {}
_HITS: list[float] = []
_MAX_PER_MIN = 20


def _rate_limited() -> bool:
    now = time.time()
    _HITS[:] = [t for t in _HITS if now - t < 60.0]
    if len(_HITS) >= _MAX_PER_MIN:
        return True
    _HITS.append(now)
    return False


def _context(p: dict) -> str:
    def block(title, items):
        return f"{title}:\n" + ("\n".join(f"- {s}" for s in items) if items else "(none)")
    return (f"Filing: 10-K, accession {p['accession']}, filed {p['filing_date']} "
            f"(vs prior accession {p['prior_accession']}).\n\n"
            + block("ADDED risk-factor sentences", p["added"]) + "\n\n"
            + block("REMOVED risk-factor sentences", p["removed"]) + "\n\n"
            + block("REWORDED risk-factor sentences (new wording)", p["modified"]))


def _answer(company: str, question: str, client=None) -> str:
    ck = company + ":" + hashlib.sha1(question.lower().encode("utf-8")).hexdigest()[:16]
    if ck in _CACHE:
        return _CACHE[ck]
    p = PASSAGES[company]
    client = client or OpenAI(base_url=BASE_URL, api_key=os.getenv("NVIDIA_API_KEY"))
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        top_p=1,
        max_tokens=700,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content":
             f"Company {company}. Question: {question}\n\n"
             f"Here are the ONLY filing changes you may use:\n\n{_context(p)}"},
        ],
    )
    content = resp.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    _CACHE[ck] = text
    return text


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        params = parse_qs(urlparse(self.path).query)
        company = (params.get("company") or [""])[0].strip().upper()
        question = (params.get("q") or [""])[0].strip()

        if company == "__LIST__":
            self._send(200, {"companies": len(PASSAGES)})
            return
        if company not in PASSAGES:
            self._send(404, {"error": f"no filing passages for {company!r}"})
            return
        if not question:
            self._send(400, {"error": "empty question"})
            return
        if len(question) > MAX_Q:
            question = question[:MAX_Q]
        if not os.getenv("NVIDIA_API_KEY"):
            self._send(503, {"error": "Q&A not configured (NVIDIA_API_KEY unset on the server)"})
            return
        if _rate_limited():
            self._send(429, {"error": "rate limit — try again shortly"})
            return

        try:
            answer = _answer(company, question)
        except Exception as exc:
            self._send(502, {"error": f"Q&A failed: {type(exc).__name__}"})
            return

        p = PASSAGES[company]
        self._send(200, {
            "company": company,
            "question": question,
            "answer": answer,
            "accession": p["accession"],
            "filing_date": p["filing_date"],
            "counts": {"added": len(p["added"]), "removed": len(p["removed"]),
                       "reworded": len(p["modified"])},
        })
