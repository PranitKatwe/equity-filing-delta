"""delta/risk_factors.py — diff-grounded LLM classification of risk-factor changes.

The LLM is shown ONLY the changed passages produced by diff.py — never the full
filing — and classifies each one. This is the honesty guarantee: the model can
label a change it is shown, but it cannot invent a change it was never given,
and every resulting feature traces back to a specific source passage.

Categories for an ADDED passage:
  new_substantive_risk    a genuinely new or materially expanded risk  (the signal)
  expanded_existing_risk  more detail on a risk already disclosed
  reworded_same_meaning   same risk, different words
  boilerplate_or_reorder  generic framing / intro / legalese, no substantive risk

Only `new_substantive_risk` counts toward the sharpened signal `n_substantive_added`,
which is cleaner than the mechanical `n_added` (that one still includes boilerplate).

Model: GLM 5.2 via NVIDIA's OpenAI-compatible endpoint (free on build.nvidia.com),
same provider as the narrator. Configurable via EQD_LLM_MODEL / EQD_LLM_BASE_URL.
Requires NVIDIA_API_KEY.
"""

from __future__ import annotations

import json
import os
import re

from openai import OpenAI

from .diff import SectionDiff

MODEL = os.getenv("EQD_LLM_MODEL", "z-ai/glm-5.2")
BASE_URL = os.getenv("EQD_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")

ADDED_CATEGORIES = [
    "new_substantive_risk",
    "expanded_existing_risk",
    "reworded_same_meaning",
    "boilerplate_or_reorder",
]

_SYSTEM = (
    "You classify year-over-year changes in the Risk Factors (Item 1A) section of "
    "SEC 10-K filings. You are shown ONLY the sentences that changed between last "
    "year's filing and this year's — never the full document. Classify each strictly "
    "from the text shown. Never speculate about content you were not given. A change "
    "is 'new_substantive_risk' only if it introduces or materially expands a real, "
    "specific risk to the business — not generic framing, cautionary boilerplate, or "
    "pure rewording.\n\n"
    "Respond with ONLY a JSON object, no prose before or after, in exactly this shape:\n"
    '{"classifications": [{"index": <int>, "category": "<one of: '
    "new_substantive_risk | expanded_existing_risk | reworded_same_meaning | "
    'boilerplate_or_reorder>", "rationale": "<at most 10 words>"}]}\n'
    "Include every input index exactly once."
)


def _require_key() -> None:
    if not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError(
            "NVIDIA_API_KEY not set — the LLM classifier runs on your free "
            "build.nvidia.com key. Add it to .env to enable classification."
        )


def _extract_json(text: str) -> dict:
    """Parse the model's JSON, tolerating a reasoning trace or stray prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def classify_added(passages: list[str], client: OpenAI | None = None) -> list[dict]:
    """Classify each ADDED passage. Returns [{index, category, rationale}, ...].

    One API call for the whole batch (cost-efficient). `client` is injectable for
    testing; by default a real client is created (needs NVIDIA_API_KEY).
    """
    if not passages:
        return []
    if client is None:
        _require_key()
        client = OpenAI(base_url=BASE_URL, api_key=os.getenv("NVIDIA_API_KEY"))

    numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
    prompt = (
        "These sentences are NEW in this year's Item 1A versus last year's. "
        "Classify each by its index.\n\n" + numbered
    )
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        top_p=1,
        max_tokens=8000,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content or ""
    rows = _extract_json(content)["classifications"]
    # Keep only well-formed rows with known categories (defensive at scale).
    return [
        r for r in rows
        if isinstance(r, dict) and r.get("category") in ADDED_CATEGORIES
        and isinstance(r.get("index"), int)
    ]


def classify_diff(diff: SectionDiff, client: OpenAI | None = None) -> dict:
    """Sharpened, LLM-derived features for one filing event's diff.

    Returns counts by category plus `n_substantive_added` — new_substantive_risk +
    expanded_existing_risk — the diff-grounded analog of 'added risk factors'.
    """
    results = classify_added(diff.added, client=client)
    counts = {c: 0 for c in ADDED_CATEGORIES}
    for r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    return {
        "n_substantive_added": counts["new_substantive_risk"] + counts["expanded_existing_risk"],
        "n_new_substantive_risk": counts["new_substantive_risk"],
        "n_boilerplate_added": counts["boilerplate_or_reorder"] + counts["reworded_same_meaning"],
        "classifications": results,
    }
