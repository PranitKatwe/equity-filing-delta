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

Model: claude-opus-4-8 (DESIGN §3: Opus for materiality classification of diffs).
Requires ANTHROPIC_API_KEY. Runs on the user's key and incurs cost.
"""

from __future__ import annotations

import json
import os

import anthropic

from .diff import SectionDiff

MODEL = os.getenv("EQD_LLM_MODEL", "claude-opus-4-8")

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
    "pure rewording."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "category": {"type": "string", "enum": ADDED_CATEGORIES},
                    "rationale": {"type": "string"},
                },
                "required": ["index", "category", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["classifications"],
    "additionalProperties": False,
}


def _require_key() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set — the LLM classifier runs on your key. "
            "Add it to .env to enable diff-grounded risk classification."
        )


def classify_added(passages: list[str], client: anthropic.Anthropic | None = None) -> list[dict]:
    """Classify each ADDED passage. Returns [{index, category, rationale}, ...].

    One API call for the whole batch (cost-efficient). `client` is injectable for
    testing; by default a real Anthropic client is created (needs the API key).
    """
    if not passages:
        return []
    if client is None:
        _require_key()
        client = anthropic.Anthropic()

    numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
    prompt = (
        "These sentences are NEW in this year's Item 1A versus last year's. "
        "Classify each by its index.\n\n" + numbered
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)["classifications"]


def classify_diff(diff: SectionDiff, client: anthropic.Anthropic | None = None) -> dict:
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
