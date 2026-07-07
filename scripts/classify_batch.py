"""Precompute diff-grounded risk classifications for every company, once.

Same categories, system prompt, and JSON schema as eqd.delta.risk_factors -
but run on Claude Sonnet 5 through the Batch API (50% off, no rate limits)
over the added Item 1A sentences already cached in api/_passages.py. Results
land as static JSON under docs/classified/, so the website can show what the
LLM labeled without ever making a live model call on a visitor's behalf.

Usage (from repo root, PYTHONPATH=src):
  python scripts/classify_batch.py submit   # create the batch, save its id
  python scripts/classify_batch.py status   # one-line progress check
  python scripts/classify_batch.py fetch    # when ended: write docs/classified/*.json

Requires ANTHROPIC_API_KEY in .env. One-time cost: the full 473-company run
came to ~1.1M input + ~1.3M output tokens, about $8 at Sonnet 5 batch pricing.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from eqd.delta.risk_factors import _SCHEMA, _SYSTEM, ADDED_CATEGORIES, _require_key

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data" / "classify_batch.json"  # gitignored
OUT_DIR = ROOT / "docs" / "classified"
MODEL = "claude-sonnet-5"
MAX_TOKENS = 8000

sys.path.insert(0, str(ROOT / "api"))
from _passages import PASSAGES  # noqa: E402


def _prompt(sentences: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(sentences))
    return (
        "These sentences are NEW in this year's Item 1A versus last year's. "
        "Classify each by its index.\n\n" + numbered
    )


def submit() -> None:
    if STATE.exists():
        state = json.loads(STATE.read_text())
        sys.exit(f"Batch {state['batch_id']} already submitted - run 'status' or 'fetch'. "
                 f"Delete {STATE} to start over.")
    _require_key()
    client = anthropic.Anthropic()

    # custom_id must match ^[a-zA-Z0-9_-]+$; tickers like BRK.B carry a dot
    requests = [
        Request(
            custom_id=ticker.replace(".", "_"),
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=_SYSTEM,
                messages=[{"role": "user", "content": _prompt(p["added"])}],
                output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            ),
        )
        for ticker, p in sorted(PASSAGES.items())
        if p["added"]
    ]
    print(f"submitting {len(requests)} requests "
          f"({sum(len(p['added']) for p in PASSAGES.values())} sentences) on {MODEL} ...")
    batch = client.messages.batches.create(requests=requests)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"batch_id": batch.id, "model": MODEL}))
    print(f"batch {batch.id} created, status={batch.processing_status}")


def status() -> anthropic.types.messages.MessageBatch:
    state = json.loads(STATE.read_text())
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(state["batch_id"])
    c = batch.request_counts
    print(f"{batch.id}: {batch.processing_status} - processing={c.processing} "
          f"succeeded={c.succeeded} errored={c.errored} canceled={c.canceled} expired={c.expired}")
    return batch


def fetch() -> None:
    batch = status()
    if batch.processing_status != "ended":
        sys.exit("not finished yet - try again later.")
    client = anthropic.Anthropic()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}
    in_tok = out_tok = failed = 0

    for result in client.messages.batches.results(batch.id):
        ticker = result.custom_id.replace("_", ".")  # undo the custom_id sanitizing
        if result.result.type != "succeeded":
            failed += 1
            print(f"  {ticker}: {result.result.type} - skipped")
            continue
        msg = result.result.message
        in_tok += msg.usage.input_tokens
        out_tok += msg.usage.output_tokens
        rows = json.loads(next(b.text for b in msg.content if b.type == "text"))["classifications"]

        p = PASSAGES[ticker]
        counts = {c: 0 for c in ADDED_CATEGORIES}
        items = []
        for r in sorted(rows, key=lambda r: r["index"]):
            if not 0 <= r["index"] < len(p["added"]):
                continue
            counts[r["category"]] += 1
            items.append({
                "text": p["added"][r["index"]],
                "category": r["category"],
                "rationale": r["rationale"],
            })
        doc = {
            "ticker": ticker,
            "sector": p["sector"],
            "filing_date": p["filing_date"],
            "accession": p["accession"],
            "prior_accession": p["prior_accession"],
            "model": MODEL,
            "n_added": len(p["added"]),
            "n_substantive": counts["new_substantive_risk"] + counts["expanded_existing_risk"],
            "counts": counts,
            "items": items,
        }
        (OUT_DIR / f"{ticker}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
        index[ticker] = {
            "sector": p["sector"],
            "filing_date": p["filing_date"],
            "n_added": doc["n_added"],
            "n_substantive": doc["n_substantive"],
            "n_new": counts["new_substantive_risk"],
        }

    (OUT_DIR / "index.json").write_text(
        json.dumps({"model": MODEL, "generated": date.today().isoformat(),
                    "companies": dict(sorted(index.items()))}, indent=1),
        encoding="utf-8",
    )
    # Sonnet 5 batch pricing (intro through 2026-08-31): $1 in / $5 out per MTok.
    cost = in_tok / 1e6 * 1.0 + out_tok / 1e6 * 5.0
    print(f"\nwrote {len(index)} companies to {OUT_DIR} ({failed} failed)")
    print(f"usage: {in_tok:,} in + {out_tok:,} out tokens ~ ${cost:.2f} at batch intro pricing")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"submit": submit, "status": status, "fetch": fetch}[cmd]()
