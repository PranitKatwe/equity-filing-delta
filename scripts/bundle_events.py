"""Build the small, fixed event bundle the live narrator API is allowed to serve.

The deployed /api/narrate function narrates ONLY these pre-computed events. That
fixed allowlist is both the grounding guarantee (no arbitrary text ever reaches
the model) and the abuse guard (a public URL cannot be coerced into new API calls
on made-up inputs). We pick the most recent priced event for a set of well-known
names so the demo is recognizable.

Output: api/data/events.json  (keyed by "<TICKER>-<filing-year>")

Usage: PYTHONPATH=src .venv/Scripts/python.exe scripts/bundle_events.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eqd.config import PANEL_DIR
from eqd.narrate.memo import _EVENT_FIELDS

# A recognizable spread across sectors. Only names actually in the panel survive.
FEATURED = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "NVDA", "KO", "BA", "PFE"]

OUT = Path(__file__).resolve().parents[1] / "api" / "data" / "events.json"


def main() -> None:
    panel = pd.read_csv(PANEL_DIR / "study_panel.csv")
    priced = panel.dropna(subset=["market_model__car_0_5"])

    bundle: dict[str, dict] = {}
    for tk in FEATURED:
        rows = priced[priced["ticker"] == tk]
        if rows.empty:
            continue
        row = rows.sort_values("t0").iloc[-1]           # most recent priced event
        year = str(row["filing_date"])[:4]
        key = f"{tk}-{year}"
        facts = {}
        for f in _EVENT_FIELDS:
            if f in row and pd.notna(row[f]):
                v = row[f]
                facts[f] = v.item() if hasattr(v, "item") else v
        bundle[key] = facts

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(bundle)} events to {OUT}:")
    for k, v in bundle.items():
        print(f"  {k}: net_added={v.get('net_added')}, "
              f"CAR[0,+5]={v.get('market_model__car_0_5')}")


if __name__ == "__main__":
    main()
