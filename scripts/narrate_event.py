"""Demo: generate a grounded memo for one filing event from the study panel.

Picks a real event row and asks the narrator to describe it — what changed and
how the abnormal return behaved — from the pre-computed numbers only.

Requires ANTHROPIC_API_KEY in .env. Runs on your key (one short summarization call).

Usage: PYTHONPATH=src .venv/Scripts/python.exe scripts/narrate_event.py [TICKER]
"""

import sys

import pandas as pd

from eqd.config import PANEL_DIR
from eqd.narrate.memo import event_memo

STUDY_PANEL = PANEL_DIR / "study_panel.csv"


def main() -> None:
    panel = pd.read_csv(STUDY_PANEL)
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    rows = panel[panel["ticker"] == ticker] if ticker else panel
    rows = rows.dropna(subset=["market_model__car_0_5"])
    if rows.empty:
        print(f"No priced events for {ticker!r}.")
        return
    row = rows.iloc[-1].to_dict()   # most recent event

    print(f"Event: {row['ticker']} 10-K filed {row['filing_date']} "
          f"(t0={row['t0']}, accession {row['accession']})")
    print(f"  net_added={row['net_added']}, doc_similarity={row['doc_similarity']}, "
          f"CAR[0,+5]={row['market_model__car_0_5']:+.2%}\n")
    print("--- grounded memo ---")
    print(event_memo(row))


if __name__ == "__main__":
    main()
