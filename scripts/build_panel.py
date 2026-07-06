"""Build the study panel: delta features + CARs, one row per 10-K event.

Expensive step (downloads + parses filings, diffs consecutive 10-Ks, computes
abnormal returns). Runs on a universe subset or the whole S&P 500. Output ->
data/panel/study_panel.csv, consumed by run_study.py.

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/build_panel.py [--limit N] [--since YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from eqd.config import PANEL_DIR
from eqd.delta.panel import build_panel
from eqd.eventtime import TradingCalendar
from eqd.study.abnormal import ReturnsBook
from eqd.study.car import compute_event
from eqd.universe import fetch_sp500

STUDY_PANEL = PANEL_DIR / "study_panel.csv"
# Start early enough that momentum [-252,-21] before the earliest ~2016 t0 exists.
CAL = TradingCalendar("2013-01-01", "2027-12-31")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--since", default="2015-01-01")
    args = ap.parse_args()

    uni = fetch_sp500()
    if args.limit:
        uni = uni.head(args.limit)
    print(f"Building delta panel for {len(uni)} names (10-Ks since {args.since})...")

    panel = build_panel(uni, CAL, since=args.since)
    if panel.empty:
        print("No events built.")
        return 1
    print(f"  delta panel: {len(panel):,} events\n")

    print("Attaching abnormal returns (3 models x 4 windows) + momentum...")
    book = ReturnsBook()
    car_rows = [
        compute_event(book, CAL, r.ticker, r.t0, sector_etf=r.sector_etf)
        for r in panel.itertuples(index=False)
    ]
    cars = pd.DataFrame(car_rows).drop(columns=["ticker", "t0"], errors="ignore")
    merged = pd.concat([panel.reset_index(drop=True), cars.reset_index(drop=True)], axis=1)

    STUDY_PANEL.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(STUDY_PANEL, index=False)
    n_priced = merged["market_model__car_0_5"].notna().sum()
    print(f"  -> {STUDY_PANEL.name}: {len(merged):,} events, {n_priced:,} with complete CARs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
