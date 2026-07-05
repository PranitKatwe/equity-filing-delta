"""Full-universe M0 ingest: filing index + prices for the whole S&P 500.

Pulls, for every constituent:
  - the 10-K/10-Q filing index with acceptance datetimes -> t0
  - daily adjusted prices (cached per ticker)
plus the benchmark + sector ETFs. Filing *text* is intentionally NOT bulk
downloaded here — it's fetched lazily in Step 3 for the specific pairs we diff.

Resilient: a failure on one name is logged and skipped, not fatal. Re-runs are
cheap (prices cache; the filing index re-pulls, which is fast metadata).

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/ingest_universe.py [--limit N] [--since YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from eqd.config import PANEL_DIR
from eqd.eventtime import TradingCalendar
from eqd.ingest.edgar import list_filings
from eqd.ingest.prices import BENCHMARK, SECTOR_ETFS, fetch_prices
from eqd.universe import fetch_sp500

FILINGS_INDEX = PANEL_DIR / "filings_index.csv"
CAL = TradingCalendar("2010-01-01", "2027-12-31")


def ingest_filings(uni: pd.DataFrame, since: str) -> pd.DataFrame:
    rows, failures = [], []
    for i, r in enumerate(uni.itertuples(), 1):
        try:
            f = list_filings(r.cik, forms=("10-K", "10-Q"), since=since)
            f.insert(0, "ticker", r.ticker)
            f.insert(1, "sector", r.sector)
            f["t0"] = f["acceptance_datetime"].map(lambda a: CAL.t0(a).date())
            rows.append(f)
        except Exception as e:  # noqa: BLE001 — resilience is the point
            failures.append((r.ticker, str(e)[:80]))
        if i % 25 == 0:
            print(f"  filings {i}/{len(uni)}  ({sum(len(x) for x in rows):,} filings so far)", flush=True)
    if failures:
        print(f"  filing failures ({len(failures)}): {failures[:5]}{' ...' if len(failures) > 5 else ''}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ingest_prices(uni: pd.DataFrame) -> None:
    symbols = list(uni["ticker"]) + [BENCHMARK] + list(SECTOR_ETFS.values())
    failures = []
    for i, sym in enumerate(symbols, 1):
        try:
            fetch_prices(sym)
        except Exception as e:  # noqa: BLE001
            failures.append((sym, str(e)[:60]))
        if i % 25 == 0:
            print(f"  prices {i}/{len(symbols)}", flush=True)
    if failures:
        print(f"  price failures ({len(failures)}): {failures[:5]}{' ...' if len(failures) > 5 else ''}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N constituents (smoke)")
    ap.add_argument("--since", default="2015-01-01", help="earliest filing date to index")
    args = ap.parse_args()

    uni = fetch_sp500()
    if args.limit:
        uni = uni.head(args.limit)
    print(f"Universe: {len(uni)} constituents. Indexing filings since {args.since}.\n")

    print("[1/2] Filing index + t0 ...")
    idx = ingest_filings(uni, args.since)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    idx.to_csv(FILINGS_INDEX, index=False)
    print(f"  -> {len(idx):,} filings across {idx['ticker'].nunique()} names -> {FILINGS_INDEX.name}\n")

    print("[2/2] Prices (constituents + benchmark + sector ETFs) ...")
    ingest_prices(uni)

    print("\nINGEST COMPLETE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
