"""End-to-end smoke test of the ingest layer on a few names.

Proves the pipeline holds together before we scale to all 500:
  universe (Wikipedia) -> filings + acceptance (EDGAR) -> prices (Stooq)
  -> t0 mapping (the spine) -> t0 is a real session with a price.

Run: PYTHONPATH=src .venv/Scripts/python.exe scripts/smoke_ingest.py
"""

import pandas as pd

from eqd.eventtime import TradingCalendar
from eqd.ingest.edgar import download_document, list_filings
from eqd.ingest.prices import fetch_prices
from eqd.universe import sample

TICKERS = ["AAPL", "MSFT", "JPM"]
CAL = TradingCalendar("2020-01-01", "2026-12-31")


def main() -> None:
    print("=== 1. Universe (S&P 500 via Wikipedia) ===")
    uni = sample(TICKERS)
    print(uni.to_string(index=False))

    print("\n=== 2. Filings + acceptance -> t0 (AAPL, 10-K/10-Q since 2023) ===")
    aapl_cik = uni.loc[uni["ticker"] == "AAPL", "cik"].iloc[0]
    filings = list_filings(aapl_cik, forms=("10-K", "10-Q"), since="2023-01-01")
    filings["t0"] = filings["acceptance_datetime"].map(lambda a: CAL.t0(a).date())
    filings["accept_ET"] = filings["acceptance_datetime"].map(
        lambda a: pd.Timestamp(a).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M ET")
    )
    print(filings[["form", "accession", "accept_ET", "filing_date", "t0"]].to_string(index=False))

    print("\n=== 3. Download one filing's primary doc (cached, SHA-256) ===")
    latest_10k = filings[filings["form"] == "10-K"].iloc[-1]
    path = download_document(aapl_cik, latest_10k["accession"], latest_10k["primary_document"])
    print(f"cached: {path.relative_to(path.parents[3])}  ({path.stat().st_size:,} bytes)")

    print("\n=== 4. Prices (yfinance) + benchmark ===")
    px = fetch_prices("AAPL")
    spy = fetch_prices("SPY")
    print(f"AAPL: {px.index.min().date()} .. {px.index.max().date()}  ({len(px):,} bars)")
    print(f"SPY : {spy.index.min().date()} .. {spy.index.max().date()}  ({len(spy):,} bars)")

    print("\n=== 5. Integration check: is every t0 a real session with a price? ===")
    all_ok = True
    for _, row in filings.iterrows():
        t0 = CAL.t0(row["acceptance_datetime"])
        has_px = t0.normalize() in px.index
        is_session = t0.normalize() in CAL.sessions
        ok = has_px and is_session
        all_ok &= ok
        print(f"  {'OK ' if ok else 'FAIL'} {row['form']:5} t0={t0.date()}  "
              f"session={is_session}  price={has_px}")
    print("\nSMOKE TEST PASSED" if all_ok else "\nSMOKE TEST FAILED — investigate above")


if __name__ == "__main__":
    main()
