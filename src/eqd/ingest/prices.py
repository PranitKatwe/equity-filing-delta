"""prices.py — free daily EOD prices.

Source: Yahoo Finance via `yfinance` (no API key). We originally targeted
Stooq (DESIGN §0), but Stooq added a JavaScript proof-of-work challenge to its
CSV endpoint (2026), making it non-scriptable. yfinance is the design's listed
fallback and returns split- *and* dividend-adjusted closes (auto_adjust=True),
which is what abnormal-return math wants.

We validate for bad ticks/spikes before anything downstream computes returns —
a single bad print silently corrupts an abnormal-return estimate.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from ..config import PRICES_DIR

# ETFs we need for expected-return models (benchmark + sector SPDRs).
BENCHMARK = "SPY"
SECTOR_ETFS = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

SPIKE_THRESHOLD = 0.50  # |daily return| above this is flagged for review


def fetch_prices(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted bars for one ticker, validated. Indexed by naive session date.

    Columns: open, high, low, close, volume. Cached to data/prices/<ticker>.csv.
    """
    cache = PRICES_DIR / f"{ticker}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, parse_dates=["date"]).set_index("date")
        df.index = df.index.normalize()
        return df

    # Yahoo uses '-' for share classes (BRK.B -> BRK-B); cache stays under `ticker`.
    yahoo_symbol = ticker.replace(".", "-")
    raw = yf.Ticker(yahoo_symbol).history(period="max", auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker!r} ({yahoo_symbol})")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    # Yahoo index is tz-aware (ET midnight); collapse to naive session dates.
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df.sort_index()

    df = _validate(df, ticker)

    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(cache, index=False)
    return df


def _validate(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Drop non-positive prices; warn on extreme single-day jumps."""
    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    bad = (df[price_cols] <= 0).any(axis=1)
    if bad.any():
        df = df[~bad]

    ret = df["close"].pct_change()
    spikes = ret.abs() > SPIKE_THRESHOLD
    if spikes.any():
        dates = ", ".join(str(d.date()) for d in df.index[spikes][:5])
        print(f"[prices] WARNING {ticker}: {int(spikes.sum())} spike(s) >|{SPIKE_THRESHOLD:.0%}| "
              f"(e.g. {dates}) — inspect before trusting returns.")
    return df


def fetch_benchmark_and_sectors(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch SPY + all sector SPDR ETFs for expected-return models."""
    out = {BENCHMARK: fetch_prices(BENCHMARK, refresh=refresh)}
    for etf in SECTOR_ETFS.values():
        out[etf] = fetch_prices(etf, refresh=refresh)
    return out
