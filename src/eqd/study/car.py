"""study/car.py — cumulative abnormal returns over event windows + placebo.

For one filing event (ticker, t0), compute CAR under each expected-return model
over several windows. Day-0 is t0; the day-d abnormal return uses the
close-to-close return of session t0+d (i.e. close_{t0+d-1} -> close_{t0+d}).

Why this is no-lookahead: for the typical after-close filing on day D, t0=D+1,
so the day-0 return close_D -> close_{D+1} captures the reaction to news that
was NOT in close_D. The estimation window for the market model is [-120,-21],
strictly before the event, so beta never sees the event.

Windows:
  ar_0_1   [0,+1]   announcement reaction
  car_0_5  [0,+5]   short drift
  car_0_21 [0,+21]  post-filing drift
  placebo_pre [-5,-1]  PRE-event: should be ~0. Large values => leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..eventtime import TradingCalendar
from .abnormal import (
    MARKET_ADJUSTED,
    MARKET_MODEL,
    SECTOR_ADJUSTED,
    ReturnsBook,
    abnormal_returns,
    estimate_market_model,
)

WINDOWS = {
    "ar_0_1": (0, 1),
    "car_0_5": (0, 5),
    "car_0_21": (0, 21),
    "placebo_pre": (-5, -1),
}
EST_WINDOW = (-120, -21)
MODELS = (MARKET_ADJUSTED, SECTOR_ADJUSTED, MARKET_MODEL)


def _car_over_windows(stock_ret: pd.Series, expected_ret: pd.Series, cal, t0) -> dict:
    """Sum abnormal returns over each window; NaN if the window is incomplete."""
    out = {}
    for name, (lo, hi) in WINDOWS.items():
        try:
            widx = cal.window(t0, lo, hi)
        except ValueError:
            out[name] = np.nan
            continue
        ar = abnormal_returns(stock_ret, expected_ret, widx)
        out[name] = float(ar.sum()) if ar.notna().all() else np.nan
    return out


def compute_event(
    book: ReturnsBook,
    cal: TradingCalendar,
    ticker: str,
    t0,
    *,
    sector_etf: str | None = None,
    benchmark: str = "SPY",
) -> dict:
    """CARs for one event under all available models. Flat dict -> a panel row.

    Keys look like 'market_model__car_0_5'. Also returns 'beta', 'n_est', and
    per-model coverage flags. Missing inputs degrade to NaN, never raise.
    """
    t0 = pd.Timestamp(t0).normalize()
    row: dict = {"ticker": ticker, "t0": t0.date()}

    try:
        stock = book.returns(ticker)
        mkt = book.returns(benchmark)
    except FileNotFoundError:
        return {**row, "error": "missing_prices"}

    # market-adjusted
    for name, car in _car_over_windows(stock, mkt, cal, t0).items():
        row[f"{MARKET_ADJUSTED}__{name}"] = car

    # sector-adjusted (if we have the sector ETF)
    if sector_etf and book.has(sector_etf):
        sector = book.returns(sector_etf)
        for name, car in _car_over_windows(stock, sector, cal, t0).items():
            row[f"{SECTOR_ADJUSTED}__{name}"] = car

    # momentum control: pre-event [-252,-21] cumulative return (point-in-time).
    try:
        mom_idx = cal.window(t0, -252, -21)
        mom = stock.reindex(mom_idx)
        row["momentum"] = float((1 + mom).prod() - 1) if mom.notna().all() else np.nan
    except ValueError:
        row["momentum"] = np.nan

    # market-model (beta on the pre-event estimation window)
    try:
        est_index = cal.window(t0, *EST_WINDOW)
        mm = estimate_market_model(stock, mkt, est_index)
        expected = mm.expected(mkt)
        row["beta"] = round(mm.beta, 4)
        row["n_est"] = mm.n_obs
        for name, car in _car_over_windows(stock, expected, cal, t0).items():
            row[f"{MARKET_MODEL}__{name}"] = car
    except (ValueError, KeyError):
        row["beta"] = np.nan
        row["n_est"] = 0

    return row
