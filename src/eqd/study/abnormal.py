"""study/abnormal.py — expected-return models and abnormal returns.

AR_{i,t} = r_{i,t} - E[r_{i,t}].  Three models for the expected term:

  market_adjusted : E = r_market                     (SPY)
  sector_adjusted : E = r_sector                     (the stock's sector SPDR)
  market_model    : E = alpha + beta * r_market, with (alpha, beta) OLS-fit on a
                    PRE-event estimation window [-120, -21] so the expectation
                    uses no event-window information (no-lookahead).

Returns are simple close-to-close daily returns on adjusted closes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from ..config import PRICES_DIR

# Model names accepted throughout the harness.
MARKET_ADJUSTED = "market_adjusted"
SECTOR_ADJUSTED = "sector_adjusted"
MARKET_MODEL = "market_model"


def simple_returns(close: pd.Series) -> pd.Series:
    """Close-to-close simple daily returns, sorted by date."""
    return close.sort_index().pct_change()


class ReturnsBook:
    """Lazily loads adjusted-close return series from the cached price CSVs."""

    def __init__(self, price_dir=PRICES_DIR):
        self._dir = price_dir
        self._cache: dict[str, pd.Series] = {}

    def returns(self, symbol: str) -> pd.Series:
        if symbol not in self._cache:
            path = self._dir / f"{symbol}.csv"
            if not path.exists():
                raise FileNotFoundError(f"no cached prices for {symbol!r} at {path}")
            px = pd.read_csv(path, parse_dates=["date"]).set_index("date")["close"]
            px.index = pd.DatetimeIndex(px.index).normalize()
            self._cache[symbol] = simple_returns(px)
        return self._cache[symbol]

    def has(self, symbol: str) -> bool:
        return symbol in self._cache or (self._dir / f"{symbol}.csv").exists()


@dataclass(frozen=True)
class MarketModel:
    """OLS fit r_stock = alpha + beta * r_market over the estimation window."""

    alpha: float
    beta: float
    n_obs: int

    def expected(self, mkt_ret: pd.Series) -> pd.Series:
        return self.alpha + self.beta * mkt_ret


def estimate_market_model(
    stock_ret: pd.Series, mkt_ret: pd.Series, est_index: pd.DatetimeIndex, *, min_obs: int = 30
) -> MarketModel:
    """Fit (alpha, beta) on the estimation-window sessions only."""
    df = pd.concat({"s": stock_ret, "m": mkt_ret}, axis=1).reindex(est_index).dropna()
    if len(df) < min_obs:
        raise ValueError(f"market-model estimation has {len(df)} obs (< {min_obs})")
    res = sm.OLS(df["s"], sm.add_constant(df["m"])).fit()
    return MarketModel(float(res.params["const"]), float(res.params["m"]), len(df))


def abnormal_returns(
    stock_ret: pd.Series, expected_ret: pd.Series, window_index: pd.DatetimeIndex
) -> pd.Series:
    """AR over the event window = realized stock return minus expected return.

    `expected_ret` is r_market / r_sector (adjusted models) or the market-model
    fitted expectation, already aligned by date.
    """
    ar = (stock_ret - expected_ret).reindex(window_index)
    return ar
