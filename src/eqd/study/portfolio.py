"""study/portfolio.py — the tradeable claim: a long-short spread, net of costs.

Sort filing events into quantiles by the signal, then form the Lazy-Prices
trade: LONG the low-signal group (few/negative net additions -> expected higher
abnormal returns) and SHORT the high-signal group (many added risks -> expected
lower returns). The spread is the mean-CAR difference. We report it BOTH gross
and net of a round-trip cost on each leg — a gross-only spread is not a
tradeable claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .costs import ROUNDTRIP_BPS, net_of_cost


def quantile_spread(
    panel: pd.DataFrame,
    signal: str,
    y: str,
    *,
    n_groups: int = 5,
    cost_bps: float = ROUNDTRIP_BPS,
) -> dict:
    """Long low-signal / short high-signal mean-CAR spread, gross and net.

    Returns the spread, a Welch t-test of the low-vs-high group difference,
    and group sizes. NaN-safe.
    """
    df = panel[[signal, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < n_groups * 4:
        return {"error": f"too few rows ({len(df)}) for {n_groups} groups"}

    # Rank-then-bin so ties don't collapse a quantile edge.
    grp = pd.qcut(df[signal].rank(method="first"), n_groups, labels=False)
    low = df.loc[grp == 0, y]                 # few added risks -> long
    high = df.loc[grp == n_groups - 1, y]     # many added risks -> short

    gross = low.mean() - high.mean()
    net = net_of_cost(gross, legs=2, bps=cost_bps)  # pay to trade both legs
    t, p = stats.ttest_ind(low, high, equal_var=False)

    return {
        "gross_spread": round(float(gross), 6),
        "net_spread": round(float(net), 6),
        "roundtrip_cost": round(2 * cost_bps / 1e4, 6),
        "t": round(float(t), 3),
        "p": round(float(p), 4),
        "n_low": int(len(low)),
        "n_high": int(len(high)),
    }
