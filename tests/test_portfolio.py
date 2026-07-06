"""Test the long-short spread on known-structure synthetic data."""

import numpy as np
import pandas as pd

from eqd.study.portfolio import quantile_spread


def test_spread_recovers_negative_signal_relationship():
    # y decreases in the signal, so LONG low-signal / SHORT high-signal is +.
    n = 500
    signal = np.arange(n) % 100
    y = -0.0004 * signal  # exact negative relation, no noise
    panel = pd.DataFrame({"net_added": signal, "market_model__car_0_5": y})

    sp = quantile_spread(panel, "net_added", "market_model__car_0_5", n_groups=5)
    assert sp["gross_spread"] > 0            # low-signal group outperforms high
    assert sp["net_spread"] < sp["gross_spread"]   # costs reduce it
    assert sp["roundtrip_cost"] > 0


def test_no_relationship_gives_small_insignificant_spread():
    n = 500
    rng = np.tile(np.arange(100), 5)
    y = np.where(rng % 2 == 0, 0.01, -0.01)  # unrelated to signal ordering
    panel = pd.DataFrame({"net_added": rng, "market_model__car_0_5": y})
    sp = quantile_spread(panel, "net_added", "market_model__car_0_5", n_groups=5)
    assert abs(sp["gross_spread"]) < 0.005   # near zero


def test_too_few_rows_returns_error():
    panel = pd.DataFrame({"net_added": [1, 2, 3], "market_model__car_0_5": [0.1, 0.2, 0.3]})
    assert "error" in quantile_spread(panel, "net_added", "market_model__car_0_5")
