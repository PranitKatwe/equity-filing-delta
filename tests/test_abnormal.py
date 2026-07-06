"""Tests for the returns engine — known-answer synthetic data.

If the market model can't recover a beta we built by construction, the
abnormal returns feeding the whole study are wrong.
"""

import numpy as np
import pandas as pd
import pytest

from eqd.study.abnormal import (
    abnormal_returns,
    estimate_market_model,
    simple_returns,
)

DATES = pd.bdate_range("2022-01-03", periods=150)


def test_simple_returns():
    close = pd.Series([100.0, 110.0, 99.0], index=DATES[:3])
    r = simple_returns(close)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(-0.10)


def test_market_model_recovers_known_alpha_beta():
    rng = np.arange(len(DATES))
    mkt = pd.Series(0.001 * np.sin(rng / 5.0), index=DATES)      # some market path
    stock = 0.0005 + 1.3 * mkt                                    # exact linear relation
    mm = estimate_market_model(stock, mkt, DATES)
    assert abs(mm.alpha - 0.0005) < 1e-6
    assert abs(mm.beta - 1.3) < 1e-6
    assert mm.n_obs == len(DATES)


def test_market_adjusted_abnormal_is_difference():
    mkt = pd.Series(np.linspace(0.01, 0.02, len(DATES)), index=DATES)
    stock = mkt + 0.005                                           # +50bps abnormal daily
    win = DATES[10:15]
    ar = abnormal_returns(stock, mkt, win)
    assert np.allclose(ar.to_numpy(), 0.005)


def test_estimation_too_short_raises():
    mkt = pd.Series([0.01, 0.02, 0.0], index=DATES[:3])
    stock = mkt * 1.1
    try:
        estimate_market_model(stock, mkt, DATES[:3])
        assert False, "expected ValueError"
    except ValueError:
        pass
