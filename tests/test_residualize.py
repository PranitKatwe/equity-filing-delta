"""residualize.py: the expanding fit must strip sector effects without lookahead."""

import numpy as np
import pandas as pd

from eqd.study.residualize import expanding_residual


def _panel(seed=7, n_per_year=60, years=(2015, 2016, 2017, 2018)):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for i in range(n_per_year):
            sector = "Utilities" if i % 2 == 0 else "Tech"
            base = 0.8 if sector == "Utilities" else 0.2  # sector-predictable level
            rows.append({
                "t0": pd.Timestamp(f"{y}-06-01"),
                "sector": sector,
                "item1a_chars": 40_000 + 100 * i,
                "emb_delta": base + rng.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


def test_first_year_has_no_residual():
    df = _panel()
    res = expanding_residual(df, "emb_delta")
    first = pd.to_datetime(df["t0"]).dt.year == 2015
    assert res[first].isna().all()
    assert res[~first].notna().all()


def test_sector_effect_is_stripped():
    df = _panel()
    res = expanding_residual(df, "emb_delta")
    df = df.assign(resid=res).dropna(subset=["resid"])
    # raw signal differs by ~0.6 across sectors; residuals should be near zero for both
    by_sector = df.groupby("sector")["resid"].mean()
    assert abs(by_sector["Utilities"] - by_sector["Tech"]) < 0.05


def test_no_lookahead():
    df = _panel()
    res_before = expanding_residual(df, "emb_delta")
    tampered = df.copy()
    last = pd.to_datetime(tampered["t0"]).dt.year == 2018
    tampered.loc[last, "emb_delta"] = 99.0  # rewrite the future
    res_after = expanding_residual(tampered, "emb_delta")
    # residuals for 2016-2017 must be identical: the future may not leak backward
    early = pd.to_datetime(df["t0"]).dt.year.isin([2016, 2017])
    pd.testing.assert_series_equal(res_before[early], res_after[early])


def test_missing_signal_rows_stay_nan():
    df = _panel()
    df.loc[df.index[100], "emb_delta"] = np.nan
    res = expanding_residual(df, "emb_delta")
    assert np.isnan(res.iloc[100])
