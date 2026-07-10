"""study/residualize.py — strip the predictable part of a signal, point-in-time.

The Kelly et al. move: what firm characteristics can explain about the text is
not news; only the residual is. Naively residualizing within each year's full
cross-section would let a January filer's residual depend on filings that did
not exist yet — a lookahead. So the fit is EXPANDING: for events in calendar
year y, coefficients come only from years strictly before y. The first usable
year therefore has no residual (NaN), by construction rather than by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_TRAIN = 50  # fewer prior events than this -> no residual for that year


def _design(df: pd.DataFrame, size_col: str, sector_cols: list[str]) -> np.ndarray:
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    X["log_size"] = np.log1p(df[size_col])
    dummies = pd.get_dummies(df["sector"], dtype=float)
    for c in sector_cols:  # align to the training set's sector columns
        X[c] = dummies[c] if c in dummies.columns else 0.0
    return X.to_numpy(dtype=float)


def expanding_residual(
    panel: pd.DataFrame,
    signal: str,
    *,
    size_col: str = "item1a_chars",
    time_col: str = "t0",
) -> pd.Series:
    """Residual of `signal` vs sector + log(section size), fit on prior years only.

    Returns a Series aligned to panel.index; NaN where the signal is missing
    or where too little prior history exists to fit.
    """
    df = panel[[signal, size_col, "sector", time_col]].copy()
    df["_year"] = pd.to_datetime(df[time_col]).dt.year
    ok = df[signal].notna() & df[size_col].notna() & df["sector"].notna()

    out = pd.Series(np.nan, index=panel.index, name=f"{signal}_resid")
    for year in sorted(df.loc[ok, "_year"].unique()):
        train = df[ok & (df["_year"] < year)]
        test = df[ok & (df["_year"] == year)]
        if len(train) < MIN_TRAIN or test.empty:
            continue
        sector_cols = sorted(train["sector"].unique())
        X_tr = _design(train, size_col, sector_cols)
        beta, *_ = np.linalg.lstsq(X_tr, train[signal].to_numpy(dtype=float), rcond=None)
        X_te = _design(test, size_col, sector_cols)
        out.loc[test.index] = test[signal].to_numpy(dtype=float) - X_te @ beta
    return out
