"""study/crosssection.py — does the filing delta explain abnormal returns?

Regress event CAR on the delta features plus controls, with robust or
clustered standard errors. Regressors are standardized by default so a
coefficient reads as "CAR change per 1 SD of the signal", comparable across
features.

Standard errors: filing events cluster in time (many firms file the same
week), so returns are cross-sectionally correlated. Clustering on the event
date (t0) is the honest default; plain HC1 robust SE is the fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def cross_section(
    panel: pd.DataFrame,
    y: str,
    regressors: list[str],
    controls: list[str] = (),
    *,
    cluster: str | None = "t0",
    standardize: bool = True,
):
    """Fit CAR ~ regressors + controls. Returns a fitted statsmodels result."""
    terms = list(regressors) + list(controls)
    cols = list(dict.fromkeys([y, *terms, *([cluster] if cluster else [])]))
    df = panel[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(df) < len(terms) + 2:
        raise ValueError(f"too few complete rows ({len(df)}) for {len(terms)} regressors")

    if standardize:
        for c in terms:
            sd = df[c].std()
            df[c] = (df[c] - df[c].mean()) / sd if sd > 0 else 0.0

    formula = f"{y} ~ " + (" + ".join(terms) if terms else "1")
    if cluster:
        return smf.ols(formula, df).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster]}
        )
    return smf.ols(formula, df).fit(cov_type="HC1")


def tidy(res, terms: list[str] | None = None) -> pd.DataFrame:
    """Coefficient table (coef, se, t, p) for the terms of interest."""
    out = pd.DataFrame(
        {"coef": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues}
    )
    if terms:
        out = out.loc[[t for t in terms if t in out.index]]
    return out.round(6)
