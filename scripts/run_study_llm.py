"""Second (and final) pre-registered question for the holdout:

    Does the LLM-sharpened signal `n_substantive_added` (genuinely new or
    materially expanded risks, boilerplate excluded) beat the mechanical
    `net_added` count?

This was documented in the README as the open next step from the start. It is
asked with the SAME frozen specs as run_study.py (same CAR windows, controls,
clustered SEs, in-sample/holdout split, and net-of-cost spread), and the
holdout is examined for this question exactly once. No other variants are run.

Requires data/panel/llm_classified.csv from scripts/classify_panel.py.

Usage: PYTHONPATH=src .venv/Scripts/python.exe scripts/run_study_llm.py
"""

from __future__ import annotations

import sys

import pandas as pd

from eqd.config import PANEL_DIR
from eqd.study.crosssection import cross_section, tidy
from eqd.study.portfolio import quantile_spread

STUDY_PANEL = PANEL_DIR / "study_panel.csv"
CLASSIFIED = PANEL_DIR / "llm_classified.csv"

# Frozen specs — identical to run_study.py.
SIGNAL = "n_substantive_added"     # the pre-registered sharpened signal
BASELINE = "net_added"             # the mechanical count it must beat
CONTROLS = ["momentum", "beta"]
PRIMARY_Y = "market_model__car_0_5"
CAR_WINDOWS = ["market_model__car_0_5", "market_adjusted__car_0_5", "market_model__car_0_21"]
PLACEBO = "market_model__placebo_pre"
SPLIT_DATE = "2022-01-01"


def _stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _cs(panel: pd.DataFrame, y: str, sig: str):
    res = cross_section(panel, y, [sig], CONTROLS, cluster="t0")
    return tidy(res, [sig]).iloc[0], int(res.nobs)


def _block(panel: pd.DataFrame, label: str) -> None:
    print(f"-- {label} --")
    for y in CAR_WINDOWS:
        sub = panel.dropna(subset=[y, SIGNAL])
        if len(sub) < 20:
            print(f"  {y:28} too few obs, skipped")
            continue
        s, n = _cs(sub, y, SIGNAL)
        b, _ = _cs(sub, y, BASELINE)
        print(f"  {y:28} sharpened coef={s['coef']:+.5f} p={s['p']:.3f} {_stars(s['p']):3} | "
              f"mechanical coef={b['coef']:+.5f} p={b['p']:.3f} {_stars(b['p']):3} (n={n})")


def main() -> int:
    if not CLASSIFIED.exists():
        print(f"Missing {CLASSIFIED}. Run classify_panel.py first.")
        return 1
    panel = pd.read_csv(STUDY_PANEL, parse_dates=["t0"])
    cls = pd.read_csv(CLASSIFIED)
    cls = cls[cls["error"].fillna("") == ""][
        ["accession", "n_sent_to_llm", "n_substantive_added",
         "n_new_substantive_risk", "n_boilerplate"]
    ]
    merged = panel.merge(cls, on="accession", how="inner")

    print("=" * 70)
    print("LLM-SHARPENED SIGNAL - second and final pre-registered holdout question")
    print("=" * 70)
    print(f"Classified events merged: {len(merged):,} of {len(panel):,} "
          f"({merged['ticker'].nunique()} names)")
    corr = merged[[SIGNAL, BASELINE]].corr().iloc[0, 1]
    share = merged["n_substantive_added"].sum() / max(merged["n_sent_to_llm"].sum(), 1)
    print(f"corr({SIGNAL}, {BASELINE}) = {corr:.3f}")
    print(f"substantive share of added sentences = {share:.1%} "
          f"(the rest is boilerplate/rewording the LLM filtered out)\n")

    insample = merged[merged["t0"] < SPLIT_DATE]
    holdout = merged[merged["t0"] >= SPLIT_DATE]
    print(f"in-sample: {len(insample):,}   holdout: {len(holdout):,}\n")

    # Leakage check for the new signal.
    print("-- LEAKAGE CHECK (signal -> placebo [-5,-1]) --")
    sub = merged.dropna(subset=[PLACEBO, SIGNAL])
    s, n = _cs(sub, PLACEBO, SIGNAL)
    print(f"  coef={s['coef']:+.5f}  p={s['p']:.3f}  (want insignificant, n={n})\n")

    _block(insample, "IN-SAMPLE")
    print()
    _block(holdout, "HOLDOUT (single look for this question)")
    print()

    print(f"-- HOLDOUT LONG-SHORT on {SIGNAL} (net of costs, {PRIMARY_Y}) --")
    sp = quantile_spread(holdout.dropna(subset=[SIGNAL]), SIGNAL, PRIMARY_Y, n_groups=5)
    if "error" in sp:
        print(f"  {sp['error']}")
    else:
        print(f"  gross={sp['gross_spread']:+.4%}  net={sp['net_spread']:+.4%}  "
              f"(t={sp['t']}, p={sp['p']}, n={sp['n_low']}+{sp['n_high']})")
        verdict = ("survives costs" if sp["net_spread"] > 0 and sp["p"] < 0.10
                   else "does NOT survive costs / not significant")
        print(f"  -> {verdict}")

    print("\n(Same frozen specs as run_study.py. Coefs per 1 SD; "
          "Lazy-Prices direction = negative coef.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
