"""The honest verdict: does the filing delta explain abnormal returns?

Reads data/panel/study_panel.csv and reports, calibrated and honestly:
  1. Sample size + coverage.
  2. LEAKAGE CHECK — mean placebo [-5,-1] CAR (must be ~0) and whether the
     signal predicts the placebo (it must not).
  3. Cross-section — CAR ~ signal + controls (momentum, beta), clustered SE.
  4. Directional sanity vs Lazy Prices — more added risk / lower similarity
     should associate with LOWER subsequent returns.

This is a sanity-check scale run (no holdout is touched here). The holdout is
run once, later, only after specs are frozen.

Usage: PYTHONPATH=src .venv/Scripts/python.exe scripts/run_study.py
"""

from __future__ import annotations

import sys

import pandas as pd

from eqd.config import PANEL_DIR
from eqd.study.crosssection import cross_section, tidy

STUDY_PANEL = PANEL_DIR / "study_panel.csv"
SIGNAL = "net_added"          # primary Lazy-Prices signal: net new risk sentences
CONTROLS = ["momentum", "beta"]
CAR_WINDOWS = ["market_model__car_0_5", "market_adjusted__car_0_5", "market_model__car_0_21"]
PLACEBO = "market_model__placebo_pre"


def main() -> int:
    if not STUDY_PANEL.exists():
        print(f"Missing {STUDY_PANEL}. Run build_panel.py first.")
        return 1
    panel = pd.read_csv(STUDY_PANEL)

    print("=" * 68)
    print("FILING-DELTA EVENT STUDY - honest verdict (sanity-scale, no holdout)")
    print("=" * 68)
    n_priced = panel[CAR_WINDOWS[0]].notna().sum()
    print(f"Events: {len(panel):,} total, {n_priced:,} with complete CARs, "
          f"{panel['ticker'].nunique()} names\n")

    # 1. Leakage check ----------------------------------------------------
    print("-- LEAKAGE CHECK (placebo pre-event window [-5,-1]) --")
    pmean = panel[PLACEBO].mean()
    print(f"  mean placebo CAR = {pmean:+.4%}  (want ~0; large => lookahead)")
    try:
        res_p = cross_section(panel, PLACEBO, [SIGNAL], CONTROLS, cluster="t0")
        p_sig = tidy(res_p, [SIGNAL]).iloc[0]
        print(f"  signal->placebo coef = {p_sig['coef']:+.5f} (p={p_sig['p']:.3f}); "
              f"should be insignificant\n")
    except ValueError as e:
        print(f"  (placebo regression skipped: {e})\n")

    # 2. Cross-section on real windows -----------------------------------
    print(f"-- CROSS-SECTION: CAR ~ {SIGNAL} + {' + '.join(CONTROLS)} (clustered by t0) --")
    print("   Lazy-Prices direction: coef on net_added should be NEGATIVE.\n")
    for y in CAR_WINDOWS:
        if panel[y].notna().sum() < 20:
            print(f"  {y}: too few obs, skipped")
            continue
        res = cross_section(panel, y, [SIGNAL], CONTROLS, cluster="t0")
        s = tidy(res, [SIGNAL]).iloc[0]
        direction = "negative (expected)" if s["coef"] < 0 else "positive (unexpected)"
        star = "***" if s["p"] < 0.01 else "**" if s["p"] < 0.05 else "*" if s["p"] < 0.10 else ""
        print(f"  {y:28} coef={s['coef']:+.5f}  p={s['p']:.3f} {star:3}  ({direction}, n={int(res.nobs)})")

    print("\n(Coefficients are per 1 SD of the standardized signal; CAR in return units.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
