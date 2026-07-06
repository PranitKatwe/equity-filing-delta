"""The honest verdict: does the filing delta explain abnormal returns?

Reads data/panel/study_panel.csv and reports, calibrated and honestly:
  1. Coverage.
  2. LEAKAGE CHECK — placebo [-5,-1] CAR must be ~0 and the signal must not
     predict it.
  3. IN-SAMPLE cross-section — where specs are defined (period < SPLIT_DATE).
  4. HOLDOUT cross-section — run ONCE on the untouched later period. The
     holdout coefficient is the honest headline.
  5. NET-OF-COST long-short spread on the holdout — the tradeable claim.

Discipline: the signal, controls, windows, and split are fixed here. Rerunning
this file does not re-optimize anything; the holdout stays a single look.

Usage: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/run_study.py
"""

from __future__ import annotations

import sys

import pandas as pd

from eqd.config import PANEL_DIR
from eqd.study.crosssection import cross_section, tidy
from eqd.study.portfolio import quantile_spread

STUDY_PANEL = PANEL_DIR / "study_panel.csv"
SIGNAL = "net_added"          # primary Lazy-Prices signal: net new risk sentences
CONTROLS = ["momentum", "beta"]
PRIMARY_Y = "market_model__car_0_5"
CAR_WINDOWS = ["market_model__car_0_5", "market_adjusted__car_0_5", "market_model__car_0_21"]
PLACEBO = "market_model__placebo_pre"
SPLIT_DATE = "2022-01-01"     # events before -> in-sample; on/after -> holdout


def _stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _run_cross_section(panel: pd.DataFrame, label: str) -> None:
    print(f"-- {label}: CAR ~ {SIGNAL} + {' + '.join(CONTROLS)} (clustered by t0) --")
    for y in CAR_WINDOWS:
        if panel[y].notna().sum() < 20:
            print(f"  {y:28} too few obs, skipped")
            continue
        res = cross_section(panel, y, [SIGNAL], CONTROLS, cluster="t0")
        s = tidy(res, [SIGNAL]).iloc[0]
        direction = "negative (expected)" if s["coef"] < 0 else "positive (UNEXPECTED)"
        print(f"  {y:28} coef={s['coef']:+.5f}  p={s['p']:.3f} {_stars(s['p']):3}  "
              f"({direction}, n={int(res.nobs)})")


def main() -> int:
    if not STUDY_PANEL.exists():
        print(f"Missing {STUDY_PANEL}. Run build_panel.py first.")
        return 1
    panel = pd.read_csv(STUDY_PANEL, parse_dates=["t0"])
    insample = panel[panel["t0"] < SPLIT_DATE]
    holdout = panel[panel["t0"] >= SPLIT_DATE]

    print("=" * 70)
    print("FILING-DELTA EVENT STUDY - honest verdict")
    print("=" * 70)
    n_priced = panel[PRIMARY_Y].notna().sum()
    print(f"Events: {len(panel):,} total ({n_priced:,} priced), {panel['ticker'].nunique()} names")
    print(f"  in-sample (t0 < {SPLIT_DATE}): {len(insample):,}   "
          f"holdout (t0 >= {SPLIT_DATE}): {len(holdout):,}\n")

    # 1. Leakage check (whole sample) ------------------------------------
    print("-- LEAKAGE CHECK (placebo pre-event window [-5,-1]) --")
    print(f"  mean placebo CAR = {panel[PLACEBO].mean():+.4%}  (want ~0; large => lookahead)")
    try:
        p_sig = tidy(cross_section(panel, PLACEBO, [SIGNAL], CONTROLS, cluster="t0"), [SIGNAL]).iloc[0]
        print(f"  signal->placebo coef = {p_sig['coef']:+.5f} (p={p_sig['p']:.3f}); "
              f"want insignificant\n")
    except ValueError as e:
        print(f"  (placebo regression skipped: {e})\n")

    # 2. In-sample (specs defined here) ----------------------------------
    _run_cross_section(insample, "IN-SAMPLE (specs defined)")
    print()

    # 3. Holdout (run once) — the honest headline ------------------------
    _run_cross_section(holdout, "HOLDOUT (run once)")
    print()

    # 4. Net-of-cost long-short spread on the holdout --------------------
    print(f"-- HOLDOUT LONG-SHORT (long low-signal / short high-signal, {PRIMARY_Y}) --")
    sp = quantile_spread(holdout, SIGNAL, PRIMARY_Y, n_groups=5)
    if "error" in sp:
        print(f"  {sp['error']}")
    else:
        print(f"  gross spread = {sp['gross_spread']:+.4%}   "
              f"round-trip cost = {sp['roundtrip_cost']:.4%}")
        print(f"  NET spread   = {sp['net_spread']:+.4%}   "
              f"(t={sp['t']}, p={sp['p']}, n={sp['n_low']}+{sp['n_high']})")
        verdict = ("survives costs" if sp["net_spread"] > 0 and sp["p"] < 0.10
                   else "does NOT survive costs / not significant")
        print(f"  -> {verdict}")

    print("\n(Coefs per 1 SD of standardized signal; CAR in return units. "
          "Lazy-Prices direction = negative coef.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
