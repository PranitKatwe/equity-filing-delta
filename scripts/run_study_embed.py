"""Signal #2 through the same harness: the embedding-change signal, IN-SAMPLE ONLY.

Motivated by Didisheim, Kelly, Pourmohammadi & Tian (NBER w35093): text change
measured in embedding space, and the part of it not explained by firm
characteristics treated as the real signal. Two variants:

  emb_delta        cosine distance between consecutive Item 1A vectors
  emb_delta_resid  the same, minus what sector + section size predict,
                   fit on prior years only (no lookahead)

DISCIPLINE NOTE: the holdout (t0 >= 2022-01-01) was spent on the original
mechanical signal and is NOT touched here. Everything below is in-sample.
Judging this signal out-of-sample requires a fresh, later window, evaluated
once — documented as the next step, not performed here.

Usage: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/run_study_embed.py
"""

from __future__ import annotations

import sys

import pandas as pd

from eqd.config import PANEL_DIR
from eqd.study.crosssection import cross_section, tidy
from eqd.study.portfolio import quantile_spread
from eqd.study.residualize import expanding_residual

CONTROLS = ["momentum", "beta"]
PRIMARY_Y = "market_model__car_0_5"
CAR_WINDOWS = ["market_model__car_0_5", "market_adjusted__car_0_5", "market_model__car_0_21"]
PLACEBO = "market_model__placebo_pre"
SPLIT_DATE = "2022-01-01"  # in-sample only; the holdout stays untouched

# Expected sign if bigger meaning-change carries bad news the market is slow
# to price (the Lazy-Prices direction): negative.
SIGNALS = [("emb_delta", "neg"), ("emb_delta_resid", "neg")]


def _stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main() -> int:
    feats_path = PANEL_DIR / "embed_features.csv"
    if not feats_path.exists():
        print(f"Missing {feats_path}. Run build_embed_features.py first.")
        return 1
    panel = pd.read_csv(PANEL_DIR / "study_panel.csv", parse_dates=["t0"])
    feats = pd.read_csv(feats_path)
    panel = panel.merge(feats, on=["ticker", "accession"], how="left")
    panel["emb_delta_resid"] = expanding_residual(panel, "emb_delta")

    insample = panel[panel["t0"] < SPLIT_DATE]

    print("=" * 70)
    print("EMBEDDING-CHANGE SIGNAL - IN-SAMPLE ONLY (holdout already spent)")
    print("=" * 70)
    print(f"Events with emb_delta: {panel['emb_delta'].notna().sum():,} of {len(panel):,}"
          f"   with residual: {panel['emb_delta_resid'].notna().sum():,}")
    print(f"In-sample (t0 < {SPLIT_DATE}): {len(insample):,} events\n")

    print("-- LEAKAGE CHECK (signal must not predict the pre-event placebo) --")
    for sig, _ in SIGNALS:
        try:
            s = tidy(cross_section(insample, PLACEBO, [sig], CONTROLS, cluster="t0"),
                     [sig]).iloc[0]
            print(f"  {sig:16} -> placebo coef={s['coef']:+.5f} (p={s['p']:.3f}); "
                  f"want insignificant")
        except ValueError as e:
            print(f"  {sig:16} skipped ({e})")
    print()

    for sig, expect in SIGNALS:
        print(f"-- IN-SAMPLE: CAR ~ {sig} + {' + '.join(CONTROLS)} (clustered by t0) --")
        for y in CAR_WINDOWS:
            try:
                res = cross_section(insample, y, [sig], CONTROLS, cluster="t0")
            except ValueError as e:
                print(f"  {y:28} skipped ({e})")
                continue
            s = tidy(res, [sig]).iloc[0]
            got = "neg" if s["coef"] < 0 else "pos"
            verdict = "as expected" if got == expect else "OPPOSITE to expected"
            print(f"  {y:28} coef={s['coef']:+.5f}  p={s['p']:.3f} {_stars(s['p']):3}  "
                  f"(want {expect}, got {got}: {verdict}, n={int(res.nobs)})")
        print()

    print(f"-- IN-SAMPLE LONG-SHORT (long low / short high emb_delta_resid, {PRIMARY_Y}) --")
    sp = quantile_spread(insample, "emb_delta_resid", PRIMARY_Y, n_groups=5)
    if "error" in sp:
        print(f"  {sp['error']}")
    else:
        print(f"  gross spread = {sp['gross_spread']:+.4%}   net = {sp['net_spread']:+.4%}   "
              f"(t={sp['t']}, p={sp['p']})")

    print("\n(In-sample exploration only. No holdout claim is made for this signal; "
          "an honest test needs a fresh out-of-time window, looked at once.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
