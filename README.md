# equity-filing-delta

**A point-in-time, no-lookahead event-study harness that tests whether year-over-year changes in SEC filing text predict subsequent abnormal returns — costs included, out-of-sample, against an honest null.**

The pipeline extracts *structured deltas* from consecutive 10-K filings (new / removed / reworded risk factors, tone shifts), assembles them into a **point-in-time** feature panel, and measures whether those textual changes relate to returns under a rigorous event-study harness.

---

## The one thing to understand about this project

**The harness is the crown jewel, not the signal.** In equities, anyone can fit something that looks like alpha in-sample. Credibility lives entirely in the methodology: point-in-time data, no lookahead, transaction costs, a real benchmark, and a holdout that is looked at once. So the headline here is *not* "a model that predicts returns" — it's **"a harness that says, honestly, whether a signal survives."** A weak-but-honest result beats a strong-but-fake one, every time.

**On novelty (stated plainly).** That 10-K text *changes* predict returns is a documented result — Cohen, Malloy & Nguyen, ["Lazy Prices," *Journal of Finance* 2020](https://doi.org/10.1111/jofi.12885) (most filing changes are negative and predict underperformance). This project does **not** claim to discover that. The contribution is the **end-to-end, point-in-time-honest reproduction-and-extension machinery** on a seam few occupy: LLM-ready text-delta extraction wired to market-data rigor. Reproducing the known direction is treated as a **sanity check**, not a finding.

---

## What "no lookahead" actually means here

Lookahead — using information you couldn't really have had yet — is the silent killer of retail "alpha." The entire credibility of this project rests on one hard gate, built and tested before anything else:

- Every EDGAR filing has an **acceptance datetime** (the instant it became public). `t0` = **the first trading session whose 9:30 ET open is strictly after that instant.** After-close filing → next day; weekend/holiday → next trading day; pre-market filing → *same* day (tradeable at the open). Verified against real filing headers ([`scripts/verify_acceptance_tz.py`](scripts/verify_acceptance_tz.py) — this caught and killed a latent 4–5-hour timezone bug).
- Every feature is computable from filings dated **≤ the acceptance datetime** — enforced in code by `assert_no_lookahead`, not merely promised in prose.
- A **placebo window `[−5, −1]`** (pre-event) is measured on every event. If a "predictive" signal shows abnormal returns *before* the filing, that's leakage, not alpha.

This is codified in [`src/eqd/eventtime.py`](src/eqd/eventtime.py) and guarded by 18 tests in [`tests/test_eventtime.py`](tests/test_eventtime.py) — the single most important test in the repo.

---

## The result (honest, calibrated)

Full S&P 500 run: **4,705 filing events across 480 names**, split in-sample (t0 < 2022-01-01, 2,529 events) vs an untouched **holdout** (2022+, 2,176 events), reported gross **and** net of a round-trip cost.

**The calibrated answer: the effect is weak and does not survive as a tradeable signal.** And that is a legitimate, credible result — the design is explicit that *"the effect is small / vanishes after costs"* is a real finding, and that a weak-but-honest answer beats a strong-but-fake one.

- **No lookahead.** The signal does not predict the pre-event placebo `[−5,−1]` window (coef p = 0.65). (A uniform +0.34% *level* in the placebo window is not signal-driven — it is absorbed by the regression intercept — so it is a model-level effect, not leakage.)
- **Holdout matches the Lazy-Prices direction, marginally.** More *net-added* risk-factor language → **lower** subsequent abnormal returns: CAR`[0,+21]` coef −0.004 (**p = 0.03**), CAR`[0,+5]` p = 0.06.
- **…but the sign is unstable.** In-sample the *same* signal is **positive** (wrong direction, CAR`[0,+21]` p = 0.01). A signal that flips sign across periods is not robust.
- **It does not survive costs.** The net-of-cost long-short spread is **+0.07%** after a 0.20% round trip — economically negligible and statistically insignificant (t = 0.66, **p = 0.51**).
- **Transparent tone deltas** (Loughran-McDonald) are likewise insignificant on the holdout — a useful cross-check that the mechanical result isn't an artifact of one signal construction.

The instructive part: a 30-name subsample showed a **strong, significant** holdout effect; the full 500 **washes it out** — a textbook demonstration of why out-of-sample, full-universe, net-of-cost discipline exists. The value of this project is the harness that says so honestly.

**Avenue not claimed.** `net_added` is a blunt mechanical count. The diff-grounded LLM classifier ([`risk_factors.py`](src/eqd/delta/risk_factors.py)) sharpens it into `n_substantive_added` (real new risks vs boilerplate); whether that survives is an open, documented next step — not a result asserted here.

Reproduce end-to-end: `build_panel.py` → `run_study.py` (see below).

---

## Method

1. **Extract deltas, not levels.** Diff consecutive 10-Ks' **Item 1A (Risk Factors)** and **Item 7 (MD&A)**. A sentence-level diff with C-accelerated similarity reconciliation isolates *added / removed / reworded* passages while neutralizing spurious churn from paragraph re-chunking and reordering ([`src/eqd/delta/diff.py`](src/eqd/delta/diff.py)). Diff **first** — so any later LLM classification only ever sees real changes and can never invent them.
2. **Transparent tone baseline.** Loughran-McDonald negative/uncertainty proportions and their YoY deltas ([`src/eqd/delta/tone.py`](src/eqd/delta/tone.py)) — a non-LLM cross-check.
3. **Abnormal returns, ≥2 models.** Market-adjusted, sector-adjusted, and a market-model with β estimated on a pre-event `[−120,−21]` window ([`src/eqd/study/abnormal.py`](src/eqd/study/abnormal.py)).
4. **CARs + placebo + costs.** Windows `[0,+1] / [0,+5] / [0,+21]`, placebo `[−5,−1]`, momentum control, net-of-cost returns ([`car.py`](src/eqd/study/car.py), [`costs.py`](src/eqd/study/costs.py)).
5. **Honest test.** Cross-section of CAR on the delta signal with controls and clustered SEs; a net-of-cost long-short spread; an in-sample/holdout split run once ([`crosssection.py`](src/eqd/study/crosssection.py), [`portfolio.py`](src/eqd/study/portfolio.py)).

---

## Data (all free)

| Source | Use |
|---|---|
| **SEC EDGAR** REST APIs | Filings + **acceptance datetimes** (the event clock), as-filed text |
| **Yahoo Finance** (`yfinance`) | Daily split/dividend-adjusted prices; benchmark + sector SPDR ETFs |
| **Loughran-McDonald** master dictionary (Notre Dame SRAF) | Transparent tone baseline |

EDGAR serves the *hard* inputs — financials **as originally filed, with a timestamp** — so point-in-time reconstruction is free, which is exactly what a no-lookahead study needs. (Stooq was the original price source but added a JS anti-bot challenge; `yfinance` is the design's listed fallback.)

---

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use source .venv/bin/activate on Unix
pip install -e .                                     # installs deps from pyproject.toml
cp .env.example .env                                 # then set SEC_USER_AGENT="<app> <your-email>"

pytest -q                                            # 34 tests: the alignment gate + everything else

python scripts/fetch_lm_dictionary.py                # cache LM tone word lists
python scripts/build_panel.py --limit 30             # delta features + CARs (drop --limit for full S&P 500)
python scripts/run_study.py                          # the honest verdict
```

`data/` (filings, prices, panel) is gitignored — everything reproduces from the sources above.

**Results surface.** The deliverable is a research artifact, not production infra ([DESIGN §10](DESIGN.md)): a static results page ([`docs/index.html`](docs/index.html)) with the honest verdict and charts, servable free via GitHub Pages, plus an offline, key-gated **narrator** ([`narrate/memo.py`](src/eqd/narrate/memo.py)) that turns computed result rows into grounded, cited prose — it never computes a number and never advises. Try it: `python scripts/narrate_event.py AAPL`.

---

## Repository map

```
src/eqd/
  eventtime.py          THE SPINE: acceptance datetime -> t0, no-lookahead audit
  universe.py           S&P 500 constituents + ticker<->CIK
  ingest/edgar.py       rate-limited EDGAR access; filings + acceptance; cache + SHA-256
  ingest/prices.py      yfinance adjusted EOD + benchmark/sector ETFs; tick validation
  ingest/sections.py    extract Item 1A / Item 7 from filing HTML
  delta/diff.py         sentence-level YoY diff (added/removed/modified) — grounds the LLM
  delta/tone.py         Loughran-McDonald tone deltas
  delta/panel.py        point-in-time feature panel (assert_no_lookahead enforced)
  study/abnormal.py     expected-return models (market/sector/market-model)
  study/car.py          CAR windows + placebo + momentum
  study/costs.py        net-of-cost returns
  study/crosssection.py CAR ~ signal + controls, clustered SE
  study/portfolio.py    net-of-cost long-short spread
  narrate/memo.py       grounded, cited, no-advice narrator (never computes/advises)
tests/                  alignment gate, diff, returns engine, portfolio, tone, narrator
scripts/                verify_acceptance_tz, fetch_lm_dictionary, build_panel,
                        run_study, classify_sample, narrate_event
docs/index.html         static results page (the honest verdict + charts) — GitHub Pages
```

---

## Honest risks & how they're handled

- **Lookahead** — acceptance→next-session alignment, an automated audit, and the placebo window. Non-negotiable.
- **Multiple testing** — specs pre-registered in `run_study.py`; the holdout is a single look.
- **Costs flip signals** — every tradeable claim is reported net of a round-trip cost.
- **Extraction noise** — the LLM is diff-grounded; boilerplate is separated from substantive change; tone is cross-checked against the transparent LM dictionary.
- **Survivorship** — an event study conditions on filings, so this is second-order; the universe keeps `date_added` for a point-in-time membership filter.

---

## Status

Complete and reproducible end-to-end at full S&P 500 scale: M0 (event-time spine + ingest), M1 (mechanical diff + diff-grounded LLM classifier), M2 (event-study harness), M3 (holdout + net-of-cost verdict, run once), and M4 (Loughran-McDonald tone deltas). 4,705 events across 480 names; 34 tests. The honest headline is above. Open next steps: evaluate whether the LLM-sharpened `n_substantive_added` beats the mechanical count, and a sector-neutral portfolio with turnover. See [`DESIGN.md`](DESIGN.md) for the full methodology.