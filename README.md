# equity-filing-delta

**A point-in-time, no-lookahead event-study harness that tests whether year-over-year changes in SEC filing text predict subsequent abnormal returns — costs included, out-of-sample, against an honest null.**

The pipeline extracts *structured deltas* from consecutive 10-K filings (new / removed / reworded risk factors, tone shifts), assembles them into a **point-in-time** feature panel, and measures whether those textual changes relate to returns under a rigorous event-study harness.

---

## In plain English (for non-technical readers)

**The question.** Every public US company files a yearly report with the SEC called a "10-K." It includes a "Risk Factors" section listing what could go wrong for the business. Companies reword that section a little every year: they add new risks, drop old ones, and rephrase others. We test a simple idea: **when a company changes that language a lot, does its stock move in a predictable way in the weeks afterward?**

**Why it might.** There is a well-known finding (nicknamed "Lazy Prices") that companies usually reuse last year's wording, so the changes they *do* make tend to carry real news, often bad news, that the market is slow to react to.

**How we measure it honestly.** A few disciplines separate real analysis from wishful thinking, and they are the whole point of this project:

- **We use "abnormal" returns, not raw returns.** If a stock rose 3% but the whole market rose 3% that week, the stock did nothing special. We always subtract what the stock would have done anyway, so we isolate the part that might be tied to the filing.
- **We never use information before it existed (no "lookahead").** Each filing has an exact public timestamp, and we only allow trading from the first market open *after* it. This sounds obvious, but it is the most common way people accidentally produce fake results.
- **We include trading costs.** An edge that vanishes once you pay to buy and sell is not a real edge.
- **We test on data we did not tune on (a "holdout"), and we look at it only once.** That is the honest final exam. If you keep tweaking until the numbers look good, you are just memorizing noise.

**What we found.** The effect is real but weak, and **it does not survive trading costs.** It leaned the expected direction only slightly, and it was not consistent across time periods.

**Why a "no" is the valuable result.** Anyone can produce an exciting "yes" by peeking at the answer, ignoring costs, and tuning until the data looks good. The hard and credible thing is a system that can honestly say "this does not hold up." That trustworthy machinery, not a magic trading signal, is what this project delivers. The two live tools at the top of the site let you see the underlying filing changes and the returns for any S&P 500 company.

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

**Avenue not claimed, but demonstrated.** `net_added` is a blunt mechanical count. The diff-grounded LLM classifier ([`risk_factors.py`](src/eqd/delta/risk_factors.py)) sharpens it into `n_substantive_added`: real new risks vs boilerplate, labeled per added sentence, seeing only the diffed passages. Sample runs (`scripts/classify_sample.py`, one Opus call per company):

| Company | 10-K | Added (mechanical) | Substantive (LLM) | Brand-new risks |
|---|---|---:|---:|---:|
| AAPL | 2025 | 51 | 35 | 16 |
| BA   | 2026 | 31 | 24 | 9  |
| NVDA | 2026 | 65 | 57 | 43 |
| JPM  | 2026 | 79 | 55 | 18 |

The labels are sensible on inspection: Apple's *"The risks and uncertainties described below are not exhaustive..."* is tagged `boilerplate_or_reorder`, while *"Beginning in the second quarter of 2025, new tariffs were announced on imports to the U.S."* is tagged `new_substantive_risk`. Roughly a quarter to a third of mechanically-added sentences are noise the classifier filters out.

The classifier was then run once over every company's latest 10-K (473 companies, 13,208 added sentences) on Claude Sonnet 5 via the Batch API ([`scripts/classify_batch.py`](scripts/classify_batch.py), about $8 one-time). The labels are cached as static JSON under [`docs/classified/`](docs/classified/) and browsable per company on the site, so visitors read a stored run rather than triggering paid model calls. Cross-model sanity check: on the four Opus-labeled names, Sonnet's substantive share agrees closely (NVDA 88% in both runs, and both models label the two Apple example sentences identically); the boundary between `new_substantive_risk` and `expanded_existing_risk` is more model-sensitive, which is why the signal counts both. Whether the sharpened signal survives the full harness (all 4,705 historical events, holdout, net of costs) remains the open, documented next step; it is not asserted as a result here.

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

pytest -q                                            # 42 tests: the alignment gate + everything else

python scripts/fetch_lm_dictionary.py                # cache LM tone word lists
python scripts/build_panel.py --limit 30             # delta features + CARs (drop --limit for full S&P 500)
python scripts/run_study.py                          # the honest verdict
```

`data/` (filings, prices, panel) is gitignored — everything reproduces from the sources above.

**Results surface (deployed on Vercel).** The deliverable is a research artifact, not production infra ([DESIGN §10](DESIGN.md)): a single self-contained static page ([`docs/index.html`](docs/index.html)) with two live grounded LLM tools on top and the honest verdict and charts below. Search **any of the 480 S&P 500 names**, pick one of its 10-K filings, and a Vercel Python serverless function ([`api/narrate.py`](api/narrate.py)) turns that event's *pre-computed* numbers into cited prose. It never computes a figure and never advises. It runs **GLM 5.2 via NVIDIA's OpenAI-compatible endpoint** ([build.nvidia.com](https://build.nvidia.com), a free key), so the deployment has **no per-call cost**. The grounding and abuse guard: the client sends **only an event ID**, which must be a real key in the pre-computed panel ([`api/_panel.py`](api/_panel.py), all 4,697 events, generated by [`scripts/build_web_panel.py`](scripts/build_web_panel.py)); the numbers are looked up server-side, so a public URL can't be coaxed into predicting, advising, or narrating spoofed data (plus a per-instance rate limit and warm-instance cache). The front-end search uses a lightweight static index ([`docs/panel_index.json`](docs/panel_index.json)). The same narrator runs offline too: `python scripts/narrate_event.py AAPL`.

**Grounded filing Q&A (RAG).** A second endpoint ([`api/ask.py`](api/ask.py)) answers plain-English questions about a company's most recent 10-K risk-factor changes — *"what new risks did Boeing add this year?"* — strictly from the **actual added / removed / reworded sentences** the harness diffs out of that filing ([`api/_passages.py`](api/_passages.py), generated by [`scripts/build_passages.py`](scripts/build_passages.py)). Same discipline as the narrator: the model is grounded only in real, extracted filing text, cites the accession, says "not covered in the filing changes" when the answer isn't there, and never predicts or advises. The question is free text but only steers *what is summarized* — the model's entire context is that one company's real diff passages (question length-capped, rate-limited, cached).

**Deploy it yourself.** Import the repo at [vercel.com/new](https://vercel.com/new); [`vercel.json`](vercel.json) serves `docs/` statically and wires the Python `/api/*` functions. No framework preset or build step needed. Add `NVIDIA_API_KEY` (free from [build.nvidia.com](https://build.nvidia.com)) as a project environment variable, and deploy. Regenerate the panel data any time with `python scripts/build_web_panel.py`. (The narrator provider is swappable via `EQD_LLM_BASE_URL` / `EQD_NARRATE_MODEL`, any OpenAI-compatible endpoint works.)

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
api/narrate.py          Vercel serverless narrator — looks up any real event server-side
api/ask.py              Vercel serverless grounded Q&A over real 10-K diff passages
api/_panel.py           generated: server-side trusted panel (all events, grounding + guard)
api/_passages.py        generated: real Item 1A diff passages per company (Q&A grounding)
tests/                  alignment gate, diff, returns engine, portfolio, tone, narrator, api, ask
scripts/                verify_acceptance_tz, fetch_lm_dictionary, build_panel, run_study,
                        classify_sample, classify_batch, narrate_event, build_web_panel, build_passages
docs/index.html         static page: grounded narrator + Q&A on top, verdict + charts below
docs/panel_index.json   generated: lightweight company/event index for the search UI
vercel.json             Vercel config (serves docs/ statically, wires /api/narrate + /api/ask)
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

Complete and reproducible end-to-end at full S&P 500 scale: M0 (event-time spine + ingest), M1 (mechanical diff + diff-grounded LLM classifier), M2 (event-study harness), M3 (holdout + net-of-cost verdict, run once), and M4 (Loughran-McDonald tone deltas). 4,705 events across 480 names; 42 tests. The honest headline is above. Open next steps: evaluate whether the LLM-sharpened `n_substantive_added` beats the mechanical count, and a sector-neutral portfolio with turnover. See [`DESIGN.md`](DESIGN.md) for the full methodology.