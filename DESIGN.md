# DESIGN.md — Filing-Delta Event Signal (Equities)

An end-to-end pipeline that extracts **structured year-over-year changes** from SEC filings (new/removed risk factors, guidance-language shifts, management-tone changes) using an LLM, assembles them into a **point-in-time** feature panel, and tests whether those textual deltas relate to subsequent returns under a **rigorous, no-lookahead event-study harness** — costs included, out-of-sample, against an honest null.

> **Read first — three principles that define this project.**
> 1. **The harness is the crown jewel, not the model.** In equities, anyone can fit something that looks like alpha in-sample; credibility lives entirely in the methodology — point-in-time data, no lookahead, transaction costs, a real benchmark, and a holdout. If the headline is "I built a model that predicts returns," the project has already failed. The headline is "I built the harness that says, honestly, whether this signal survives." A weak-but-honest result beats a strong-but-fake one.
> 2. **Lookahead is the silent killer.** The single thing that makes or breaks this project is aligning every feature and every return to information that was actually public at that instant. The event-time alignment (M0) is a hard gate; nothing else matters if it's wrong.
> 3. **Be honest about novelty.** That 10-K/10-Q text *changes* predict returns is a documented academic result (Cohen, Malloy & Nguyen, "Lazy Prices," *Journal of Finance* 2020 — most filing changes are negative and predict underperformance). You are **not** discovering this. The novel, hireable thing is the **end-to-end, LLM-powered, point-in-time-honest reproduction-and-extension machinery** on a seam few occupy (LLM extraction + market-data rigor). Claim that, and treat reproducing the known direction as a *sanity check*, not a finding.

---

## 0. Data-access reality (all free; the asymmetry shapes everything)

| Source | Cost / access | Use |
|---|---|---|
| **SEC EDGAR** — filings (10-K/10-Q/8-K), full-text search (2001+), submissions & XBRL REST APIs, bulk files | **Free**, requires a declared `User-Agent`; ≤10 req/s | Filing text (risk factors, MD&A), **as-filed financials with timestamps**, the event clock |
| EDGAR **acceptance datetime** per filing | Free | The instant info became public → the point-in-time anchor |
| **Stooq** (free bulk EOD CSV) / **Tiingo** free tier | Free | Daily adjusted prices for the universe + benchmark/sector ETFs |
| Benchmark/sector ETFs (SPY, sector SPDRs) | Free (same price sources) | Expected-return / abnormal-return models |
| Loughran-McDonald finance sentiment word lists (Notre Dame SRAF) | Free | Transparent, non-LLM tone baseline to cross-check the LLM |

**The asymmetry to internalize:** EDGAR makes the *hard* inputs free and unusually good — financials are served **as originally filed, with a filing timestamp**, so point-in-time (no-lookahead) reconstruction is free, the thing vendors normally charge for. The only gated equity data is clean *delisted* price history — and an **event study barely cares**, because you condition on a filing event and measure a short reaction, not a decades-long tradeable universe. So this project's free-data story is clean.

**Universe:** start with a few hundred liquid large-caps (e.g., current S&P 500 constituents) — enough filing events for statistics, clean prices, manageable scale. Note the mild survivorship caveat and move on; it's second-order for event studies.

---

## 1. The method (the intellectual core)

**Extract deltas, not levels.** For each company, diff consecutive same-type filings (10-K vs prior 10-K; 10-Q vs year-ago 10-Q) and characterize *what changed*:
- **Risk-factor deltas** — Item 1A risk factors added / removed / materially reworded (boilerplate reordering ≠ signal). Added risk factors are the canonical negative signal.
- **Guidance / forward-language shifts** — direction and hedging of forward-looking statements in MD&A.
- **Tone deltas** — year-over-year change in sentiment/uncertainty (LM dictionary as a transparent baseline; LLM as a richer, diff-grounded measure).

**Test with an event study, honestly.** Anchor each filing to the first session its information was tradeable, compute **abnormal returns** vs an expected-return model over event windows, then ask cross-sectionally whether the deltas explain abnormal returns **after costs, out-of-sample, against a placebo null**. The output is a *calibrated answer*, including "the effect is small / vanishes after costs," which is a legitimate and credible result.

---

## 2. Point-in-time event alignment (M0 — the make-or-break)

This is the project's spine; specify and unit-test it before anything else.

- Each EDGAR filing has an **acceptance datetime** (ET) = when the info went public.
- Define the event session `t0` = **the first trading session whose open occurs strictly after the acceptance datetime.** Consequences:
  - Filed after the 16:00 ET close (typical for 10-K/10-Q) → `t0` = next trading day.
  - Filed during a non-trading day/holiday → `t0` = next trading day.
  - Filed mid-session before close → conservatively still use the next session's open as the entry to avoid same-bar lookahead (document this choice; offer an intraday-aware variant as stretch).
- Measure returns **close-to-close from `t0`** (consistent convention, documented).
- **Every feature** in the panel must be computable using **only** filings/prices dated ≤ acceptance datetime. No restated financials, no future index membership, no same-bar entry.
- **Lookahead audit (automated, a hard gate):** a test that scans each feature's input timestamps and fails the build if any input post-dates the event anchor. Plus a **pre-event placebo window** `[-5,-1]`: if a "predictive" signal shows large abnormal returns *before* the event, that's leakage, not alpha.

---

## 3. Tech stack

- **Python** 3.11+
- **EDGAR access:** `edgartools` (or `sec-edgar-downloader`, or raw `httpx` with a declared `User-Agent`); respect the 10 req/s limit; cache raw filings to disk (SHA-256 dedup).
- **Text diff & sections:** `difflib`/`rapidfuzz` for alignment; section parsing for Item 1A / Item 7; Loughran-McDonald lists for the transparent tone baseline.
- **LLM extraction:** Anthropic SDK (`claude-opus-4-8` for materiality/tone classification of *diffs*; `claude-sonnet-4-6` for routine summarization). **Diff-grounded only** — the model classifies changes it is shown, it never free-associates new "changes."
- **Prices:** Stooq bulk CSV (via `pandas-datareader` or direct) and/or Tiingo free tier; `yfinance` fallback. Validate for bad ticks/spikes.
- **Panel & returns:** DuckDB + Parquet, polars/pandas.
- **Stats:** `statsmodels` (cross-sectional regressions, clustered/robust SE), `scipy`.
- **pytest** for the alignment audit and extraction gold set.
- *(Deploy is a weak fit here — see §9. The deliverable is a clean repo + a notebook/report, not a web app.)*

---

## 4. Repository structure

```
equity-filing-delta/
├── README.md                 # resume artifact: method, the no-lookahead design, the HONEST result, costs, caveats, Lazy-Prices framing
├── DESIGN.md                 # this file
├── pyproject.toml
├── .env.example
├── data/                     # gitignored
│   ├── filings/              # cached raw EDGAR filings
│   ├── prices/               # EOD parquet
│   └── panel/                # the point-in-time feature panel (parquet)
├── src/eqd/
│   ├── config.py
│   ├── universe.py           # constituent list + CIK<->ticker mapping (point-in-time-aware)
│   ├── ingest/
│   │   ├── edgar.py          # fetch filings + acceptance datetimes; cache + SHA-256 dedup
│   │   ├── sections.py       # extract Item 1A / Item 7 text blocks
│   │   └── prices.py         # Stooq/Tiingo EOD; benchmark + sector ETFs; tick validation
│   ├── eventtime.py          # THE SPINE: acceptance datetime -> t0; calendar; lookahead audit helpers
│   ├── delta/                # extraction (AI surface, but secondary to rigor)
│   │   ├── diff.py           # align consecutive filings; isolate changed passages
│   │   ├── risk_factors.py   # added/removed/material-change classification (LLM, diff-grounded)
│   │   ├── tone.py           # LM-dictionary delta + LLM tone delta; cross-check
│   │   └── panel.py          # assemble point-in-time feature rows {cik, accession, t0, features...}
│   ├── study/                # CROWN JEWEL — the harness
│   │   ├── abnormal.py       # expected-return models: market-adjusted, market-model (beta), sector-adjusted
│   │   ├── car.py            # CAR over windows [0,+1],[0,+5],[0,+21]; placebo [-5,-1]
│   │   ├── costs.py          # spread + impact proxy; net-of-cost returns
│   │   └── crosssection.py   # CAR ~ deltas + controls (size, B/M, momentum, filing type); robust/clustered SE
│   ├── narrate/
│   │   └── memo.py           # per-event "what changed + how the market reacted" memo (grounded, cited, no advice)
│   └── report/
│       └── notebook.ipynb    # the deliverable: figures, tables, the honest verdict
├── tests/
│   ├── test_eventtime.py     # after-close -> next session; holiday handling; THE alignment gate
│   ├── test_lookahead_audit.py # fails if any feature input post-dates its event anchor
│   ├── test_delta_gold.py    # extraction precision/recall vs hand-labeled changes
│   └── fixtures/             # a handful of real filing pairs + a tiny price slice
└── scripts/
    ├── build_panel.py        # ingest -> sections -> deltas -> point-in-time panel
    └── run_study.py          # panel + prices -> abnormal returns -> cross-section -> verdict
```

---

## 5. Delta extraction (the AI surface — kept honest)

- **Diff first, LLM second.** Mechanically isolate changed passages between consecutive filings; the LLM only ever sees and classifies *actual diffs*. This prevents the model from inventing changes and keeps every feature traceable to source text (provenance: accession + section + passage).
- **Risk-factor classification:** for each changed/added/removed risk factor, classify {new substantive risk | removed risk | boilerplate/reorder | reworded-same-meaning} and a short rationale. Only substantive changes become signal; boilerplate is dropped.
- **Tone delta:** compute the Loughran-McDonald negative/uncertainty proportion change YoY as a transparent baseline, **and** an LLM tone-shift score on the diffs; report both and their agreement (if they disagree wildly, trust the transparent one and investigate).
- **Output:** one point-in-time row per filing event with numeric/categorical delta features, each carrying provenance. No feature uses post-event information (enforced by §2).

---

## 6. The event-study harness (crown jewel)

- **Abnormal returns:** start with market-adjusted (`r_i − r_SPY`); add a market-model variant (estimate beta on a pre-event window, e.g., `[-120,-21]`) and a sector-adjusted variant. Report results under at least two models — a signal that only survives one is fragile.
- **CAR windows:** announcement `[0,+1]`, short drift `[0,+5]`, post-filing drift `[0,+21]`; **placebo `[-5,-1]`** for leakage detection.
- **Costs:** subtract a realistic round-trip (bid-ask proxy + simple impact); report gross **and** net. The net result is the one that counts.
- **Cross-section:** regress CAR on the delta features with controls (size, book-to-market, momentum, filing type, and earnings surprise if cheaply available), robust/clustered standard errors. Optionally form a long-short portfolio at filing dates (e.g., short "added risk factors," long "removed") and report net-of-cost performance with turnover.
- **Out-of-sample:** split by time — define signals/specs on an in-sample period, then run **once** on an untouched holdout. The holdout number is the headline.

---

## 7. Validation & evals

- **Alignment gate (`test_eventtime`)**: after-close filing → next session; holiday/weekend handling; the single most important test in the repo.
- **Lookahead audit (`test_lookahead_audit`)**: automated timestamp check; build fails on any post-event input. Plus the empirical placebo: near-zero abnormal returns in `[-5,-1]`.
- **Extraction gold set (`test_delta_gold`)**: ~15–20 hand-labeled filing pairs (substantive change: yes/no, per risk factor); report precision/recall; boilerplate must not be flagged as substantive.
- **Known-result sanity:** the harness should *directionally* reproduce the documented effect — added risk factors / increased negative tone associate with negative drift. If it can't reproduce a published direction in-sample, the pipeline is broken (not a discovery).
- **Anti-p-hacking discipline:** pre-register windows, signals, and controls **before** touching the holdout; the holdout is run once. State the number of specifications tried so the reader can judge multiple-testing.
- **Narrator grounding/no-advice:** every figure in a memo traces to a computed result; no buy/sell/hold language.

---

## 8. Milestones (open-ended, each independently demonstrable)

| # | Milestone | Done when |
|---|---|---|
| **M0** | **Event-time spike** | EDGAR filings + acceptance datetimes + EOD prices ingested for a small universe; `acceptance → t0` mapping implemented; `test_eventtime` and the lookahead audit pass. **The spine works.** |
| **M1** | **Delta extraction + gold set** | Risk-factor add/remove/material-change features extracted (diff-grounded LLM), validated on the gold set; point-in-time panel built with provenance. |
| **M2** | **Event-study harness** | Abnormal returns (≥2 expected-return models), CAR windows, placebo window, costs; all wired and reproducible from the panel. |
| **M3** | **The honest verdict** | Cross-section of CAR on deltas with controls, net of costs, on a held-out period; directional sanity vs the known result; **a calibrated yes/weak/no answer.** The payoff. |
| **M4** | **Tone/guidance deltas + narrator** | LM + LLM tone deltas added; per-event grounded memos generated. |
| **M5** *(stretch)* | **Breadth + portfolio sim** | Larger universe; sector-neutral long-short with turnover/costs; factor-model abnormal returns; lightweight results viewer. |

**First demonstrable target:** M0 + M1 + M3-lite — "from free EDGAR data, here is a point-in-time, no-lookahead test of whether new risk factors predict abnormal returns, net of costs." That, with an honest answer, is the portfolio centerpiece.

---

## 9. Honest risks & how to handle them

- **Lookahead is the killer.** Acceptance-datetime → next-session alignment, the automated audit, and the placebo window are non-negotiable. Most retail "alpha" is just leakage.
- **Multiple testing / p-hacking.** Pre-register specs; run the holdout once; disclose how many things you tried. A small honest effect is the win condition.
- **Costs flip signals.** Always report net-of-cost; a gross-only result is fiction for a tradeable claim.
- **Extraction noise.** Diff-ground the LLM; drop boilerplate; cross-check tone against the LM dictionary. Don't let the model invent changes.
- **Novelty honesty.** "Lazy Prices" already established the effect. Claim the machinery + the LLM extraction + the point-in-time rigor, and frame reproduction as a sanity check — not a discovery. Overclaiming here loses a sharp interviewer instantly.
- **Survivorship.** Minimal for an event study (you condition on filings), but build the universe/ticker map point-in-time-aware and note the caveat rather than ignoring it.
- **Price data quality.** Validate for bad ticks and spikes before computing returns; cross-check a sample against a second source.
- **Scope creep.** Open-ended ≠ unbounded. M3 (the honest verdict on risk-factor deltas) is a complete, defensible artifact; tone, portfolio sims, and breadth are bonus.

---

## 10. Deploy note (intentionally minimal)

Unlike the muni and MBS projects, the right final artifact here is a **clean repo + a reproducible notebook/report**, not a web app — equity research is judged on the analysis and its rigor, not a UI. If you want a public surface, ship a static results page (charts + the verdict + the methodology writeup); keep any LLM endpoint local/gated (it runs on your key). Frame it as a research artifact, not production infra.

---

## 11. Conventions for the coding agent

- **M0 first and absolutely.** Do not build extraction or the study until `test_eventtime` and the lookahead audit pass. The point-in-time alignment is the entire credibility of the project.
- **Diff-ground the LLM.** It classifies changes it is shown; it never generates "changes" from whole filings. Every feature traces to source text.
- **No-lookahead everywhere.** Features use only data dated ≤ the acceptance datetime; entry is the next session; no restated financials; no future index membership.
- **Report gross and net; run the holdout once.** Pre-register specs; disclose how many were tried.
- **Reproduce the known direction as a sanity check, not a finding.** If you can't, the pipeline is broken.
- **DuckDB/Parquet for the panel and prices; statsmodels for inference.** No in-memory full scans.
- **Narrator never computes and never advises.** Numbers come from result rows; outputs are descriptive.
- The README is the resume artifact: the method, the no-lookahead design and why it matters, the **honest** net-of-cost out-of-sample result, the caveats, and the "Lazy Prices" framing of what's known vs. what you built.
```
