"""delta/panel.py — assemble the point-in-time feature panel.

One row per 10-K filing event: diff this year's Item 1A (Risk Factors) against
the prior year's 10-K and emit delta features, tagged with the event anchor
(acceptance datetime) and t0. Every feature is computable from filings dated
<= the current acceptance, so each row is point-in-time honest — enforced here
by assert_no_lookahead, not merely asserted in prose.

Features (per event):
  n_added        # sentences that are genuinely new         (the Lazy-Prices signal)
  n_removed      # sentences dropped
  n_modified     # sentences reworded
  net_added      n_added - n_removed  (net growth in risk language)
  doc_similarity paragraph/sentence-level similarity vs prior (1.0 = identical)
  item1a_chars   size of the current risk-factor section
Provenance: accession + prior_accession.
"""

from __future__ import annotations

import pandas as pd

from ..eventtime import TradingCalendar, assert_no_lookahead
from ..ingest.edgar import download_document, list_filings
from ..ingest.prices import SECTOR_ETFS
from ..ingest.sections import extract_sections
from .diff import diff_text
from .tone import tone_features

_FEATURE_COLS = [
    "n_added",
    "n_removed",
    "n_modified",
    "net_added",
    "doc_similarity",
    "item1a_chars",
]


def _read_item_1a(cik: str, accession: str, primary_document: str) -> str | None:
    path = download_document(cik, accession, primary_document)
    html = path.read_text(encoding="utf-8", errors="replace")
    return extract_sections(html)["item_1a"]


def build_company_panel(
    cik: str,
    ticker: str,
    sector: str,
    cal: TradingCalendar,
    *,
    since: str = "2015-01-01",
) -> list[dict]:
    """Point-in-time delta rows for one company's consecutive 10-Ks."""
    tenks = list_filings(cik, forms=("10-K",), since=since).sort_values("filing_date")
    sector_etf = SECTOR_ETFS.get(sector)

    rows: list[dict] = []
    prior = None  # (accession, acceptance, item_1a_text)
    for f in tenks.itertuples(index=False):
        cur_1a = _read_item_1a(cik, f.accession, f.primary_document)

        if prior is not None and prior[2] and cur_1a:
            prior_acc, prior_accept, prior_1a = prior
            # No-lookahead gate: both filings must predate this event's anchor.
            assert_no_lookahead(
                [prior_accept, f.acceptance_datetime],
                f.acceptance_datetime,
                label=f"{ticker} 10-K {f.accession}",
            )
            d = diff_text(prior_1a, cur_1a)
            row = {
                "ticker": ticker,
                "cik": cik,
                "sector": sector,
                "sector_etf": sector_etf,
                "accession": f.accession,
                "prior_accession": prior_acc,
                "acceptance_datetime": f.acceptance_datetime,
                "filing_date": f.filing_date,
                "t0": cal.t0(f.acceptance_datetime).date(),
                "n_added": len(d.added),
                "n_removed": len(d.removed),
                "n_modified": len(d.modified),
                "net_added": len(d.added) - len(d.removed),
                "doc_similarity": round(d.doc_similarity, 4),
                "item1a_chars": len(cur_1a),
            }
            row.update(tone_features(prior_1a, cur_1a))  # LM tone levels + YoY deltas
            rows.append(row)

        if cur_1a:  # only advance prior when we have usable text
            prior = (f.accession, f.acceptance_datetime, cur_1a)

    return rows


def build_panel(universe: pd.DataFrame, cal: TradingCalendar, *, since: str = "2015-01-01") -> pd.DataFrame:
    """Build the panel across a universe DataFrame [ticker, sector, cik, ...]."""
    all_rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    for i, r in enumerate(universe.itertuples(), 1):
        try:
            all_rows.extend(build_company_panel(r.cik, r.ticker, r.sector, cal, since=since))
        except Exception as e:  # noqa: BLE001 — one bad filer shouldn't sink the panel
            failures.append((r.ticker, str(e)[:80]))
        if i % 10 == 0:
            print(f"  panel {i}/{len(universe)}  ({len(all_rows):,} events)", flush=True)
    if failures:
        print(f"  panel failures ({len(failures)}): {failures[:5]}")
    return pd.DataFrame(all_rows)
