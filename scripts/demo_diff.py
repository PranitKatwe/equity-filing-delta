"""Demo: real year-over-year risk-factor diff for one company.

Downloads the two most recent 10-Ks, extracts Item 1A from each, and shows
what the mechanical diff isolates — the exact passages the LLM will later
classify. Proves the sections + diff pipeline works on real filings.

Usage: PYTHONPATH=src .venv/Scripts/python.exe scripts/demo_diff.py [TICKER]
"""

import sys
import textwrap

from eqd.delta.diff import diff_text
from eqd.ingest.edgar import download_document, list_filings
from eqd.ingest.sections import extract_sections
from eqd.universe import sample


def clip(s: str, n: int = 220) -> str:
    return textwrap.shorten(" ".join(s.split()), width=n, placeholder=" …")


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    cik = sample([ticker])["cik"].iloc[0]

    tenks = list_filings(cik, forms=("10-K",), since="2015-01-01")
    if len(tenks) < 2:
        print(f"Need >=2 10-Ks for {ticker}; found {len(tenks)}.")
        return
    prior_f, curr_f = tenks.iloc[-2], tenks.iloc[-1]
    print(f"{ticker}: diffing 10-K {prior_f['filing_date']} -> {curr_f['filing_date']}\n")

    prior_html = download_document(cik, prior_f["accession"], prior_f["primary_document"]).read_text(
        encoding="utf-8", errors="replace"
    )
    curr_html = download_document(cik, curr_f["accession"], curr_f["primary_document"]).read_text(
        encoding="utf-8", errors="replace"
    )

    prior_1a = extract_sections(prior_html)["item_1a"]
    curr_1a = extract_sections(curr_html)["item_1a"]
    if not prior_1a or not curr_1a:
        print(f"Item 1A extraction failed (prior={bool(prior_1a)}, curr={bool(curr_1a)}).")
        return

    d = diff_text(prior_1a, curr_1a)
    print("Item 1A sizes:", f"{len(prior_1a):,} -> {len(curr_1a):,} chars")
    print("Diff summary:", d.summary(), "\n")

    for label, items in [("ADDED (candidate new risks)", d.added), ("REMOVED", d.removed)]:
        print(f"--- {label}: {len(items)} ---")
        for p in items[:4]:
            print("  •", clip(p))
        print()

    print(f"--- MODIFIED (reworded): {len(d.modified)} ---")
    for prior_m, curr_m, ratio in d.modified[:3]:
        print(f"  ~ (sim={ratio}) {clip(curr_m, 180)}")


if __name__ == "__main__":
    main()
