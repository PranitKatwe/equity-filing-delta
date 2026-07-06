"""Demo: run the diff-grounded LLM classifier on one real filing pair.

Downloads a company's two most recent 10-Ks, diffs Item 1A, and asks the LLM
to classify each ADDED sentence — showing which are substantive new risks vs
boilerplate. This is the sharpened signal on top of the mechanical diff.

Requires ANTHROPIC_API_KEY in .env. Runs on your key and incurs cost
(one Opus call over the added passages for the chosen ticker).

Usage: PYTHONPATH=src .venv/Scripts/python.exe scripts/classify_sample.py [TICKER]
"""

import sys
import textwrap

from eqd.delta.diff import diff_text
from eqd.delta.risk_factors import classify_diff
from eqd.ingest.edgar import download_document, list_filings
from eqd.ingest.sections import extract_sections
from eqd.universe import sample


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    cik = sample([ticker])["cik"].iloc[0]

    tenks = list_filings(cik, forms=("10-K",), since="2015-01-01")
    prior_f, curr_f = tenks.iloc[-2], tenks.iloc[-1]
    print(f"{ticker}: classifying added Item 1A passages, "
          f"{prior_f['filing_date']} -> {curr_f['filing_date']}\n")

    def item1a(f):
        html = download_document(cik, f["accession"], f["primary_document"]).read_text(
            encoding="utf-8", errors="replace"
        )
        return extract_sections(html)["item_1a"]

    d = diff_text(item1a(prior_f), item1a(curr_f))
    print(f"mechanical diff: {len(d.added)} added passages -> asking the LLM...\n")

    feats = classify_diff(d)
    print(f"SHARPENED SIGNAL: n_substantive_added = {feats['n_substantive_added']} "
          f"(vs mechanical n_added = {len(d.added)})")
    print(f"  new_substantive_risk = {feats['n_new_substantive_risk']}, "
          f"boilerplate/reworded = {feats['n_boilerplate_added']}\n")

    print("--- sample classifications ---")
    for r in feats["classifications"][:12]:
        passage = textwrap.shorten(" ".join(d.added[r["index"]].split()), 90, placeholder=" …")
        print(f"  [{r['category']:22}] {passage}")


if __name__ == "__main__":
    main()
