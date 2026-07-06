"""Fetch the Loughran-McDonald master dictionary and cache category word lists.

LM (Notre Dame SRAF) is the standard finance sentiment lexicon. We cache each
category (negative, uncertainty, litigious, positive, constraining) as a plain
word list under data/lm/, used by delta/tone.py as a transparent, non-LLM tone
baseline that cross-checks the LLM.

Source dictionary: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
(fetched here from a public CSV mirror; override with --url for the official file).

Usage: PYTHONPATH=src .venv/Scripts/python.exe scripts/fetch_lm_dictionary.py
"""

from __future__ import annotations

import argparse
import io
import sys

import httpx
import pandas as pd

from eqd.config import DATA

DEFAULT_URL = (
    "https://raw.githubusercontent.com/james-pavlicek/"
    "algorithmic-trading-with-artificial-intelligence/main/"
    "Loughran-McDonald_MasterDictionary.csv"
)
LM_DIR = DATA / "lm"
CATEGORIES = ["Negative", "Uncertainty", "Litigious", "Positive", "Constraining"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()

    print(f"Downloading LM master dictionary from {args.url[:60]}...")
    resp = httpx.get(args.url, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    print(f"  {len(df):,} words, {len(df.columns)} columns")

    LM_DIR.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        # A word is in a category iff its column value is > 0 (the year it was added).
        words = sorted(df.loc[df[cat] > 0, "Word"].astype(str).str.upper())
        (LM_DIR / f"{cat.lower()}.txt").write_text("\n".join(words), encoding="utf-8")
        print(f"  {cat.lower():14} {len(words):5} words -> data/lm/{cat.lower()}.txt")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
