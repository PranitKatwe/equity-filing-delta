"""Compute the embedding-change signal for every panel event, from cached filings.

For each event, embed the current and prior Item 1A (bge-small via fastembed,
local CPU, free) and record emb_delta = cosine distance between the two mean-
pooled section vectors. Each filing is embedded once per company even when it
appears as both "current" and "prior" across events.

Resumable: results append to data/panel/embed_features.csv one company at a
time; rerunning skips companies already done.

Usage: PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/build_embed_features.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pandas as pd

from eqd.config import FILINGS_DIR, PANEL_DIR
from eqd.delta.embed import embed_delta, get_model, section_vector
from eqd.ingest.sections import extract_sections

OUT = PANEL_DIR / "embed_features.csv"
CACHE_DIR = str(Path(__file__).resolve().parents[1] / "data" / "fastembed_cache")
FIELDS = ["ticker", "accession", "emb_delta"]


def _read_item_1a(cik: str, accession: str) -> str | None:
    d = FILINGS_DIR / str(cik).zfill(10) / accession
    if not d.is_dir():
        return None
    htmls = [p for p in d.iterdir() if p.suffix.lower() in (".htm", ".html")]
    if not htmls:
        return None
    html = htmls[0].read_text(encoding="utf-8", errors="replace")
    return extract_sections(html).get("item_1a")


def main() -> None:
    panel = pd.read_csv(PANEL_DIR / "study_panel.csv")
    done = set(pd.read_csv(OUT)["ticker"]) if OUT.exists() else set()
    todo = [t for t in panel["ticker"].unique() if t not in done]
    print(f"{len(todo)} companies to embed ({len(done)} already done)")
    if not todo:
        return

    model = get_model(cache_dir=CACHE_DIR)
    t_start = time.time()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            w.writeheader()
        for i, ticker in enumerate(todo, 1):
            rows = panel[panel["ticker"] == ticker]
            cik = rows["cik"].iloc[0]
            accessions = set(rows["accession"]) | set(rows["prior_accession"])
            vecs = {}
            for acc in accessions:
                text = _read_item_1a(cik, acc)
                if text:
                    vecs[acc] = section_vector(text, model)
            n_ok = 0
            for r in rows.itertuples(index=False):
                pv, cv = vecs.get(r.prior_accession), vecs.get(r.accession)
                if pv is None or cv is None:
                    continue
                w.writerow({"ticker": ticker, "accession": r.accession,
                            "emb_delta": f"{embed_delta(pv, cv):.6f}"})
                n_ok += 1
            f.flush()
            rate = i / (time.time() - t_start)
            eta_min = (len(todo) - i) / rate / 60 if rate > 0 else 0
            print(f"[{i}/{len(todo)}] {ticker}: {n_ok}/{len(rows)} events  "
                  f"(eta {eta_min:.0f} min)", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
