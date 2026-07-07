"""Classify ADDED risk-factor sentences for every panel event, at full scale.

For each of the ~4,700 filing events in the study panel, re-diff Item 1A from
the cached filings and ask the LLM (GLM 5.2, free NVIDIA endpoint) to label each
added sentence: substantive new risk vs expansion vs rewording vs boilerplate.
Output feeds the pre-registered follow-up question documented in the README:
does the sharpened `n_substantive_added` beat the mechanical `net_added`?

Built for a long free-tier run:
  * resumable  — results append to data/panel/llm_classified.csv; on restart,
                 events already done (without error) are skipped
  * throttled  — a shared rate limiter keeps request starts under --rpm
  * threaded   — --workers concurrent events (diff CPU overlaps API latency)
  * defensive  — retries with backoff; a failed event is recorded and retried
                 on the next run instead of killing the job

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/classify_panel.py --limit 5   # sample test
  PYTHONPATH=src .venv/Scripts/python.exe scripts/classify_panel.py             # full run
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from eqd.config import FILINGS_DIR, PANEL_DIR                      # noqa: E402
from eqd.delta.diff import diff_text                               # noqa: E402
from eqd.delta.risk_factors import BASE_URL, classify_added        # noqa: E402
from eqd.ingest.sections import extract_sections                   # noqa: E402

OUT = PANEL_DIR / "llm_classified.csv"
FIELDS = [
    "accession", "ticker", "n_added_total", "n_sent_to_llm", "n_returned",
    "n_substantive_added", "n_new_substantive_risk", "n_expanded_existing",
    "n_reworded", "n_boilerplate", "error",
]
MAX_PASSAGES = 120          # cap per event (rare tail); keeps one call per event
MAX_CHARS = 450             # truncate pathological run-on sentences


def _read_item_1a(cik: str, accession: str) -> str | None:
    d = FILINGS_DIR / str(cik).zfill(10) / accession
    if not d.is_dir():
        return None
    htmls = [p for p in d.iterdir() if p.suffix.lower() in (".htm", ".html")]
    if not htmls:
        return None
    return extract_sections(htmls[0].read_text(encoding="utf-8", errors="replace")).get("item_1a")


class RateLimiter:
    """Shared token-spacing limiter: at most `rpm` request starts per minute."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)


def classify_event(row, client: OpenAI, limiter: RateLimiter) -> dict:
    out = {f: "" for f in FIELDS}
    out["accession"], out["ticker"] = row.accession, row.ticker

    prior_1a = _read_item_1a(row.cik, row.prior_accession)
    cur_1a = _read_item_1a(row.cik, row.accession)
    if not prior_1a or not cur_1a:
        out["error"] = "missing cached section text"
        return out

    added = diff_text(prior_1a, cur_1a).added
    out["n_added_total"] = len(added)
    passages = [" ".join(p.split())[:MAX_CHARS] for p in added][:MAX_PASSAGES]
    out["n_sent_to_llm"] = len(passages)
    if not passages:                                    # nothing added: zero signal, no API call
        for f in FIELDS[4:-1]:
            out[f] = 0
        return out

    last_err = "unknown"
    for attempt in range(4):
        limiter.wait()
        try:
            rows = classify_added(passages, client=client)
            counts = {"new_substantive_risk": 0, "expanded_existing_risk": 0,
                      "reworded_same_meaning": 0, "boilerplate_or_reorder": 0}
            for r in rows:
                counts[r["category"]] += 1
            out["n_returned"] = len(rows)
            out["n_new_substantive_risk"] = counts["new_substantive_risk"]
            out["n_expanded_existing"] = counts["expanded_existing_risk"]
            out["n_reworded"] = counts["reworded_same_meaning"]
            out["n_boilerplate"] = counts["boilerplate_or_reorder"]
            out["n_substantive_added"] = (
                counts["new_substantive_risk"] + counts["expanded_existing_risk"]
            )
            return out
        except Exception as exc:  # noqa: BLE001 — rate limits / transient API errors
            last_err = f"{type(exc).__name__}: {str(exc)[:120]}"
            time.sleep(8.0 * (attempt + 1))
    out["error"] = last_err
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="classify only the first N pending events")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rpm", type=float, default=30.0)
    args = ap.parse_args()

    if not os.getenv("NVIDIA_API_KEY"):
        raise SystemExit("NVIDIA_API_KEY not set in .env")

    panel = pd.read_csv(PANEL_DIR / "study_panel.csv")
    done: set[str] = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(prev.loc[prev["error"].fillna("") == "", "accession"])
    pending = panel[~panel["accession"].isin(done)]
    if args.limit:
        pending = pending.head(args.limit)
    print(f"events total={len(panel)}  done={len(done)}  this run={len(pending)}", flush=True)
    if pending.empty:
        return

    client = OpenAI(base_url=BASE_URL, api_key=os.getenv("NVIDIA_API_KEY"), timeout=180.0)
    limiter = RateLimiter(args.rpm)
    write_lock = threading.Lock()
    new_file = not OUT.exists()
    fh = open(OUT, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        fh.flush()

    n_done = 0
    t0 = time.time()

    def work(row) -> None:
        nonlocal n_done
        res = classify_event(row, client, limiter)
        with write_lock:
            writer.writerow(res)
            fh.flush()
            n_done += 1
            if n_done % 25 == 0 or n_done == len(pending):
                rate = n_done / max(time.time() - t0, 1)
                eta_min = (len(pending) - n_done) / max(rate, 1e-9) / 60
                print(f"  {n_done}/{len(pending)}  ({rate*60:.1f}/min, eta {eta_min:.0f} min)", flush=True)

    rows = list(pending.itertuples(index=False))
    threads: list[threading.Thread] = []
    sem = threading.Semaphore(args.workers)

    def runner(row) -> None:
        with sem:
            work(row)

    for row in rows:
        t = threading.Thread(target=runner, args=(row,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    fh.close()
    errs = pd.read_csv(OUT)
    n_err = int((errs["error"].fillna("") != "").sum())
    print(f"finished. rows={len(errs)}  errors={n_err}  (re-run to retry errors)", flush=True)


if __name__ == "__main__":
    main()
