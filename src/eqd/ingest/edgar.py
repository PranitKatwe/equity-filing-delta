"""edgar.py — rate-limited SEC EDGAR access.

Two things we pull from EDGAR, both free:
  1. The *submissions* index per company -> every filing with its
     `acceptanceDateTime` (the instant it went public = the input to `t0`).
  2. The raw primary document text (cached, SHA-256 recorded) for later
     diffing/extraction.

SEC rules we respect: a declared User-Agent with contact info, and <=10
req/s (we throttle to 8).

Timezone (VERIFIED, critical): EDGAR's submissions API returns
`acceptanceDateTime` as "2023-11-02T22:08:27.000Z" — genuine **UTC**; the 'Z'
is correct. The filing's own SGML header states the same instant in Eastern
wall-clock (`<ACCEPTANCE-DATETIME>20231102180827` = 18:08 ET = 22:08 UTC in
EDT). We therefore keep the raw ISO-UTC string unchanged and let
`eventtime.to_utc` parse it as tz-aware UTC. Do NOT strip the 'Z' and re-localize
to Eastern — that injects a 4-5 hour lookahead error. Confirmed across
2023-2026 filings by `scripts/verify_acceptance_tz.py`.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx
import pandas as pd

from ..config import FILINGS_DIR, SEC_RATE_LIMIT_PER_SEC, require_user_agent

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVE_DOC_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
)

_MIN_INTERVAL = 1.0 / SEC_RATE_LIMIT_PER_SEC
_last_request = [0.0]


def _throttle() -> None:
    """Space requests to stay under the SEC rate ceiling."""
    wait = _MIN_INTERVAL - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()


def _get(url: str) -> httpx.Response:
    _throttle()
    resp = httpx.get(
        url,
        headers={"User-Agent": require_user_agent(), "Accept-Encoding": "gzip"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp


def fetch_submissions(cik10: str) -> dict:
    """Full submissions JSON for a company, with older-filing pages merged in.

    The `recent` block holds up to ~1000 filings; companies with more have
    additional pages listed under `filings.files`, which we fetch and stitch
    onto the recent arrays so callers see the complete history.
    """
    data = _get(SUBMISSIONS_URL.format(cik10=cik10)).json()
    recent = data["filings"]["recent"]
    frames = [pd.DataFrame(recent)]

    for extra in data["filings"].get("files", []):
        page = _get(f"https://data.sec.gov/submissions/{extra['name']}").json()
        frames.append(pd.DataFrame(page))

    data["_filings_df"] = pd.concat(frames, ignore_index=True)
    return data


def list_filings(
    cik10: str,
    forms: list[str] | None = ("10-K", "10-Q"),
    since: str | None = None,
) -> pd.DataFrame:
    """Tidy filing index for a company: one row per filing.

    Columns: cik, form, accession, filing_date, report_date,
             acceptance_datetime (naive ET string), primary_document.
    """
    sub = fetch_submissions(cik10)
    df = sub["_filings_df"].copy()

    out = pd.DataFrame(
        {
            "cik": cik10,
            "form": df["form"],
            "accession": df["accessionNumber"],
            "filing_date": pd.to_datetime(df["filingDate"], errors="coerce").dt.date,
            "report_date": pd.to_datetime(df["reportDate"], errors="coerce").dt.date,
            # Raw ISO-UTC string (keep the 'Z'); to_utc parses it as tz-aware UTC.
            "acceptance_datetime": df["acceptanceDateTime"],
            "primary_document": df["primaryDocument"],
        }
    )

    if forms is not None:
        out = out[out["form"].isin(list(forms))]
    if since is not None:
        cutoff = pd.Timestamp(since).date()
        out = out[out["filing_date"] >= cutoff]

    return out.sort_values(["filing_date", "form"]).reset_index(drop=True)


def document_url(cik10: str, accession: str, primary_document: str) -> str:
    return ARCHIVE_DOC_URL.format(
        cik_int=int(cik10),
        acc_nodash=accession.replace("-", ""),
        doc=primary_document,
    )


def download_document(
    cik10: str, accession: str, primary_document: str, *, refresh: bool = False
) -> Path:
    """Fetch a filing's primary document, cache to disk, record SHA-256.

    Layout: data/filings/<cik10>/<accession>/<primary_document> plus a
    sibling meta.json {sha256, bytes, url}. Cached hits skip the network.
    """
    dest_dir = FILINGS_DIR / cik10 / accession
    dest = dest_dir / primary_document
    meta = dest_dir / "meta.json"
    if dest.exists() and meta.exists() and not refresh:
        return dest

    url = document_url(cik10, accession, primary_document)
    content = _get(url).content
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    meta.write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "url": url,
            },
            indent=2,
        )
    )
    return dest
