"""Verify EDGAR's acceptanceDateTime timezone against the filing's own header.

The submissions API returns e.g. "2023-11-02T22:08:27.000Z" — genuine UTC.
Each filing's complete-submission .txt starts with an SGML header line
    <ACCEPTANCE-DATETIME>20231102180827
in US/Eastern wall-clock. We convert the API's UTC value into Eastern and
check it equals the header digit-for-digit. If so, our convention (parse the
API string as tz-aware UTC) is correct.

Run: PYTHONPATH=src .venv/Scripts/python.exe scripts/verify_acceptance_tz.py
"""

import re
import sys

import httpx
import pandas as pd

from eqd.config import FILING_TZ, require_user_agent
from eqd.ingest.edgar import list_filings

CIK = "0000320193"  # Apple


def header_acceptance(cik10: str, accession: str) -> str:
    acc_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc_nodash}/{accession}.txt"
    # Range request: the header is in the first ~1KB — don't pull the whole filing.
    resp = httpx.get(
        url,
        headers={"User-Agent": require_user_agent(), "Range": "bytes=0-1200"},
        timeout=30,
    )
    resp.raise_for_status()
    m = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", resp.text)
    return m.group(1) if m else "(not found)"


def main() -> int:
    filings = list_filings(CIK, forms=("10-K", "10-Q"), since="2022-01-01")
    print(f"Checking {len(filings)} Apple filings...\n")
    ok = True
    for _, row in filings.iterrows():
        # API value is UTC; convert to Eastern wall-clock and format like the header.
        api_et = (
            pd.Timestamp(row["acceptance_datetime"]).tz_convert(FILING_TZ).strftime("%Y%m%d%H%M%S")
        )
        hdr = header_acceptance(CIK, row["accession"])         # 20231102180827 (ET)
        match = api_et == hdr
        ok &= match
        flag = "OK " if match else "MISMATCH"
        print(f"  [{flag}] {row['form']:5} {row['accession']}  api->ET={api_et}  header_ET={hdr}")

    print()
    if ok:
        print("VERIFIED: API acceptanceDateTime (UTC) == header (ET) for every filing.")
        print("Convention is correct: parse the raw ISO-UTC string as tz-aware UTC.")
        return 0
    print("WARNING: mismatch found — the tz handling in edgar.py is WRONG. Do not proceed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
