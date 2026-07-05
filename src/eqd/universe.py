"""universe.py — the investable set + ticker<->CIK mapping.

We start from the current S&P 500 constituents (a few hundred liquid
large-caps: enough filing events for statistics, clean prices, manageable
scale). CIK is the SEC's permanent company identifier and the key EDGAR
indexes filings by, so ticker->CIK is the bridge from "a company" to
"its filings."

Survivorship caveat (documented, not ignored): the *current* constituent
list omits companies that left the index. For an event study this is
second-order — we condition on filing events, not on a tradeable universe —
but we keep `date_added` so a point-in-time membership filter can be layered
on later.
"""

from __future__ import annotations

import io

import httpx
import pandas as pd

from .config import PANEL_DIR, require_user_agent

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_CACHE = PANEL_DIR / "universe.csv"


def _cik10(raw) -> str:
    """EDGAR wants a zero-padded 10-digit CIK string (e.g. 320193 -> '0000320193')."""
    return str(int(raw)).zfill(10)


def fetch_sp500(refresh: bool = False) -> pd.DataFrame:
    """Return the S&P 500 constituents as [ticker, name, sector, cik, date_added].

    Cached to data/panel/universe.csv; pass refresh=True to re-pull.
    """
    if _CACHE.exists() and not refresh:
        return pd.read_csv(_CACHE, dtype={"cik": str})

    # Wikipedia blocks requests without a real User-Agent.
    resp = httpx.get(WIKI_SP500, headers={"User-Agent": require_user_agent()}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    raw = tables[0]  # the "constituents" table is first

    df = pd.DataFrame(
        {
            "ticker": raw["Symbol"].astype(str).str.strip(),
            "name": raw["Security"].astype(str).str.strip(),
            "sector": raw["GICS Sector"].astype(str).str.strip(),
            "cik": raw["CIK"].map(_cik10),
            "date_added": pd.to_datetime(raw["Date added"], errors="coerce").dt.date,
        }
    )
    df = df.sort_values("ticker").reset_index(drop=True)

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_CACHE, index=False)
    return df


def sample(tickers: list[str]) -> pd.DataFrame:
    """Convenience: the universe rows for a hand-picked ticker list (smoke tests)."""
    uni = fetch_sp500()
    out = uni[uni["ticker"].isin(tickers)].reset_index(drop=True)
    missing = set(tickers) - set(out["ticker"])
    if missing:
        raise ValueError(f"tickers not in S&P 500 universe: {sorted(missing)}")
    return out
