"""Central config: paths and constants. No secrets here (those live in .env)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FILINGS_DIR = DATA / "filings"
PRICES_DIR = DATA / "prices"
PANEL_DIR = DATA / "panel"

# --- Event-time conventions ---
EXCHANGE_CALENDAR = "XNYS"          # NYSE
FILING_TZ = "America/New_York"      # EDGAR acceptance datetimes are ET

# --- SEC access ---
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
SEC_RATE_LIMIT_PER_SEC = 8          # stay under the 10 req/s ceiling


def require_user_agent() -> str:
    """SEC blocks requests without a declared User-Agent + contact."""
    if not SEC_USER_AGENT or "@" not in SEC_USER_AGENT:
        raise RuntimeError(
            "Set SEC_USER_AGENT in .env to '<app> <your-email>' — "
            "SEC requires a contact-bearing User-Agent."
        )
    return SEC_USER_AGENT
