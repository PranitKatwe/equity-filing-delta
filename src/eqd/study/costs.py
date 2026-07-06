"""study/costs.py — transaction costs. Gross alpha is fiction; net is the truth.

A tradeable claim must survive round-trip costs. We model a round trip as a
flat cost in return units, defaulting to a conservative estimate for liquid
large-caps. The design's refinement (a Corwin-Schultz high-low spread estimator
from the OHLC we already cache, plus an impact term) can replace the flat proxy
later; the harness reports gross AND net either way so the reader sees both.

For a long-short portfolio you pay to trade BOTH legs, so the portfolio round
trip is 2x a single-name round trip.
"""

from __future__ import annotations

# One-way trade cost (bps) for a liquid large-cap: ~half-spread + slippage.
ONE_WAY_BPS = 5.0
ROUNDTRIP_BPS = 2 * ONE_WAY_BPS  # enter + exit


def roundtrip_cost(bps: float = ROUNDTRIP_BPS) -> float:
    """Round-trip cost as a return decimal (e.g. 10 bps -> 0.0010)."""
    return bps / 1e4


def net_of_cost(gross_return: float, *, legs: int = 1, bps: float = ROUNDTRIP_BPS) -> float:
    """Subtract round-trip cost from a gross return.

    legs=1: single position (long OR short). legs=2: a long-short pair.
    """
    return gross_return - legs * roundtrip_cost(bps)
