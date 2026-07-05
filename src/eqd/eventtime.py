"""eventtime.py — THE SPINE.

Maps an EDGAR filing's acceptance datetime to `t0`, the first tradeable
session, under strict no-lookahead semantics. Everything downstream aligns
to `t0`. If this module is wrong, every result in the project is fiction.

Core rule
---------
`t0` = the first trading session whose **open (9:30 ET)** occurs *strictly
after* the acceptance datetime.

    - Filed after the 16:00 ET close (typical 10-K/10-Q) -> next session.
    - Filed on a weekend / holiday                       -> next session.
    - Filed intraday after the open                      -> next session
      (conservative: avoids same-bar lookahead).
    - Filed pre-market, before 9:30 on a trading day     -> that same session
      (the info is public before the bell).
    - Filed exactly at 9:30:00                            -> next session
      ("strictly after" excludes the equal case).
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

from .config import EXCHANGE_CALENDAR, FILING_TZ


def to_utc(when: str | dt.datetime | pd.Timestamp, *, assume_tz: str = FILING_TZ) -> pd.Timestamp:
    """Normalize any acceptance-datetime input to a tz-aware UTC Timestamp.

    Naive inputs are assumed to be in `assume_tz` (Eastern, since that is how
    EDGAR reports acceptance times). Tz-aware inputs are converted as-is.
    """
    ts = pd.Timestamp(when)
    if ts.tz is None:
        ts = ts.tz_localize(assume_tz)
    return ts.tz_convert("UTC")


class TradingCalendar:
    """Wraps an exchange schedule for a fixed date range.

    Build one for the span your data covers; it caches the schedule so
    repeated `t0()` / `offset()` calls are cheap.
    """

    def __init__(self, start: str, end: str, calendar: str = EXCHANGE_CALENDAR):
        self._cal = mcal.get_calendar(calendar)
        sched = self._cal.schedule(start_date=start, end_date=end)
        # `market_open` is tz-aware UTC; the index is tz-naive session dates.
        self._opens: pd.Series = sched["market_open"]
        self._closes: pd.Series = sched["market_close"]
        self.sessions: pd.DatetimeIndex = sched.index
        self._start, self._end = start, end

    def t0(self, acceptance: str | dt.datetime | pd.Timestamp) -> pd.Timestamp:
        """First tradeable session for a filing accepted at `acceptance`.

        Returns the session date (tz-naive, normalized to midnight).
        """
        ts = to_utc(acceptance)  # tz-aware UTC
        # Both sides are tz-aware UTC; let pandas do the elementwise compare.
        mask = (self._opens > ts).to_numpy()
        later = self._opens.index[mask]
        if len(later) == 0:
            raise ValueError(
                f"acceptance {ts} is beyond the calendar range "
                f"[{self._start}, {self._end}]; extend `end`."
            )
        return pd.Timestamp(later[0]).normalize()

    def offset(self, session: str | pd.Timestamp, n: int) -> pd.Timestamp:
        """The session `n` trading days from `session` (n may be negative)."""
        s = pd.Timestamp(session).normalize()
        pos = self.sessions.get_indexer([s])[0]
        if pos == -1:
            raise ValueError(f"{s.date()} is not a trading session")
        j = pos + n
        if j < 0 or j >= len(self.sessions):
            raise ValueError(
                f"offset {n} from {s.date()} falls outside the calendar range; "
                f"extend the range."
            )
        return self.sessions[j]

    def window(self, t0: str | pd.Timestamp, lo: int, hi: int) -> pd.DatetimeIndex:
        """Sessions from `t0+lo` .. `t0+hi` inclusive (event-time window).

        Examples: `window(t0, 0, 1)` -> [t0, t0+1]; `window(t0, -5, -1)` ->
        the pre-event placebo window.
        """
        s = pd.Timestamp(t0).normalize()
        pos = self.sessions.get_indexer([s])[0]
        if pos == -1:
            raise ValueError(f"{s.date()} is not a trading session")
        a, b = pos + lo, pos + hi
        if a < 0 or b >= len(self.sessions):
            raise ValueError("window falls outside calendar range; extend it.")
        return self.sessions[a : b + 1]


def assert_no_lookahead(
    input_timestamps,
    anchor: str | dt.datetime | pd.Timestamp,
    *,
    label: str = "feature",
) -> None:
    """Hard gate: fail if any feature input post-dates the event anchor.

    `input_timestamps` is an iterable of the timestamps that fed a feature
    (e.g. the acceptance datetimes of every filing used). `anchor` is the
    acceptance datetime of the event the feature is attached to. Any input
    strictly after the anchor is lookahead and raises.
    """
    a = to_utc(anchor)
    offenders = [t for t in input_timestamps if to_utc(t) > a]
    if offenders:
        raise AssertionError(
            f"LOOKAHEAD in {label}: {len(offenders)} input(s) post-date the "
            f"anchor {a}. First offender: {to_utc(offenders[0])}."
        )


@lru_cache(maxsize=4)
def default_calendar(start: str, end: str) -> TradingCalendar:
    """Cached calendar for convenience in scripts/tests."""
    return TradingCalendar(start, end)
