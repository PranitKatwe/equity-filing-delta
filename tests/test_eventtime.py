"""THE ALIGNMENT GATE.

Every case pins one no-lookahead timing rule to a concrete, checkable NYSE
date. If any of these break, the point-in-time claim of the whole project is
void. Do not weaken a case to make it pass — fix the code.

Reference dates (all 2023, US/Eastern):
    Thu 02-02, Fri 02-03  : ordinary consecutive trading days
    Sat 02-04, Mon 02-06  : weekend then next session
    Fri 01-13, Mon 01-16  : 01-16 is the MLK holiday (market closed)
    Tue 01-17             : first session after the MLK long weekend
"""

import datetime as dt

import pandas as pd
import pytest

from eqd.eventtime import TradingCalendar, assert_no_lookahead, to_utc

CAL = TradingCalendar("2023-01-01", "2023-03-01")


def d(s: str) -> dt.date:
    return pd.Timestamp(s).date()


# ---------------------------------------------------------------- t0 mapping

def test_after_close_goes_to_next_session():
    # 10-K filed at 16:30 ET (after the 16:00 close) -> next trading day.
    assert CAL.t0("2023-02-02 16:30:00").date() == d("2023-02-03")


def test_pre_open_files_same_session():
    # News public at 08:00 ET, before the 09:30 bell -> tradeable that day.
    assert CAL.t0("2023-02-02 08:00:00").date() == d("2023-02-02")


def test_intraday_after_open_is_conservative_next_session():
    # Filed 11:00 ET, after the open -> next session (avoids same-bar lookahead).
    assert CAL.t0("2023-02-02 11:00:00").date() == d("2023-02-03")


def test_exactly_at_open_excluded_by_strict_inequality():
    # Accepted at 09:30:00 sharp -> the open is NOT strictly after -> next session.
    assert CAL.t0("2023-02-02 09:30:00").date() == d("2023-02-03")


def test_one_second_before_open_files_same_session():
    assert CAL.t0("2023-02-02 09:29:59").date() == d("2023-02-02")


def test_weekend_rolls_to_monday():
    assert CAL.t0("2023-02-04 10:00:00").date() == d("2023-02-06")  # Sat -> Mon


def test_after_close_before_holiday_skips_the_holiday():
    # Filed Fri 01-13 17:00 ET; next session skips the MLK Monday -> Tue 01-17.
    assert CAL.t0("2023-01-13 17:00:00").date() == d("2023-01-17")


def test_weekend_before_holiday_rolls_past_it():
    assert CAL.t0("2023-01-14 12:00:00").date() == d("2023-01-17")  # Sat -> Tue


def test_acceptance_beyond_calendar_range_raises():
    with pytest.raises(ValueError, match="beyond the calendar range"):
        CAL.t0("2023-06-01 10:00:00")


# ------------------------------------------------------- timezone handling

def test_naive_input_is_treated_as_eastern():
    # 09:30 ET == 14:30 UTC in winter (EST, UTC-5).
    assert to_utc("2023-02-02 09:30:00") == pd.Timestamp("2023-02-02 14:30:00", tz="UTC")


def test_tz_aware_input_is_respected():
    # Same instant, expressed in UTC, must map to the same t0.
    aware = pd.Timestamp("2023-02-02 21:30:00", tz="UTC")  # 16:30 ET
    assert CAL.t0(aware).date() == d("2023-02-03")


# ------------------------------------------------ offset / window helpers

def test_offset_forward_and_back():
    assert CAL.offset("2023-02-02", 1).date() == d("2023-02-03")
    assert CAL.offset("2023-02-03", -1).date() == d("2023-02-02")


def test_offset_skips_weekend():
    assert CAL.offset("2023-02-03", 1).date() == d("2023-02-06")  # Fri +1 -> Mon


def test_offset_on_non_session_raises():
    with pytest.raises(ValueError, match="not a trading session"):
        CAL.offset("2023-02-04", 1)  # Saturday


def test_event_window_inclusive():
    w = CAL.window("2023-02-02", 0, 1)
    assert [x.date() for x in w] == [d("2023-02-02"), d("2023-02-03")]


def test_placebo_window_is_five_prior_sessions():
    w = CAL.window("2023-02-06", -5, -1)
    assert len(w) == 5
    assert w[-1].date() == d("2023-02-03")  # last pre-event session
    assert all(x.date() < d("2023-02-06") for x in w)


# ------------------------------------------------------- lookahead audit

def test_lookahead_audit_passes_when_inputs_precede_anchor():
    anchor = "2023-02-02 16:30:00"
    inputs = ["2023-02-02 16:30:00", "2022-02-03 16:30:00"]  # this filing + prior year
    assert_no_lookahead(inputs, anchor)  # must not raise


def test_lookahead_audit_fails_on_future_input():
    anchor = "2023-02-02 16:30:00"
    inputs = ["2023-02-02 16:30:00", "2023-02-03 09:00:00"]  # one input AFTER anchor
    with pytest.raises(AssertionError, match="LOOKAHEAD"):
        assert_no_lookahead(inputs, anchor)
