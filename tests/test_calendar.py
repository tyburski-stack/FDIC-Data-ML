"""
test_calendar.py — proves prior_business_day handles the gaps the synthetic
fixture deliberately avoids (weekends, holidays, long weekends).

All dates hand-verifiable against a 2025 calendar.
"""

import datetime as dt
import pytest

from fdicfs.calendar import (
    is_business_day,
    prior_business_day,
    business_day_of_month,
    days_to_quarter_end,
)


# ---------------------------------------------------------------------------
# is_business_day — the predicate, tested independently of the walk
# ---------------------------------------------------------------------------

def test_normal_weekday_is_business_day():
    # Wednesday, June 25 2025 — an ordinary weekday
    assert is_business_day(dt.date(2025, 6, 25))


def test_saturday_is_not_business_day():
    # Saturday, June 28 2025
    assert not is_business_day(dt.date(2025, 6, 28))


def test_sunday_is_not_business_day():
    # Sunday, June 29 2025
    assert not is_business_day(dt.date(2025, 6, 29))


def test_july_4_is_not_business_day():
    # Friday, July 4 2025 — Independence Day
    assert not is_business_day(dt.date(2025, 7, 4))


def test_christmas_is_not_business_day():
    # Thursday, December 25 2025
    assert not is_business_day(dt.date(2025, 12, 25))


# ---------------------------------------------------------------------------
# prior_business_day — the walk
# ---------------------------------------------------------------------------

def test_plain_weekday_steps_back_one():
    # T = Wednesday June 25 → T-1 = Tuesday June 24
    assert prior_business_day(dt.date(2025, 6, 25)) == dt.date(2025, 6, 24)


def test_monday_skips_back_to_friday():
    # T = Monday June 30 → skips Sun/Sat → Friday June 27
    assert prior_business_day(dt.date(2025, 6, 30)) == dt.date(2025, 6, 27)


def test_tuesday_after_memorial_day():
    # Memorial Day = Monday May 26 2025.
    # T = Tuesday May 27 → skips Mon holiday + weekend → Friday May 23
    assert prior_business_day(dt.date(2025, 5, 27)) == dt.date(2025, 5, 23)


def test_monday_after_july_4():
    # July 4 2025 is a Friday. T = Monday July 7.
    # Must skip Sun, Sat, AND the Friday holiday → Thursday July 3
    assert prior_business_day(dt.date(2025, 7, 7)) == dt.date(2025, 7, 3)


# ---------------------------------------------------------------------------
# The strictly-before guarantee — the temporal contract in calendar form
# ---------------------------------------------------------------------------

def test_result_is_strictly_before_T():
    T = dt.date(2025, 6, 25)
    assert prior_business_day(T) < T


def test_business_day_T_does_not_return_itself():
    # T is itself a business day (Wednesday) — result must still be < T
    T = dt.date(2025, 6, 25)
    assert prior_business_day(T) != T


# ---------------------------------------------------------------------------
# TODO for you — verify these dates by hand, then fill in the expected answer
# ---------------------------------------------------------------------------

def test_tuesday_after_labor_day():
    # Labor Day 2025 = Monday Sept 1. T = Tuesday Sept 2.
    # What should prior_business_day return? Work it out and assert it.
    expected = dt.date(2025, 8, 29)
    assert prior_business_day(dt.date(2025, 9, 2)) == expected


def test_day_after_thanksgiving():
    # Thanksgiving 2025 = Thursday Nov 27. The Friday after (Nov 28) is NOT
    # a federal holiday — it's a normal business day. So what is T-1 for
    # T = Friday Nov 28? And separately, what's T-1 for the FOLLOWING Monday
    # Dec 1? (Thanksgiving Thursday is the gap to skip in that second one.)
    expected = dt.date(2025, 11, 28)
    assert prior_business_day(dt.date(2025, 12, 1)) == expected

# ---------------------------------------------------------------------------
# business_day_of_month — hand-verified against June 2025.
# June 1 2025 is a Sunday; Juneteenth (Thu Jun 19) is a federal holiday and is
# correctly skipped, so Jun 20 is the 14th business day, not the 15th.
# ---------------------------------------------------------------------------

def test_bdom_first_business_day():
    # Sun Jun 1 is not a business day; Mon Jun 2 is the 1st business day.
    assert business_day_of_month(dt.date(2025, 6, 2)) == 1


def test_bdom_skips_juneteenth():
    # Fri Jun 20 — Juneteenth (Thu Jun 19) was skipped, so this is the 14th.
    assert business_day_of_month(dt.date(2025, 6, 20)) == 14


def test_bdom_fixture_T():
    # Fri Jun 27 (the fixture's predicted day T) is the 19th business day.
    assert business_day_of_month(dt.date(2025, 6, 27)) == 19


def test_bdom_last_business_day_of_june():
    # Mon Jun 30 is the 20th and final business day of June 2025.
    assert business_day_of_month(dt.date(2025, 6, 30)) == 20


def test_bdom_weekend_returns_prior_ordinal():
    # Sat Jun 28 is not a business day; counted on-or-before, it matches Fri
    # Jun 27's ordinal (19) rather than advancing.
    assert business_day_of_month(dt.date(2025, 6, 28)) == 19


# ---------------------------------------------------------------------------
# days_to_quarter_end — business days to the last business day of the quarter.
# Q2 2025 ends Mon Jun 30 (itself a business day).
# ---------------------------------------------------------------------------

def test_days_to_qe_on_quarter_end_is_zero():
    # Mon Jun 30 IS the last business day of Q2 → 0.
    assert days_to_quarter_end(dt.date(2025, 6, 30)) == 0


def test_days_to_qe_fixture_T():
    # Fri Jun 27 → one business day (Mon Jun 30) to quarter end → 1.
    assert days_to_quarter_end(dt.date(2025, 6, 27)) == 1


def test_days_to_qe_two_out():
    # Thu Jun 26 → Fri Jun 27, Mon Jun 30 → 2 business days.
    assert days_to_quarter_end(dt.date(2025, 6, 26)) == 2


def test_days_to_qe_start_of_quarter():
    # Tue Apr 1 2025, start of Q2 → 62 business days to Jun 30.
    assert days_to_quarter_end(dt.date(2025, 4, 1)) == 62
