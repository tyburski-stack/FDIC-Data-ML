
import datetime as dt
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


_BUS_DAY = CustomBusinessDay(calendar = USFederalHolidayCalendar())

def is_business_day(d: dt.date) -> bool:
    """ 
    Determine if a day is a business day based on US Federal Holidays and Weekends
    """
    ts = pd.Timestamp(d)
    return _BUS_DAY.rollforward(ts) == ts
    

def prior_business_day(date: dt.date) -> dt.date:
    """
    Determine the most recent business day to a certain date
    """
    candidate = date - dt.timedelta(days = 1) # start at T-1
    # cap loop at 10 days (you don't see a break nearly that long ever)
    for _ in range(10):
        if is_business_day(candidate):
            return candidate
        candidate -= dt.timedelta(days = 1)
    
    raise RuntimeError(f"No business day found within 10 days before {date}")


# ---------------------------------------------------------------------------
# Cyclical / relative time features — deferred out of features.py into the
# single home for "what does this date MEAN". All computed from the DATE
# alone, never from any snapshot's contents, so they are legal features for
# predicting T (knowing "T is the last business day of the quarter" needs no
# data from T). They reuse is_business_day so the holiday set has one owner.
# ---------------------------------------------------------------------------

def business_day_of_month(d: dt.date) -> int:
    """Ordinal of `d` among the business days of its month (1-based).

    The first business day of the month is 1, the next is 2, and so on.
    Counts business days on-or-before `d` within the same month, so a holiday
    or weekend returns the same ordinal as the preceding business day (for the
    predicted day T this never bites, since T is always a real business day).

    Holiday-aware via is_business_day: e.g. June 2025 skips Juneteenth, so
    Jun 27 is the 19th business day, not the 20th.

    This is where institutional cash seasonality lives (month-end
    distribution/operating cycles).
    """
    n = 0
    cur = d.replace(day=1)
    while cur <= d:
        if is_business_day(cur):
            n += 1
        cur += dt.timedelta(days=1)
    return n


def _last_business_day_of_quarter(d: dt.date) -> dt.date:
    """The last business day of the calendar quarter containing `d`.

    Walks back from the quarter's final calendar day until it hits a business
    day, so a quarter-end falling on a weekend/holiday resolves to the prior
    business day.
    """
    q_end_month = ((d.month - 1) // 3) * 3 + 3  # 3, 6, 9, or 12
    if q_end_month == 12:
        last = dt.date(d.year, 12, 31)
    else:
        last = dt.date(d.year, q_end_month + 1, 1) - dt.timedelta(days=1)
    while not is_business_day(last):
        last -= dt.timedelta(days=1)
    return last


def days_to_quarter_end(d: dt.date) -> int:
    """Business days from `d` to the last business day of its quarter.

    Counted in BUSINESS days (matching the rest of this module), so 0 means
    "`d` IS the last business day of the quarter" — the regime the model most
    wants flagged. Distinct from month-end: quarter-ends are rarer and larger.
    """
    target = _last_business_day_of_quarter(d)
    n = 0
    cur = d
    while cur < target:
        cur += dt.timedelta(days=1)
        if is_business_day(cur):
            n += 1
    return n



