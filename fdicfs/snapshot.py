
"""
snapshot.py — piece 2 — THE CHOKEPOINT.

The single door every feature read passes through. Nothing upstream of this
file talks to a concrete source table; everything asks here. This is where
two guarantees are made physical:

  1. T-1 resolution is automatic (via calendar.prior_business_day), not the
     caller's job — so no caller can fake it with `T - timedelta(days=1)`.
  2. The feature path NEVER touches T. T is consumed into T-1 on the first
     line and never seen again. The label path is the ONE place T reaches a
     source, and it is a physically separate function.

The source is passed IN (never imported here), so snapshot.py knows nothing
about synthetic vs Snowflake. Swapping backends is a call-site change:
`source=synthetic` -> `source=snowflake`. The poison test plugs in here too,
via the per-read overrides.
"""

from __future__ import annotations

import datetime as dt
from typing import NamedTuple

import pandas as pd

from fdicfs.calendar import prior_business_day
from fdicfs import contract


# ---------------------------------------------------------------------------
# What the feature path hands back: state as-of T-1 — a WINDOW, not one row.
# Named (not a bare tuple) so downstream unpacks by name and a reorder can't
# silently swap them.
# ---------------------------------------------------------------------------

class AsOfState(NamedTuple):
    as_of: dt.date          # the resolved T-1, useful for asserts/debugging
    balances: pd.DataFrame  # all balance rows <= T-1 
    schedule: pd.DataFrame  # all schedule rows visible <= T-1


# Feature read - <= T-1 only, T consumed on line 1 and not ever reused

def features_asof(
    T: dt.date,
    portfolio_id: str,
    source,
    *,
    balance_source = None,
    schedule_source = None
) -> AsOfState:
    """
    Return state as-of T-1 for predicting day T. Never reads T.
    `source` is any object exposing get_balance_asof / get_schedule_asof
    (synthetic now, snowflake later). The overrides let the poison test
    swap ONE read at a time while leaving the others clean.
    """
    balance_source = balance_source or source.get_balance_asof
    schedule_source = schedule_source or source.get_schedule_asof

    # T consumed here, shouldn't appear below this line again
    as_of = prior_business_day(T)

    # a row's READ DATE would stay the same across columns, so for efficiency's sake
    # assert over one representative lagged column (use mmkt_balance)
    contract.assert_feature_date_is_legal("mmkt_balance", as_of, T)

    balances = balance_source(as_of, portfolio_id)
    schedule = schedule_source(as_of, portfolio_id)


    # handle a case with no balance data
    if balances.empty:
        raise ValueError(
            f"No balance data on/before {as_of} for {portfolio_id}. "
            f"Cannot build features for T = {T}."
        )

    return AsOfState(as_of = as_of, balances = balances, schedule = schedule)


# Label read - legitimate access of T for comparison. Intentionally separate function

def label_asof(
    T: dt.date,
    portfolio_id: str,
    source,
    *,
    label_source = None
) -> float:
    """
    Return the day-T FDIC balance — the label. The only read of T.

    Physically separate from features_asof so a single query can never serve
    both the feature and the label (the classic leak).
    """
    label_source = label_source or source.get_label

    return label_source(T, portfolio_id)
