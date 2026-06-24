"""
sources/synthetic.py
=====================

A HAND-BUILT, throwaway data source for the FDIC feature store.

Why this exists
---------------
Before Snowflake access lands, the poison test needs *something* to run
against. This module fabricates two small tables with values chosen BY HAND so
that the correct output of `get_features_asof(T)` is known in advance. A test is
only meaningful if you already know the right answer; that is the entire job of
this file.

It is temporary. When Snowflake access arrives, `sources/snowflake.py` replaces
it. The thing that must NOT change across that swap is the *shape* of what this
returns: pandas DataFrames, these exact column names, accessed only through the
two as-of read functions below. The poison test is written against that shape,
so if the shape holds, the test runs unchanged against real data.

What it deliberately does NOT do
--------------------------------
- It does not compute features (no lags, no day-of-week). Those are built in
  features.py from what this returns. This source only stores raw EOD state.
- It does not enforce the temporal contract. Enforcement lives at the
  snapshot.py chokepoint. This is just a dumb table that answers "what did
  state look like as of date D".
- It is not realistic. The numbers are small and legible on purpose so you can
  verify the test by eye.

The seeded trap (this is the important part)
--------------------------------------------
On the predicted day T, `mmkt_balance` is set to a value WILDLY different from
its T-1 value. FDIC and same-day MMKT are two halves of the same end-of-day
sweep, so a leak would pull this same-day MMKT number into the features. Because
the T value is so different from T-1, a leaked feature vector would look
obviously wrong — and, more to the point, poisoning T would CHANGE the output.
A correct features.py never reads T, so poisoning T must leave the output
byte-identical. The trap is what gives that assertion teeth: against bland data,
a passing test proves nothing.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from fdicfs import contract


# ---------------------------------------------------------------------------
# Column names — must match the contract's spec exactly.
# Kept here as constants so a typo fails loudly rather than silently making a
# column "disappear" from a feature read.
# ---------------------------------------------------------------------------

# balance_snapshot table
PORTFOLIO_ID = "portfolio_id"
SNAPSHOT_DATE = "snapshot_date"
FDIC_BALANCE = "fdic_balance"
MMKT_BALANCE = "mmkt_balance"
MMKT_REMAINING = "mmkt_remaining"
MMKT_TOTAL = "mmkt_total"
OTHER_INSTRUMENTS_BALANCE = "other_instruments_balance"
PORTFOLIO_PURPOSE = "portfolio_purpose"
STRATEGY_FLAG = "strategy_flag"
BUCKET = "bucket"

# distribution_schedule table
RECORD_VISIBLE_DATE = "record_visible_date"  # when WE could first see this row
DISTRIBUTION_DATE = "distribution_date"      # when the money actually moves
SCHEDULED_DISTRIBUTION_AMOUNT = "scheduled_distribution_amount"


# ---------------------------------------------------------------------------
# A fixed cast of dates. One single portfolio keeps the fixture legible.
# These are real weekdays in late June 2025 (no weekend gaps in this run, so
# T-1 is just the previous row — calendar.py handles real gaps later).
# ---------------------------------------------------------------------------

P = "PORT_001"

D_TM3 = dt.date(2025, 6, 24)  # T-3  (Tue)
D_TM2 = dt.date(2025, 6, 25)  # T-2  (Wed)
D_TM1 = dt.date(2025, 6, 26)  # T-1  (Thu)  <- the last LEGAL feature day
T = dt.date(2025, 6, 27)      # T    (Fri)  <- predicted day; OFF-LIMITS to features

# Make T easy to refer to from the test.
PREDICTED_DAY = T
LAST_LEGAL_FEATURE_DAY = D_TM1


# ---------------------------------------------------------------------------
# Table 1: balance_snapshot
# ---------------------------------------------------------------------------
#
# Read this column-by-column. Every <=T-1 row is a calm, slowly-moving series.
# The T row is the TRAP: mmkt_balance leaps from 300 to 9_999_999. If the
# feature builder ever touches T, that monstrous number (or something derived
# from it) shows up in the output and the poison assertion fails — which is
# exactly what we want a leak to do.

def _balance_rows() -> list[dict]:
    return [
        # ---- T-3 -----------------------------------------------------------
        {
            PORTFOLIO_ID: P,
            SNAPSHOT_DATE: D_TM3,
            FDIC_BALANCE: 1_000,
            MMKT_BALANCE: 100,
            MMKT_REMAINING: 40,
            MMKT_TOTAL: 140,
            OTHER_INSTRUMENTS_BALANCE: 500,
            PORTFOLIO_PURPOSE: "operating",
            STRATEGY_FLAG: "A",
            BUCKET: "core",
        },
        # ---- T-2 -----------------------------------------------------------
        {
            PORTFOLIO_ID: P,
            SNAPSHOT_DATE: D_TM2,
            FDIC_BALANCE: 1_100,
            MMKT_BALANCE: 200,
            MMKT_REMAINING: 50,
            MMKT_TOTAL: 250,
            OTHER_INSTRUMENTS_BALANCE: 520,
            PORTFOLIO_PURPOSE: "operating",
            STRATEGY_FLAG: "A",
            BUCKET: "core",
        },
        # ---- T-1  (the last legal feature day; these are the values a correct
        #            feature vector should be built from) ---------------------
        {
            PORTFOLIO_ID: P,
            SNAPSHOT_DATE: D_TM1,
            FDIC_BALANCE: 1_200,
            MMKT_BALANCE: 300,
            MMKT_REMAINING: 60,
            MMKT_TOTAL: 360,
            OTHER_INSTRUMENTS_BALANCE: 540,
            PORTFOLIO_PURPOSE: "operating",
            STRATEGY_FLAG: "A",
            BUCKET: "core",
        },
        # ---- T  (THE TRAP) -------------------------------------------------
        # fdic_balance here (1_500) is the LABEL the dataset will read.
        # mmkt_balance here (9_999_999) is the poison bait: absurdly far from
        # the T-1 value of 300, so any leak is glaringly visible.
        {
            PORTFOLIO_ID: P,
            SNAPSHOT_DATE: T,
            FDIC_BALANCE: 1_500,            # <- legitimate label for day T
            MMKT_BALANCE: 9_999_999,        # <- TRAP: must never reach a feature
            MMKT_REMAINING: 9_999_999,
            MMKT_TOTAL: 19_999_998,
            OTHER_INSTRUMENTS_BALANCE: 9_999_999,
            PORTFOLIO_PURPOSE: "REASSIGNED",  # <- also a trap: "static" attr changed on T
            STRATEGY_FLAG: "Z",
            BUCKET: "POISON",
        },
    ]


# ---------------------------------------------------------------------------
# Table 2: distribution_schedule
# ---------------------------------------------------------------------------
#
# A scheduled distribution is only a LEGAL feature if its record was visible
# BEFORE its distribution date. Two rows make the distinction concrete:
#
#   - HONEST row: a distribution dated T-1, whose record was visible at T-3.
#     Knowable in advance -> legal to use as a feature for predicting T.
#   - TRAP row: a distribution dated T whose record only becomes visible ON T.
#     Not knowable before T -> a leak if used. The poison test corrupts this
#     row to prove the feature builder never depends on it.

def _schedule_rows() -> list[dict]:
    return [
        # ---- HONEST: visible at T-3, pays out at T-1 -----------------------
        {
            PORTFOLIO_ID: P,
            RECORD_VISIBLE_DATE: D_TM3,
            DISTRIBUTION_DATE: D_TM1,
            SCHEDULED_DISTRIBUTION_AMOUNT: 75,
        },
        # ---- TRAP: only visible on T, pays out on T ------------------------
        # A correct feature builder predicting T must NOT see this, because at
        # prediction time (before T closes) this record does not exist yet.
        {
            PORTFOLIO_ID: P,
            RECORD_VISIBLE_DATE: T,
            DISTRIBUTION_DATE: T,
            SCHEDULED_DISTRIBUTION_AMOUNT: 9_999_999,  # <- poison bait
        },
    ]


# ---------------------------------------------------------------------------
# THE INTERFACE  —  the only two functions the rest of the system calls.
# snapshot.py will wrap these; features.py never calls them directly.
# When Snowflake lands, snowflake.py implements these same two signatures.
# ---------------------------------------------------------------------------

def get_balance_asof(as_of: dt.date, portfolio_id: str = P) -> pd.DataFrame:
    """Return all balance_snapshot rows with snapshot_date <= as_of.

    "As of date D" means: everything we could legitimately have seen by the
    end of D, and nothing later. This is the single primitive the whole
    point-in-time story rests on. Note the boundary is INCLUSIVE of `as_of`
    (<=), because as_of is itself a day we are allowed to have observed.

    IMPORTANT: this function does NOT know what T is. The CALLER decides which
    date to pass. To stay on the legal side of the contract, a feature read for
    predicting T must call this with as_of = T-1 (resolved by calendar.py),
    NOT T. Passing T here is how a leak would happen — and that is precisely
    what the snapshot.py chokepoint and the poison test exist to prevent.
    """
    df = pd.DataFrame(_balance_rows())
    mask = (df[SNAPSHOT_DATE] <= as_of) & (df[PORTFOLIO_ID] == portfolio_id)
    return df.loc[mask].sort_values(SNAPSHOT_DATE).reset_index(drop=True)


def get_schedule_asof(as_of: dt.date, portfolio_id: str = P) -> pd.DataFrame:
    """Return distribution_schedule rows VISIBLE as of `as_of`.

    Visibility, not the distribution date, is what gates legality. A row counts
    only if record_visible_date <= as_of. A distribution that pays out far in
    the future is still a legal feature today IF its record is already visible;
    a distribution paying out today whose record only appears today is NOT.

    So the filter is on record_visible_date, deliberately — filtering on
    distribution_date instead would be a subtle leak, letting same-day-announced
    events sneak into the feature set.
    """
    df = pd.DataFrame(_schedule_rows())
    mask = (df[RECORD_VISIBLE_DATE] <= as_of) & (df[PORTFOLIO_ID] == portfolio_id)
    return df.loc[mask].sort_values(DISTRIBUTION_DATE).reset_index(drop=True)


def get_label(as_of: dt.date, portfolio_id: str = P) -> float:
    """Return the day-`as_of` FDIC balance — the LABEL, the one legal read of T.

    This is intentionally a SEPARATE function from get_balance_asof. The label
    read and the feature read must be physically distinct calls; the classic
    leak is one query that serves both and silently grabs T for the features
    too. Keeping them apart in the interface makes that mistake hard to make by
    accident.
    """
    df = pd.DataFrame(_balance_rows())
    row = df[(df[SNAPSHOT_DATE] == as_of) & (df[PORTFOLIO_ID] == portfolio_id)]
    if row.empty:
        raise KeyError(f"No balance snapshot for {portfolio_id} on {as_of}.")
    return float(row.iloc[0][FDIC_BALANCE])


# ---------------------------------------------------------------------------
# Poison hooks  —  used ONLY by the poison test.
# These return copies of the fixtures with the day-T rows corrupted, so the
# test can ask "if T is garbage, does the feature output change?" The answer
# must be no. Kept here (next to the data) rather than in the test so the test
# stays about ASSERTIONS, not about knowing the fixture's internals.
# ---------------------------------------------------------------------------

def poisoned_balance_asof(as_of: dt.date, portfolio_id: str = P) -> pd.DataFrame:
    """Like get_balance_asof, but the day-T row is replaced with garbage.

    Only the T row changes; every <=T-1 row is untouched. So for any legal
    feature read (as_of <= T-1) this returns EXACTLY what get_balance_asof
    returns — the poison is invisible unless someone illegally reaches T.
    """
    rows = _balance_rows()
    for r in rows:
        if r[SNAPSHOT_DATE] == T:
            r[FDIC_BALANCE] = -1
            r[MMKT_BALANCE] = -1
            r[MMKT_REMAINING] = -1
            r[MMKT_TOTAL] = -1
            r[OTHER_INSTRUMENTS_BALANCE] = -1
            r[PORTFOLIO_PURPOSE] = "CORRUPT"
            r[STRATEGY_FLAG] = "CORRUPT"
            r[BUCKET] = "CORRUPT"
    df = pd.DataFrame(rows)
    mask = (df[SNAPSHOT_DATE] <= as_of) & (df[PORTFOLIO_ID] == portfolio_id)
    return df.loc[mask].sort_values(SNAPSHOT_DATE).reset_index(drop=True)


def poisoned_schedule_asof(as_of: dt.date, portfolio_id: str = P) -> pd.DataFrame:
    """Like get_schedule_asof, but the day-T schedule row is corrupted.

    Corrupts the trap row (the one visible only on T). Any legal read
    (as_of <= T-1) never includes that row, so the output is unchanged unless
    the feature builder illegally consults a same-day-visible schedule record.
    """
    rows = _schedule_rows()
    for r in rows:
        if r[RECORD_VISIBLE_DATE] == T:
            r[SCHEDULED_DISTRIBUTION_AMOUNT] = -1
            r[DISTRIBUTION_DATE] = T
    df = pd.DataFrame(rows)
    mask = (df[RECORD_VISIBLE_DATE] <= as_of) & (df[PORTFOLIO_ID] == portfolio_id)
    return df.loc[mask].sort_values(DISTRIBUTION_DATE).reset_index(drop=True)


if __name__ == "__main__":
    # Eyeball the fixture and confirm the trap behaves as intended.
    print("=== balance as-of T-1 (the legal feature window) ===")
    print(get_balance_asof(LAST_LEGAL_FEATURE_DAY).to_string(index=False))
    print(f"\nLabel for T ({T}): {get_label(T)}")
    print("\n=== schedule visible as-of T-1 (honest row only) ===")
    print(get_schedule_asof(LAST_LEGAL_FEATURE_DAY).to_string(index=False))

    # The key invariant the poison test will lean on: poisoning T must not
    # change a T-1 read, because a T-1 read never includes the T row.
    clean = get_balance_asof(LAST_LEGAL_FEATURE_DAY)
    dirty = poisoned_balance_asof(LAST_LEGAL_FEATURE_DAY)
    assert clean.equals(dirty), "T-1 read changed under poison — fixture is wrong!"
    print("\nOK: poisoning T leaves the T-1 balance read byte-identical.")



