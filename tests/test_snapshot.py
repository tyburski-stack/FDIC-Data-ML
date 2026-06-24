"""
test_snapshot.py — proves the chokepoint holds its two guarantees:
  1. the feature path returns state as-of T-1 and NEVER the T row
  2. the label path is the one legitimate read of T
plus: poison on T doesn't leak through the door, and the empty-data guard fires.

Runs against the synthetic source (passed in), exactly as Snowflake will later.
"""

import datetime as dt
import pytest

from fdicfs import snapshot
from fdicfs.sources import synthetic as src


# ---------------------------------------------------------------------------
# The feature path stops at T-1 — it physically cannot return the T row
# ---------------------------------------------------------------------------

def test_features_resolve_to_T_minus_1():
    # src.T is Fri Jun 27; T-1 is Thu Jun 26 (= src.D_TM1), no gap to skip
    state = snapshot.features_asof(src.T, src.P, source=src)
    assert state.as_of == src.D_TM1


def test_feature_window_never_includes_T():
    # The max snapshot_date in the returned balances must be T-1, never T.
    # This is the core guarantee made testable: the T row (the trap) cannot
    # appear in a feature read.
    state = snapshot.features_asof(src.T, src.P, source=src)
    assert state.balances[src.SNAPSHOT_DATE].max() == src.D_TM1


def test_feature_window_excludes_the_poison_value():
    # The trap mmkt_balance (9_999_999) lives only on T,
    # so it must not appear anywhere in the feature window.
    state = snapshot.features_asof(src.T, src.P, source=src)
    assert 9_999_999 not in state.balances[src.MMKT_BALANCE].values


# ---------------------------------------------------------------------------
# The label path reads T — and that is correct, here and only here
# ---------------------------------------------------------------------------

def test_label_reads_day_T():
    # Known label from the fixture: fdic_balance on T is 1_500.
    assert snapshot.label_asof(src.T, src.P, source=src) == 1_500.0


# ---------------------------------------------------------------------------
# Poison on T does not leak through the chokepoint
# (the poison test's logic, one layer down — proving the DOOR holds, not the stub)
# ---------------------------------------------------------------------------

def test_poison_on_T_does_not_change_feature_window():
    clean = snapshot.features_asof(src.T, src.P, source=src)
    poisoned = snapshot.features_asof(
        src.T, src.P, source=src,
        balance_source=src.poisoned_balance_asof,
        schedule_source=src.poisoned_schedule_asof,
    )
    # byte-identical frames — poisoning T changed nothing a feature read sees
    assert clean.balances.equals(poisoned.balances)
    assert clean.schedule.equals(poisoned.schedule)


# ---------------------------------------------------------------------------
# The empty-data guard fires when nothing precedes T-1
# ---------------------------------------------------------------------------

def test_empty_balance_window_raises():
    # T = Tue Jun 24 2025 -> T-1 = Mon Jun 23, which is BEFORE the fixture's
    # earliest row (D_TM3 = Jun 24). So the <=T-1 window is empty and the
    # guard must raise. (Verify by hand: prior_business_day(Jun 24) == Jun 23.)
    early_T = dt.date(2025, 6, 24)
    with pytest.raises(ValueError):
        snapshot.features_asof(early_T, src.P, source=src)


def test_label_is_separate_from_features():
    # The whole anti-leak design is that the label read and feature read are
    # different calls hitting different dates. Prove they disagree: the label
    # (day-T fdic_balance) should NOT equal the most recent feature-window
    # fdic_balance (day-T-1). Fetch both and assert they differ.
    # Fixture values: T-1 fdic_balance = 1_200, T fdic_balance = 1_500.
    # TODO: fetch label_asof and the last fdic_balance from features_asof, assert !=
    label = snapshot.label_asof(src.T, src.P, source = src)
    state = snapshot.features_asof(src.T, src.P, source = src)
    last_feature_fdic = state.balances.iloc[-1][src.FDIC_BALANCE]
    assert label != last_feature_fdic



def test_schedule_window_excludes_same_day_record():
    # The honest schedule row is visible at T-3 (legal); the trap row is only
    # visible on T (a leak if used). A feature read as-of T-1 must include the
    # honest row and EXCLUDE the trap. Assert the trap amount (9_999_999) is
    # not in the returned schedule, and the honest amount (75) is.
    state = snapshot.features_asof(src.T, src.P, source = src)
    amounts = state.schedule[src.SCHEDULED_DISTRIBUTION_AMOUNT].values
    assert 75 in amounts            # row (visible T-3) included
    assert 9_999_999 not in amounts # trap row (visible only at T) excluded


