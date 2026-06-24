"""
test_contract.py — proves the contract's ENFORCEMENT actually fires.

Until now `contract.py` was exercised only by its `__main__` self-check and
indirectly through the snapshot/poison suites. Those prove the happy path. This
file proves the unhappy path: that `assert_is_legal_feature` and
`assert_feature_date_is_legal` RAISE on bad input, and that
`assert_contract_is_wellformed` catches a contract edited into an inconsistent
state. A guard that never fires is not a guard — these tests give it teeth.

Data-independent: nothing here touches a source table. It reads only the
contract spec and pokes the enforcement functions with hand-made inputs.
"""

import datetime as dt

import pytest

from fdicfs import contract
from fdicfs.contract import (
    ColumnSpec,
    LagRule,
    LeakageError,
    Role,
)


# ---------------------------------------------------------------------------
# LeakageError is an AssertionError subclass — a leak is a broken assertion.
# Code that catches AssertionError (or runs under `python -O` awareness) should
# know this relationship holds, so pin it.
# ---------------------------------------------------------------------------

def test_leakage_error_is_assertion_error():
    assert issubclass(LeakageError, AssertionError)


# ---------------------------------------------------------------------------
# assert_is_legal_feature — the ROLE guard
# ---------------------------------------------------------------------------

def test_legal_feature_passes_silently():
    # A real FEATURE column must not raise.
    contract.assert_is_legal_feature("mmkt_balance")


def test_every_declared_feature_passes():
    # The guard must accept every column the contract itself calls a feature.
    for name in contract.feature_columns():
        contract.assert_is_legal_feature(name)


def test_forbidden_column_raises_leakage():
    # raw_day_index is FORBIDDEN — using it as a feature is the whole point of
    # the ban, so the guard must stop it.
    with pytest.raises(LeakageError):
        contract.assert_is_legal_feature("raw_day_index")


def test_label_column_rejected_as_feature():
    # The label (fdic_balance) read AS A FEATURE is the classic leak. The
    # lagged copy (fdic_balance_lag1) is the legal form; the bare label is not.
    with pytest.raises(LeakageError):
        contract.assert_is_legal_feature("fdic_balance")


def test_metadata_column_rejected_as_feature():
    with pytest.raises(LeakageError):
        contract.assert_is_legal_feature("portfolio_id")


def test_unknown_column_raises_keyerror():
    # A column not in the contract at all is a KeyError (from get_spec), not a
    # LeakageError — "you never declared this" is a different failure than
    # "you declared this and it's illegal".
    with pytest.raises(KeyError):
        contract.assert_is_legal_feature("column_that_does_not_exist")


# ---------------------------------------------------------------------------
# assert_feature_date_is_legal — the TEMPORAL guard (the < T boundary)
# ---------------------------------------------------------------------------

T = dt.date(2025, 6, 27)
T_MINUS_1 = dt.date(2025, 6, 26)


def test_prior_business_day_feature_legal_when_strictly_before_T():
    # Reading mmkt_balance from T-1 is exactly what we want — no raise.
    contract.assert_feature_date_is_legal("mmkt_balance", T_MINUS_1, T)


def test_prior_business_day_feature_leaks_when_read_on_T():
    # feature_date == T is the leak: not strictly before T.
    with pytest.raises(LeakageError):
        contract.assert_feature_date_is_legal("mmkt_balance", T, T)


def test_prior_business_day_feature_leaks_when_read_after_T():
    after = dt.date(2025, 6, 30)
    with pytest.raises(LeakageError):
        contract.assert_feature_date_is_legal("mmkt_balance", after, T)


def test_known_in_advance_feature_legal_when_record_visible_before_T():
    # Scheduled features gate on RECORD VISIBILITY, but the boundary is the
    # same: the record must be visible strictly before T.
    contract.assert_feature_date_is_legal(
        "scheduled_distribution_flag", T_MINUS_1, T
    )


def test_known_in_advance_feature_leaks_when_visible_only_on_T():
    # A schedule record first visible ON T is the same-day-announcement leak.
    with pytest.raises(LeakageError):
        contract.assert_feature_date_is_legal(
            "scheduled_distribution_flag", T, T
        )


def test_temporal_guard_rejects_non_feature():
    # Asking the temporal guard about the label is a category error — the label
    # has no feature read at all, so the guard refuses rather than waving it by.
    with pytest.raises(LeakageError):
        contract.assert_feature_date_is_legal("fdic_balance", T_MINUS_1, T)


# ---------------------------------------------------------------------------
# Convenience lookups — derived from the spec, so pin their contract
# ---------------------------------------------------------------------------

def test_label_column_is_fdic_balance():
    assert contract.label_column() == "fdic_balance"


def test_feature_columns_exclude_label_metadata_forbidden():
    feats = set(contract.feature_columns())
    assert "fdic_balance" not in feats        # label
    assert "portfolio_id" not in feats        # metadata
    assert "snapshot_date" not in feats       # metadata
    assert "raw_day_index" not in feats       # forbidden
    # and it DOES include a known feature
    assert "mmkt_balance" in feats


def test_forbidden_columns_contains_raw_day_index():
    assert "raw_day_index" in contract.forbidden_columns()


def test_lagged_features_are_the_must_lag_ones():
    lagged = set(contract.lagged_feature_columns())
    # the dangerous allocation features must be in here
    assert {"mmkt_balance", "mmkt_remaining", "mmkt_total",
            "other_instruments_balance", "fdic_balance_lag1"} <= lagged
    # a KNOWN_IN_ADVANCE feature (computed from the date) is NOT a must-lag read
    assert "day_of_week" not in lagged


def test_get_spec_raises_on_unknown_column():
    with pytest.raises(KeyError):
        contract.get_spec("not_a_real_column")


# ---------------------------------------------------------------------------
# assert_contract_is_wellformed — the checker must itself have teeth.
# We monkeypatch contract.COLUMN_SPECS with deliberately broken specs and
# assert the checker rejects each. (The real spec is restored automatically by
# monkeypatch after each test.)
# ---------------------------------------------------------------------------

def test_real_contract_is_wellformed():
    # The shipped contract must pass its own check.
    contract.assert_contract_is_wellformed()


def _one_label_spec() -> ColumnSpec:
    return ColumnSpec(name="fdic_balance", role=Role.LABEL,
                      lag=LagRule.NOT_APPLICABLE)


def _a_feature(name: str) -> ColumnSpec:
    return ColumnSpec(name=name, role=Role.FEATURE,
                      lag=LagRule.PRIOR_BUSINESS_DAY)


def test_wellformed_rejects_duplicate_names(monkeypatch):
    bad = (_one_label_spec(), _a_feature("dup"), _a_feature("dup"))
    monkeypatch.setattr(contract, "COLUMN_SPECS", bad)
    with pytest.raises(AssertionError):
        contract.assert_contract_is_wellformed()


def test_wellformed_rejects_zero_labels(monkeypatch):
    bad = (_a_feature("x"), _a_feature("y"))
    monkeypatch.setattr(contract, "COLUMN_SPECS", bad)
    with pytest.raises(AssertionError):
        contract.assert_contract_is_wellformed()


def test_wellformed_rejects_two_labels(monkeypatch):
    bad = (
        ColumnSpec(name="label_a", role=Role.LABEL, lag=LagRule.NOT_APPLICABLE),
        ColumnSpec(name="label_b", role=Role.LABEL, lag=LagRule.NOT_APPLICABLE),
    )
    monkeypatch.setattr(contract, "COLUMN_SPECS", bad)
    with pytest.raises(AssertionError):
        contract.assert_contract_is_wellformed()


def test_wellformed_rejects_feature_without_lag_rule(monkeypatch):
    bad = (
        _one_label_spec(),
        ColumnSpec(name="lazy_feature", role=Role.FEATURE,
                   lag=LagRule.NOT_APPLICABLE),  # feature must declare a lag
    )
    monkeypatch.setattr(contract, "COLUMN_SPECS", bad)
    with pytest.raises(AssertionError):
        contract.assert_contract_is_wellformed()


def test_wellformed_rejects_label_carrying_a_lag_rule(monkeypatch):
    bad = (
        ColumnSpec(name="fdic_balance", role=Role.LABEL,
                   lag=LagRule.PRIOR_BUSINESS_DAY),  # label must not carry a lag
    )
    monkeypatch.setattr(contract, "COLUMN_SPECS", bad)
    with pytest.raises(AssertionError):
        contract.assert_contract_is_wellformed()