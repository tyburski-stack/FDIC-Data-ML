"""
test_features.py — hand-computed spot checks for get_features_asof.

The poison test proves the feature output does not CHANGE when day T is
corrupted. That is a necessary property, but it is not the same as the output
being CORRECT — a builder that always returned `{}` would also be unmoved by
poison. This file closes that gap: it pins the exact feature values against the
hand-built synthetic fixture, whose answers are known by construction.

The fixture (sources/synthetic.py):
  T   = Fri 2025-06-27   (predicted day; OFF-LIMITS to features)
  T-1 = Thu 2025-06-26   (D_TM1; the last legal feature day)
  T-1 balance row: fdic 1200, mmkt 300, remaining 60, total 360, other 540,
                   purpose "operating", strategy "A", bucket "core".
  T   balance row (the TRAP): mmkt 9_999_999, purpose "REASSIGNED", ...
  schedule: honest row amount 75 (visible T-3); trap row 9_999_999 (visible T).

So every "lag1" value below is asserted to be the T-1 number, NEVER the T trap
value — which is the value-level statement of "we read T-1, not T".
"""

import datetime as dt

import pytest

from fdicfs import calendar, contract
from fdicfs.features import get_features_asof
from fdicfs.sources import synthetic as src


@pytest.fixture(scope="module")
def feats() -> dict:
    return get_features_asof(src.T)


# ---------------------------------------------------------------------------
# Shape: exactly the expected keys, no more, no less
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "fdic_balance_lag1",
    "mmkt_balance_lag1",
    "mmkt_remaining_lag1",
    "mmkt_total_lag1",
    "other_instruments_balance_lag1",
    "portfolio_purpose",
    "strategy_flag",
    "bucket",
    "scheduled_distribution_flag",
    "scheduled_distribution_amount",
    "day_of_week",
    "business_day_of_month",
    "days_to_quarter_end",
}


def test_feature_keys_exact(feats):
    assert set(feats.keys()) == EXPECTED_KEYS


def test_every_emitted_key_is_a_legal_feature(feats):
    # Cross-check the output against the contract itself: every key must map to
    # a declared FEATURE column. This mirrors features.py's own output guard,
    # but as an independent assertion rather than trusting the builder ran it.
    for key in feats:
        contract.assert_is_legal_feature(_contract_name(key))


def _contract_name(key: str) -> str:
    """Mirror of features._contract_name: strip _lag1 only if the bare name is
    the declared column. Kept tiny and local so the test does not import a
    private helper."""
    declared = set(contract.feature_columns()) | {contract.label_column()}
    if key in declared:
        return key
    if key.endswith("_lag1"):
        stripped = key[: -len("_lag1")]
        if stripped in declared:
            return stripped
    return key


# ---------------------------------------------------------------------------
# Lagged balance/allocation features — must equal the T-1 row, never the trap
# ---------------------------------------------------------------------------

def test_fdic_balance_lag1_is_t_minus_1_value(feats):
    assert feats["fdic_balance_lag1"] == 1_200.0


def test_mmkt_balance_lag1_is_t_minus_1_not_the_trap(feats):
    # 300 is the T-1 value; 9_999_999 is the T trap. Reading the trap here is
    # exactly the closed-ledger leak the project exists to prevent.
    assert feats["mmkt_balance_lag1"] == 300.0
    assert feats["mmkt_balance_lag1"] != 9_999_999


def test_mmkt_remaining_lag1(feats):
    assert feats["mmkt_remaining_lag1"] == 60.0


def test_mmkt_total_lag1(feats):
    assert feats["mmkt_total_lag1"] == 360.0


def test_other_instruments_balance_lag1(feats):
    assert feats["other_instruments_balance_lag1"] == 540.0


def test_lagged_values_are_python_floats(feats):
    # Guard against numpy scalars sneaking into the output (they compare equal
    # but can surprise downstream serialization / strict identity checks).
    for key in ("fdic_balance_lag1", "mmkt_balance_lag1", "mmkt_remaining_lag1",
                "mmkt_total_lag1", "other_instruments_balance_lag1"):
        assert type(feats[key]) is float


# ---------------------------------------------------------------------------
# Suspected-static attributes — pulled from T-1, NOT the reassigned T values
# ---------------------------------------------------------------------------

def test_portfolio_purpose_is_t_minus_1_not_reassigned(feats):
    # T-1 says "operating"; the T row was reassigned to "REASSIGNED". Pulling
    # from T-1 is what closes the suspected-static leak.
    assert feats["portfolio_purpose"] == "operating"
    assert feats["portfolio_purpose"] != "REASSIGNED"


def test_strategy_flag_is_t_minus_1(feats):
    assert feats["strategy_flag"] == "A"   # not the T trap "Z"


def test_bucket_is_t_minus_1(feats):
    assert feats["bucket"] == "core"       # not the T trap "POISON"


def test_static_attrs_are_python_str(feats):
    for key in ("portfolio_purpose", "strategy_flag", "bucket"):
        assert type(feats[key]) is str


# ---------------------------------------------------------------------------
# Scheduled-event feature — honest row only (visible <= T-1), trap excluded
# ---------------------------------------------------------------------------

def test_scheduled_distribution_flag_is_one(feats):
    # The honest row (visible at T-3) is in the <=T-1 window, so the flag is set.
    assert feats["scheduled_distribution_flag"] == 1
    assert type(feats["scheduled_distribution_flag"]) is int


def test_scheduled_distribution_amount_is_honest_row_only(feats):
    # 75 is the honest row; 9_999_999 is the trap visible only on T. The sum
    # over the <=T-1 window must be 75 alone.
    assert feats["scheduled_distribution_amount"] == 75.0
    assert feats["scheduled_distribution_amount"] != 9_999_999


# ---------------------------------------------------------------------------
# Relative-time features — computed from the DATE T, independently re-derived
# ---------------------------------------------------------------------------

def test_day_of_week_is_friday(feats):
    # Fri 2025-06-27 -> weekday() == 4. Re-derive from the date so the test
    # documents WHY rather than just pinning a magic number.
    assert feats["day_of_week"] == 4
    assert feats["day_of_week"] == src.T.weekday()


def test_business_day_of_month_matches_calendar(feats):
    # Independently recompute via the calendar helper.
    assert feats["business_day_of_month"] == calendar.business_day_of_month(src.T)
    assert feats["business_day_of_month"] == 19   # hand-verified (Juneteenth skipped)


def test_days_to_quarter_end_matches_calendar(feats):
    # Q2 2025 ends Mon Jun 30; from Fri Jun 27 that is 1 business day out.
    assert feats["days_to_quarter_end"] == calendar.days_to_quarter_end(src.T)
    assert feats["days_to_quarter_end"] == 1


def test_relative_time_values_are_python_ints(feats):
    for key in ("day_of_week", "business_day_of_month", "days_to_quarter_end"):
        assert type(feats[key]) is int


# ---------------------------------------------------------------------------
# The whole vector, in one assertion — the canonical known-good output.
# If any value drifts, this is the single test that says "the fixture's answer
# changed" in one place.
# ---------------------------------------------------------------------------

def test_full_feature_vector_exact(feats):
    expected = {
        "fdic_balance_lag1": 1_200.0,
        "mmkt_balance_lag1": 300.0,
        "mmkt_remaining_lag1": 60.0,
        "mmkt_total_lag1": 360.0,
        "other_instruments_balance_lag1": 540.0,
        "portfolio_purpose": "operating",
        "strategy_flag": "A",
        "bucket": "core",
        "scheduled_distribution_flag": 1,
        "scheduled_distribution_amount": 75.0,
        "day_of_week": 4,
        "business_day_of_month": 19,
        "days_to_quarter_end": 1,
    }
    assert feats == expected