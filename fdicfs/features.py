"""
features.py — piece 4 — get_features_asof(T)
============================================

The real feature builder. Replaces features_stub.py. A THIN layer on top of
the snapshot chokepoint: ask snapshot.features_asof for an AsOfState window,
then build the feature vector from state.balances / state.schedule plus the
calendar (relative-time) features computed from the DATE T.

Why this file is structurally incapable of leaking
---------------------------------------------------
It never reads a source directly and never sees T as a data date. It asks the
chokepoint (snapshot.features_asof), which resolves T-1 via calendar, enforces
the contract at the door, and returns only <= T-1 state. T reaches a source on
exactly one path — the LABEL read — which lives in snapshot.label_asof and is
not called here.

Three families of feature, all <= T-1 legal:
  * Lagged balance/allocation features — the dangerous ones (FDIC & same-day
    MMKT are two halves of one EOD sweep), pulled from the T-1 row.
  * Suspected-static attributes — pulled from <=T-1 regardless; if truly
    static, T-1 == T and nothing is lost; if reassigned, the hole is closed.
  * Cyclical / relative-time features — computed from the DATE T alone
    (calendar.py), never from T's data. Legal because they need no T snapshot.

The output keys are validated against the temporal contract: every key must be
a declared FEATURE column. That turns "we only emit legal features" from a hope
into a mechanical check.
"""

from __future__ import annotations

import datetime as dt

from fdicfs import calendar, contract, snapshot
from fdicfs.sources import synthetic as src


def _t_minus_1_row(state: snapshot.AsOfState):
    """The T-1 balance row, found by EXPLICIT date match, not row position.

    Decision (row-order loose thread): rather than trust the source's
    ascending sort and take iloc[-1], we select the row whose snapshot_date
    equals the resolved as_of (== T-1). This is robust if a future source
    (Snowflake) returns rows unsorted, and it fails loudly if the T-1 row is
    somehow absent rather than silently grabbing the wrong row.
    """
    match = state.balances[state.balances[src.SNAPSHOT_DATE] == state.as_of]
    if match.empty:
        raise ValueError(
            f"No balance row dated {state.as_of} (resolved T-1) in the feature "
            f"window. Cannot build lagged features."
        )
    if len(match) > 1:
        raise ValueError(
            f"Multiple balance rows dated {state.as_of} — grain is not one row "
            f"per (portfolio, date) as the contract assumes."
        )
    return match.iloc[0]


def get_features_asof(
    T: dt.date,
    portfolio_id: str = src.P,
    *,
    balance_source=None,
    schedule_source=None,
) -> dict:
    """Build the <= T-1 feature vector for predicting day `T`.

    Routes through snapshot.features_asof (the chokepoint). The balance_source
    / schedule_source overrides are forwarded so the poison test can swap one
    read at a time; defaulting to None lets snapshot own the real-source
    fallback. Returns a plain dict so callers can compare with ==.
    """
    state = snapshot.features_asof(
        T,
        portfolio_id,
        source=src,
        balance_source=balance_source,
        schedule_source=schedule_source,
    )

    # The T-1 row, by explicit date match (see _t_minus_1_row).
    last = _t_minus_1_row(state)

    features = {
        # --- lagged balance / allocation (the dangerous, must-lag features) ---
        "fdic_balance_lag1": float(last[src.FDIC_BALANCE]),
        "mmkt_balance_lag1": float(last[src.MMKT_BALANCE]),
        "mmkt_remaining_lag1": float(last[src.MMKT_REMAINING]),
        "mmkt_total_lag1": float(last[src.MMKT_TOTAL]),
        "other_instruments_balance_lag1": float(last[src.OTHER_INSTRUMENTS_BALANCE]),

        # --- suspected-static attributes, pulled from <= T-1 ---
        "portfolio_purpose": str(last[src.PORTFOLIO_PURPOSE]),
        "strategy_flag": str(last[src.STRATEGY_FLAG]),
        "bucket": str(last[src.BUCKET]),

        # --- scheduled-event feature (visibility already gated by the chokepoint) ---
        "scheduled_distribution_flag": int(len(state.schedule) > 0),
        "scheduled_distribution_amount": float(
            state.schedule[src.SCHEDULED_DISTRIBUTION_AMOUNT].sum()
        ),

        # --- cyclical / relative-time, computed from the DATE T (calendar.py) ---
        "day_of_week": T.weekday(),
        "business_day_of_month": calendar.business_day_of_month(T),
        "days_to_quarter_end": calendar.days_to_quarter_end(T),
    }

    _assert_keys_are_legal_features(features)
    return features


# ---------------------------------------------------------------------------
# Output guard: every emitted key must be a declared FEATURE in the contract.
# Maps the lagged feature names (e.g. mmkt_balance_lag1) back to their contract
# column (mmkt_balance) before checking, since the contract names the column,
# not the lag-suffixed feature.
# ---------------------------------------------------------------------------

def _contract_name(feature_key: str) -> str:
    """Map an output key to the column name the contract actually declares.

    The contract is deliberately inconsistent about the _lag1 suffix: the
    lagged LABEL is declared WITH it (fdic_balance_lag1, to distinguish the
    legal lagged copy from the label itself), while the must-lag allocation
    features are declared WITHOUT it (mmkt_balance, not mmkt_balance_lag1).

    So we resolve to whichever form exists in the contract: prefer the key as
    written, and only strip _lag1 if the bare name is the declared column.
    """
    declared = set(contract.feature_columns()) | {contract.label_column()}
    if feature_key in declared:
        return feature_key
    if feature_key.endswith("_lag1"):
        stripped = feature_key[: -len("_lag1")]
        if stripped in declared:
            return stripped
    return feature_key  # unknown — let assert_is_legal_feature raise on it


def _assert_keys_are_legal_features(features: dict) -> None:
    for key in features:
        contract.assert_is_legal_feature(_contract_name(key))


if __name__ == "__main__":
    print("Clean features for T:", get_features_asof(src.T))
