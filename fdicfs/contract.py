"""
temporal_contract.py
====================

THE TEMPORAL CONTRACT for the FDIC daily-balance feature store.

This module is the SINGLE SOURCE OF TRUTH for one question:
    "Which data is legal to use as a feature, and which is off-limits?"

The whole project rests on one invariant:

    A training example for predicting day T is
        ( feature vector drawn STRICTLY from snapshots <= T-1,
          label = the day-T FDIC balance ).
    Features are sealed at T-1. The label is the ONLY legitimate read
    of the day-T snapshot.

Everything in this file is structured DATA describing the columns and
their temporal rules, plus ENFORCEMENT functions that read from that data.
Nothing here touches real data. It is buildable (and testable) before
Snowflake access lands, and the poison test imports from it directly.

Design choice: the spec (the dictionaries below) and the enforcement
(the functions below) live together so they cannot silently drift apart.
The companion document `temporal_contract.md` explains this file in plain
language for a non-expert reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# 1. THE VOCABULARY  —  what role can a column play?
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """The role a column is allowed to play in a training example.

    This is the heart of the features-vs-label distinction. A column is
    EITHER something we feed the model (a feature, sealed at T-1) OR the
    answer we are trying to predict (the label, read at T). It is never
    both, and a FEATURE is never allowed to be read at T.
    """

    FEATURE = "feature"      # legal input to the model; must resolve to <= T-1
    LABEL = "label"          # the answer; the ONLY column read at day T
    METADATA = "metadata"    # identifiers / bookkeeping; never fed to the model
    FORBIDDEN = "forbidden"  # explicitly banned from ever being a feature


class LagRule(str, Enum):
    """How far back a FEATURE column must be read from.

    "T-1" here means "the most recent BUSINESS day strictly before T",
    resolved by the calendar module — not literally yesterday. Weekends,
    holidays and month boundaries are the calendar's job, not this file's.
    """

    PRIOR_BUSINESS_DAY = "<=T-1"   # standard: most recent snapshot before T
    KNOWN_IN_ADVANCE = "known_by_T-1"  # scheduled event whose RECORD must exist before its date
    NOT_APPLICABLE = "n/a"             # for LABEL / METADATA rows


# ---------------------------------------------------------------------------
# 2. THE COLUMN SPEC  —  one entry per column, the source of truth
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnSpec:
    """A single column's temporal rules.

    Attributes
    ----------
    name : the column name as it appears in the source table.
    role : FEATURE / LABEL / METADATA / FORBIDDEN.
    lag  : the temporal rule the column must obey (see LagRule).
    source_table : which physical table the column comes from. Two sources
        share this contract: the balance snapshots and the distribution
        schedule. Both must obey it.
    leak_risk : free-text note on WHY this column is dangerous, so the
        reasoning survives in one place. Empty for low-risk columns.
    suspected_static : True for fields that LOOK immutable (Portfolio
        Purpose, Strategy Flag, Bucket) but may be reassigned. We pull
        these from <=T-1 regardless: if truly static, T-1 == T and we lose
        nothing; if ever reassigned, reading T would leak. (Open item:
        confirm whether these are ever reassigned.)
    """

    name: str
    role: Role
    lag: LagRule = LagRule.NOT_APPLICABLE
    source_table: str = "balance_snapshot"
    leak_risk: str = ""
    suspected_static: bool = False


# --- The spec itself. THIS LIST IS THE CONTRACT. ---------------------------
#
# NOTE: column names are PLACEHOLDERS reflecting our *assumed* schema. When
# Snowflake access lands, reconcile these names against the real tables.
# That reconciliation is exactly the "assumed-schema checklist" the build
# plan calls for — if a real column is missing here, you found a gap.

COLUMN_SPECS: tuple[ColumnSpec, ...] = (

    # ----- THE LABEL -------------------------------------------------------
    # The one and only legitimate read of the day-T snapshot.
    ColumnSpec(
        name="fdic_balance",
        role=Role.LABEL,
        lag=LagRule.NOT_APPLICABLE,
        source_table="balance_snapshot",
        leak_risk=(
            "This is the answer. When used as the label it is read at T. "
            "It must NEVER be read at T as a feature. A lagged copy "
            "(fdic_balance at <=T-1) IS a legal feature and must be a "
            "physically separate read."
        ),
    ),

    # ----- DANGEROUS ALLOCATION FEATURES -----------------------------------
    # FDIC balance and same-day MMKT are two halves of the same EOD sweep.
    # Using same-day MMKT to predict same-day FDIC reconstructs a closed
    # ledger: perfect in backtest, useless live. MUST lag to <=T-1.
    ColumnSpec(
        name="mmkt_balance",
        role=Role.FEATURE,
        lag=LagRule.PRIOR_BUSINESS_DAY,
        source_table="balance_snapshot",
        leak_risk=(
            "Same-day MMKT is the other half of the same sweep as FDIC. "
            "Same-day read = closed-ledger leak. Lag to <=T-1."
        ),
    ),
    ColumnSpec(
        name="mmkt_remaining",
        role=Role.FEATURE,
        lag=LagRule.PRIOR_BUSINESS_DAY,
        source_table="balance_snapshot",
        leak_risk="Allocation feature; same closed-ledger risk as mmkt_balance.",
    ),
    ColumnSpec(
        name="mmkt_total",
        role=Role.FEATURE,
        lag=LagRule.PRIOR_BUSINESS_DAY,
        source_table="balance_snapshot",
        leak_risk="Allocation feature; same closed-ledger risk as mmkt_balance.",
    ),
    ColumnSpec(
        name="other_instruments_balance",
        role=Role.FEATURE,
        lag=LagRule.PRIOR_BUSINESS_DAY,
        source_table="balance_snapshot",
        leak_risk="Allocation feature; lag to <=T-1.",
    ),

    # ----- THE LAGGED-LABEL FEATURE ----------------------------------------
    # Yesterday's FDIC balance is a legal and probably strong feature.
    ColumnSpec(
        name="fdic_balance_lag1",
        role=Role.FEATURE,
        lag=LagRule.PRIOR_BUSINESS_DAY,
        source_table="balance_snapshot",
        leak_risk=(
            "Legal ONLY because it resolves to <=T-1. Must be a separate "
            "read from the label. The leak hides here: a feature-read and "
            "the label-read sharing one join can silently both grab T."
        ),
    ),

    # ----- SCHEDULED-EVENT FEATURES (second source) ------------------------
    # The schedule is a feature source, so it obeys the same contract.
    # A scheduled distribution is legal to use ONLY if its RECORD is visible
    # before its distribution date. If it only appears on the day, it is
    # secretly a leak. (Open item: confirm visibility window.)
    ColumnSpec(
        name="scheduled_distribution_flag",
        role=Role.FEATURE,
        lag=LagRule.KNOWN_IN_ADVANCE,
        source_table="distribution_schedule",
        leak_risk=(
            "Legal only if the schedule record exists before its date. "
            "Poison test must corrupt the schedule record at T as well as "
            "the balance snapshot."
        ),
    ),
    ColumnSpec(
        name="scheduled_distribution_amount",
        role=Role.FEATURE,
        lag=LagRule.KNOWN_IN_ADVANCE,
        source_table="distribution_schedule",
        leak_risk=(
            "Amount may not be known in advance even when the date is. "
            "Date-only is still a useful feature; do not assume amount is "
            "known. (Open item.)"
        ),
    ),

    # ----- SUSPECTED-STATIC ATTRIBUTES -------------------------------------
    # Look immutable; pull from <=T-1 regardless to close the hole for free.
    ColumnSpec(
        name="portfolio_purpose",
        role=Role.FEATURE,
        lag=LagRule.PRIOR_BUSINESS_DAY,
        source_table="balance_snapshot",
        suspected_static=True,
        leak_risk=(
            "If ever reassigned and posted in the T snapshot, reading T "
            "leaks. Pull <=T-1: a truly static field is unchanged, so we "
            "give up nothing."
        ),
    ),
    ColumnSpec(
        name="strategy_flag",
        role=Role.FEATURE,
        lag=LagRule.PRIOR_BUSINESS_DAY,
        source_table="balance_snapshot",
        suspected_static=True,
        leak_risk="Suspected-static; pull <=T-1 in case of reassignment.",
    ),
    ColumnSpec(
        name="bucket",
        role=Role.FEATURE,
        lag=LagRule.PRIOR_BUSINESS_DAY,
        source_table="balance_snapshot",
        suspected_static=True,
        leak_risk="Suspected-static; pull <=T-1 in case of reassignment.",
    ),

    # ----- DERIVED TIME FEATURES (computed from T's DATE, not T's data) -----
    # Cyclical / relative time features are computed from the calendar date
    # T, which is known in advance — NOT from the T snapshot's contents.
    # This is legal: knowing "T is the last business day of the month" needs
    # no data from T. Contrast with raw day-index, which is FORBIDDEN.
    ColumnSpec(
        name="day_of_week",
        role=Role.FEATURE,
        lag=LagRule.KNOWN_IN_ADVANCE,
        source_table="calendar",
        leak_risk="Computed from the date T, not the T snapshot. Legal.",
    ),
    ColumnSpec(
        name="business_day_of_month",
        role=Role.FEATURE,
        lag=LagRule.KNOWN_IN_ADVANCE,
        source_table="calendar",
        leak_risk="Computed from the date T. Where institutional cash seasonality lives.",
    ),
    ColumnSpec(
        name="days_to_quarter_end",
        role=Role.FEATURE,
        lag=LagRule.KNOWN_IN_ADVANCE,
        source_table="calendar",
        leak_risk="Computed from the date T. Legal.",
    ),

    # ----- FORBIDDEN -------------------------------------------------------
    # Raw absolute day counter. A tree splits on it and hardcodes a regime
    # boundary that will not generalize. Banned outright; use cyclical /
    # relative time features above instead.
    ColumnSpec(
        name="raw_day_index",
        role=Role.FORBIDDEN,
        lag=LagRule.NOT_APPLICABLE,
        source_table="calendar",
        leak_risk=(
            "Absolute counter. Beginner tell. Hardcodes a regime boundary. "
            "Replace with cyclical/relative time features."
        ),
    ),

    # ----- METADATA --------------------------------------------------------
    ColumnSpec(
        name="portfolio_id",
        role=Role.METADATA,
        lag=LagRule.NOT_APPLICABLE,
        source_table="balance_snapshot",
        leak_risk="Identifier for joins/grain. Never fed to the model.",
    ),
    ColumnSpec(
        name="snapshot_date",
        role=Role.METADATA,
        lag=LagRule.NOT_APPLICABLE,
        source_table="balance_snapshot",
        leak_risk="The as-of date of the row. Used for date logic, not as a feature.",
    ),
)


# ---------------------------------------------------------------------------
# 3. CONVENIENCE LOOKUPS  —  derived from the spec, not hand-maintained
# ---------------------------------------------------------------------------

def _by_name() -> dict[str, ColumnSpec]:
    return {c.name: c for c in COLUMN_SPECS}


def get_spec(name: str) -> ColumnSpec:
    """Return the ColumnSpec for a column, or raise if it is not in the contract."""
    specs = _by_name()
    if name not in specs:
        raise KeyError(
            f"Column {name!r} is not in the temporal contract. Every column "
            f"that enters the pipeline must be declared here first."
        )
    return specs[name]


def feature_columns() -> list[str]:
    """All columns legally usable as features."""
    return [c.name for c in COLUMN_SPECS if c.role is Role.FEATURE]


def label_column() -> str:
    """The single label column. Raises if the contract does not define exactly one."""
    labels = [c.name for c in COLUMN_SPECS if c.role is Role.LABEL]
    if len(labels) != 1:
        raise AssertionError(
            f"The contract must define exactly one LABEL column; found {labels!r}."
        )
    return labels[0]


def forbidden_columns() -> list[str]:
    return [c.name for c in COLUMN_SPECS if c.role is Role.FORBIDDEN]


def lagged_feature_columns() -> list[str]:
    """Features that must resolve to a snapshot at <= T-1 (the dangerous ones)."""
    return [
        c.name for c in COLUMN_SPECS
        if c.role is Role.FEATURE and c.lag is LagRule.PRIOR_BUSINESS_DAY
    ]


# ---------------------------------------------------------------------------
# 4. ENFORCEMENT  —  functions that make the rules defend themselves
# ---------------------------------------------------------------------------
#
# These are the assertions that turn the spec from a description into a
# mechanical guarantee. get_features_asof() and the dataset assembler call
# these; if a rule is broken the program STOPS rather than silently leaking.

class LeakageError(AssertionError):
    """Raised when a temporal-contract rule is violated. A leak, caught."""


def assert_is_legal_feature(name: str) -> None:
    """Refuse to let a non-feature (or forbidden) column be used as a feature.

    Call this at the moment a column is about to be added to a feature
    vector. It is the guard that makes 'features come from the spec' a fact
    rather than a hope.
    """
    spec = get_spec(name)
    if spec.role is Role.FORBIDDEN:
        raise LeakageError(
            f"{name!r} is FORBIDDEN as a feature: {spec.leak_risk}"
        )
    if spec.role is not Role.FEATURE:
        raise LeakageError(
            f"{name!r} has role {spec.role.value!r}, not 'feature'. "
            f"Only FEATURE columns may enter the feature vector. "
            f"(The label and metadata are read elsewhere.)"
        )


def assert_feature_date_is_legal(name: str, feature_date, T) -> None:
    """The core temporal guard: a feature's as-of date must be strictly < T.

    `feature_date` is the snapshot date the value was actually read from.
    `T` is the day being predicted. For any FEATURE with the standard lag
    rule, reading on or after T is the leak we exist to prevent.

    Dates are compared with plain `<`; pass whatever comparable date type
    the pipeline uses (datetime.date is fine). This function does NOT do
    calendar math — resolving what "T-1" means across weekends/holidays is
    the calendar module's job. This only enforces the boundary itself.
    """
    spec = get_spec(name)
    if spec.role is not Role.FEATURE:
        raise LeakageError(
            f"{name!r} is not a feature; it should not have a feature read at all."
        )
    if spec.lag is LagRule.PRIOR_BUSINESS_DAY:
        if not (feature_date < T):
            raise LeakageError(
                f"LEAK: feature {name!r} was read from {feature_date!r}, which is "
                f"not strictly before the predicted day T={T!r}. "
                f"Features must come from <= T-1. {spec.leak_risk}"
            )
    elif spec.lag is LagRule.KNOWN_IN_ADVANCE:
        # Scheduled/calendar features: the RECORD must be visible before T.
        # Visibility (not the event's own date) is what must precede T.
        if not (feature_date < T):
            raise LeakageError(
                f"LEAK: feature {name!r} relies on a record visible at {feature_date!r}, "
                f"which is not strictly before T={T!r}. A scheduled-event or calendar "
                f"feature is only legal if its record is knowable before T. {spec.leak_risk}"
            )


def assert_contract_is_wellformed() -> None:
    """Sanity-check the spec itself: exactly one label, unique names, etc.

    Run this once at import or in a test. It guards against the contract
    being edited into an inconsistent state.
    """
    names = [c.name for c in COLUMN_SPECS]
    if len(names) != len(set(names)):
        dupes = {n for n in names if names.count(n) > 1}
        raise AssertionError(f"Duplicate column names in contract: {sorted(dupes)}")

    # Exactly one label.
    label_column()  # raises if not exactly one

    # Every FEATURE has a real lag rule (not N/A).
    for c in COLUMN_SPECS:
        if c.role is Role.FEATURE and c.lag is LagRule.NOT_APPLICABLE:
            raise AssertionError(
                f"Feature {c.name!r} has no lag rule. Every feature must declare "
                f"how far back it is read from."
            )
        if c.role in (Role.LABEL, Role.METADATA) and c.lag is not LagRule.NOT_APPLICABLE:
            raise AssertionError(
                f"{c.name!r} is {c.role.value} but declares a lag rule; that is "
                f"meaningless. Only features carry lag rules."
            )


if __name__ == "__main__":
    # Self-check the contract when run directly.
    assert_contract_is_wellformed()
    print("Temporal contract is well-formed.")
    print(f"  Label column          : {label_column()}")
    print(f"  Feature columns ({len(feature_columns()):>2})    : {feature_columns()}")
    print(f"  Must-lag features ({len(lagged_feature_columns()):>2})  : {lagged_feature_columns()}")
    print(f"  Forbidden columns     : {forbidden_columns()}")
