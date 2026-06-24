"""
dataset.py — piece 6 — assemble the training table across a date range.
=======================================================================

Pieces 1-5 produced a feature vector (and a label) for ONE predicted day T.
This file does the only thing left before modelling: walk a range of days and
turn each one into a single training example —

    ( feature vector drawn strictly from snapshots <= T-1 ,  label = day-T balance )

It is the ONLY place in the whole project where day T is read as DATA, and even
here T is read exactly once, as the LABEL, through the physically separate
`snapshot.label_asof` door. The features for that same T come from
`features.get_features_asof`, which never touches T (it routes through the
snapshot chokepoint that consumes T into T-1 on its first line). So this is
where the two halves finally sit side by side in one row — and the architecture
guarantees they arrived from two different reads at two different dates. That
separation is the whole project; here is where it pays off.

What "walk the range" means
---------------------------
We iterate every BUSINESS day T in [start, end] (calendar.is_business_day
decides; weekends/holidays are skipped). For each T, one of three things happens:

  1. EXAMPLE built — features (<=T-1) + label (T) → one row.
  2. RAMP-UP skip — no <=T-1 history exists yet (the very start of the range),
     so features_asof raises and we cannot train on no data. We SKIP and COUNT
     (decided: a reported skip beats a silent crash on the first valid-looking
     date).
  3. MISSING-LABEL skip — features exist but the day-T balance does not (T is
     past the end of the data / in the future). A day we can featurize but
     cannot grade is not a training example. Also skipped + counted, but logged
     louder, because a business day with prior history yet no snapshot is more
     surprising than a ramp-up day.

The BuildReport carries those counts back so the caller sees exactly what was
dropped and why, rather than trusting log lines to have been read.

The Snowflake swap touches THIS file too
-----------------------------------------
`features.get_features_asof` currently bakes in the synthetic source, and this
file reads the LABEL from the synthetic source as well — so features and label
come from the SAME backend. A mismatch there (features from one source, label
from another) would be a silent leak-shaped bug, so it is called out, not
hidden. When `sources/snowflake.py` lands, the swap is a coordinated call-site
change in BOTH features.py and here.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import pandas as pd

from fdicfs import calendar, contract, snapshot
from fdicfs.features import get_features_asof
from fdicfs.sources import synthetic as src


log = logging.getLogger(__name__)

# The two non-feature columns, derived from the contract so they cannot drift:
#   LABEL_COL — the answer, the day-T balance (the contract names it).
#   DATE_COL  — the T each row predicts. METADATA: walk-forward validation uses
#               it to ORDER and SPLIT folds; it is never fed to the model. Its
#               calendar value is known in advance, so carrying it is not a leak
#               (same reason day_of_week is a legal feature).
LABEL_COL = contract.label_column()      # "fdic_balance"
DATE_COL = "prediction_date"


@dataclass
class BuildReport:
    """What the build did: examples kept, days skipped, and why.

    Kept as DATA (not just log output) so a caller or notebook can assert on it
    — e.g. "exactly one ramp-up day at the start of this range, nothing else
    dropped". A silent count is only trustworthy if you can test it.
    """

    n_examples: int = 0
    ramp_up_dates: list[dt.date] = field(default_factory=list)       # empty <=T-1 window
    missing_label_dates: list[dt.date] = field(default_factory=list)  # no day-T label

    @property
    def n_ramp_up_skips(self) -> int:
        return len(self.ramp_up_dates)

    @property
    def n_missing_label_skips(self) -> int:
        return len(self.missing_label_dates)

    def summary(self) -> str:
        return (
            f"{self.n_examples} example(s); "
            f"{self.n_ramp_up_skips} ramp-up skip(s); "
            f"{self.n_missing_label_skips} missing-label skip(s)."
        )


def _business_days(start: dt.date, end: dt.date):
    """Yield each business day in [start, end] INCLUSIVE, ascending.

    is_business_day handles weekends + Fed holidays (it returns a numpy.bool_,
    which behaves correctly in an `if`). Ascending order is load-bearing: the
    rows come out time-ordered, which is exactly what walk-forward validation
    (piece 7) needs — it must never shuffle time.
    """
    cur = start
    while cur <= end:
        if calendar.is_business_day(cur):
            yield cur
        cur += dt.timedelta(days=1)


def build_dataset(
    start: dt.date,
    end: dt.date,
    portfolio_id: str = src.P,
) -> tuple[pd.DataFrame, BuildReport]:
    """Walk [start, end] business day by business day; build the training table.

    Returns (df, report). Each df row is one training example, laid out as
        [DATE_COL]  +  <feature columns>  +  [LABEL_COL]
    and rows are time-ordered ascending by DATE_COL. `report` records what was
    skipped and why.
    """
    rows: list[dict] = []
    report = BuildReport()

    for T in _business_days(start, end):
        # --- FEATURES: strictly <= T-1. Raises ValueError if no history yet. ---
        # This is the ramp-up gate. We never see T here — get_features_asof
        # routes through the snapshot chokepoint, which resolved T into T-1.
        try:
            feats = get_features_asof(T, portfolio_id)
        except ValueError as exc:
            report.ramp_up_dates.append(T)
            log.info("skip %s — ramp-up (no <=T-1 history): %s", T, exc)
            continue

        # --- LABEL: the one and only read of day T, and only as the answer. ---
        # KeyError == no day-T snapshot (T is past the data / in the future): a
        # day we can featurize but cannot grade, so it is not a training example.
        try:
            label = snapshot.label_asof(T, portfolio_id, source=src)
        except KeyError as exc:
            report.missing_label_dates.append(T)
            log.warning("skip %s — features built but no day-T label: %s", T, exc)
            continue

        # --- PAIR them. Belt-and-suspenders: a feature key must never collide
        #     with the label or date column. (It shouldn't — the lagged label is
        #     fdic_balance_lag1, distinct from the label fdic_balance — but the
        #     assert turns a future rename mistake into a loud failure here.) ---
        assert LABEL_COL not in feats, (
            f"feature dict already contains the label column {LABEL_COL!r}; a "
            f"lagged feature must use a distinct name (e.g. {LABEL_COL}_lag1)."
        )
        assert DATE_COL not in feats, f"feature dict already contains {DATE_COL!r}."

        rows.append({DATE_COL: T, **feats, LABEL_COL: label})
        report.n_examples += 1

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning(
            "build_dataset produced 0 examples for [%s, %s] — was the whole "
            "range ramp-up / unlabelled?", start, end,
        )
    log.info("build_dataset [%s, %s]: %s", start, end, report.summary())
    return df, report


def split_features_label(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a built dataset into (X, y).

    X = feature columns only (DATE_COL and LABEL_COL dropped); y = the label
    series. The date deliberately stays on the original `df` so walk-forward
    validation can order folds by df[DATE_COL] BEFORE calling this per fold —
    the date is needed to split time correctly, but must not become a feature.
    """
    X = df.drop(columns=[DATE_COL, LABEL_COL])
    y = df[LABEL_COL]
    return X, y


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Walk the synthetic fixture's world, deliberately one day PAST the data
    # (Mon Jun 30) to exercise all THREE outcomes in a single run:
    #   Jun 24  — ramp-up skip       (T-1 = Jun 23, before any data exists)
    #   Jun 25, 26, 27 — examples    (three clean training rows)
    #   Jun 30  — missing-label skip (features come from the Jun 27 row, which is
    #             legitimate <=T-1 history for T=Jun 30; but the Jun 30 balance
    #             does not exist yet, so there is no label to train against —
    #             the live prediction frontier in miniature)
    start = src.D_TM3           # Tue Jun 24 2025
    end = dt.date(2025, 6, 30)  # Mon Jun 30 2025

    df, report = build_dataset(start, end)

    print("\n=== training table ===")
    print(df.to_string(index=False))
    print("\n=== report ===")
    print(report.summary())
    print("  ramp-up skips       :", report.ramp_up_dates)
    print("  missing-label skips :", report.missing_label_dates)

    X, y = split_features_label(df)
    print("\nX columns:", list(X.columns))
    print("y (labels):", list(y))