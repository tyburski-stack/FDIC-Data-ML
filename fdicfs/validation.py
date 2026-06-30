"""
validation.py — piece 7 — walk-forward (expanding-window) validation.
=====================================================================

The model is commodity; THIS is the part worth anything. A feature store that
provably never peeks at the future (pieces 1–6) can still be evaluated in a way
that secretly peeks at the future — and then every honest thing upstream is
wasted. Walk-forward validation is the evaluation-level version of the same
T-1/T rule the whole project rests on:

    Every training date must be strictly BEFORE every test date.
    Never train on the future to score the past. Never shuffle time.

Standard k-fold cross-validation shuffles rows, so a fold can train on August
to score March. For a time series that is a leak: the model is graded on
information order it will never have live. Walk-forward fixes this by only ever
moving forward in time.

Expanding window (the default here)
-----------------------------------
Sort the timeline. Carve the tail into `n_splits` contiguous TEST blocks. For
each block, TRAIN on everything before it (the train set EXPANDS as we walk
forward), TEST on the block:

    fold 0:  train [........]            test [###]
    fold 1:  train [...........]         test    [###]
    fold 2:  train [..............]      test       [###]
                    (train grows →)          (test walks →)

A rolling window (fixed-size train that slides instead of growing) is available
via `max_train_size`; expanding is the default because more history is usually
better and we are data-poor on rare events (only ~12 quarter-ends in the whole
sample).

The one correctness property that is easy to get wrong
------------------------------------------------------
We split by DATE, never by row position. In the real store there are MANY rows
per day (one per portfolio), so a naive positional split could put some
portfolios of day D in train and the rest of day D in test — same day on both
sides of the boundary, a contamination leak. So the fold boundaries are drawn
on the sorted UNIQUE dates, and every row of a date goes wholly to one side.
With a single portfolio this is invisible; with many it is the whole ballgame.

And because a shuffled-time split is the same crime as a leaked feature, the
boundary is enforced mechanically: if any fold's max train date is not strictly
before its min test date, we raise `contract.LeakageError` — the same exception
the feature contract throws. The math below guarantees it can't fire; the guard
is there so a future edit to the fold arithmetic fails loudly instead of
silently shuffling time.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from fdicfs import contract
from fdicfs.dataset import DATE_COL, LABEL_COL, split_features_label


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The splitter — date-aware, expanding by default, sklearn-shaped API.
# ---------------------------------------------------------------------------

class WalkForwardSplit:
    """Walk-forward splitter that yields (train_idx, test_idx) POSITIONAL indices.

    Shaped like sklearn's TimeSeriesSplit so it reads familiarly, but with two
    deliberate differences that matter for this project:

      * It splits on UNIQUE DATES, not row positions, so multiple rows sharing a
        date (e.g. many portfolios on the same day) never straddle the
        train/test boundary.
      * It enforces the strictly-before boundary itself, raising LeakageError if
        it is ever violated.

    Parameters
    ----------
    n_splits : number of test folds (each a contiguous block of dates).
    test_size : number of consecutive DATES in each test fold. If None, it is
        derived as ``n_dates // (n_splits + 1)`` — the same even-ish partition
        sklearn uses, which also fixes the size of the initial train block.
    gap : number of dates to EMBARGO between the end of train and the start of
        test (default 0). For a point-in-time single-day label like ours, 0 is
        correct — there is no forward-looking label window to bleed across the
        boundary. Exposed because the moment a label becomes multi-day
        (e.g. a 5-day-ahead change), you need gap >= the label horizon to stop
        train labels from overlapping the test period. Left at 0, documented.
    max_train_size : if set, the train window keeps at most this many of the
        most-recent dates (a ROLLING window). If None (default), the train set
        EXPANDS to include all prior dates.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: int | None = None,
        gap: int = 0,
        max_train_size: int | None = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1.")
        if gap < 0:
            raise ValueError("gap must be >= 0.")
        if test_size is not None and test_size < 1:
            raise ValueError("test_size, if given, must be >= 1.")
        if max_train_size is not None and max_train_size < 1:
            raise ValueError("max_train_size, if given, must be >= 1.")
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size

    def get_n_splits(self) -> int:
        return self.n_splits

    def __repr__(self) -> str:
        kind = "expanding" if self.max_train_size is None else f"rolling({self.max_train_size})"
        return (
            f"WalkForwardSplit(n_splits={self.n_splits}, test_size={self.test_size}, "
            f"gap={self.gap}, {kind})"
        )

    def split(
        self, df: pd.DataFrame, date_col: str = DATE_COL
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) positional index arrays for each fold.

        Indices are POSITIONAL (use ``df.iloc[idx]``). The split is computed on
        the sorted unique values of ``df[date_col]``, then mapped back to every
        row carrying a date in the fold — so the split is robust to row order
        and to multiple rows per date. (Input is expected ascending by date per
        the dataset.py contract, but correctness here does not depend on it.)
        """
        if date_col not in df.columns:
            raise KeyError(
                f"{date_col!r} not in df. validation needs the prediction-date "
                f"column dataset.py emits to order folds in time."
            )

        all_dates = df[date_col].to_numpy()
        unique_dates = np.sort(pd.unique(all_dates))  # ascending timeline
        n = len(unique_dates)

        test_size = self.test_size or (n // (self.n_splits + 1))
        if test_size < 1:
            raise ValueError(
                f"Not enough distinct dates ({n}) for {self.n_splits} folds. "
                f"Reduce n_splits, set a smaller test_size, or widen the range."
            )

        first_test_start = n - self.n_splits * test_size
        # Need at least one training date before the FIRST fold's test block,
        # after accounting for the embargo gap.
        if first_test_start - self.gap < 1:
            raise ValueError(
                f"Configuration leaves no training data before the first fold: "
                f"{n} dates, n_splits={self.n_splits}, test_size={test_size}, "
                f"gap={self.gap}. Reduce n_splits/test_size/gap or add more history."
            )

        for k in range(self.n_splits):
            test_start = first_test_start + k * test_size
            test_end = test_start + test_size
            train_end = test_start - self.gap
            if self.max_train_size is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.max_train_size)

            train_dates = unique_dates[train_start:train_end]
            test_dates = unique_dates[test_start:test_end]

            # The mechanical boundary check — the validation-level anti-leak.
            # Same exception the feature contract raises: shuffling time is the
            # same kind of error as reading T.
            if not (train_dates.max() < test_dates.min()):
                raise contract.LeakageError(
                    f"VALIDATION LEAK in fold {k}: max train date "
                    f"{train_dates.max()!r} is not strictly before min test date "
                    f"{test_dates.min()!r}. A walk-forward fold must never train "
                    f"on a date >= a test date."
                )

            train_idx = np.flatnonzero(np.isin(all_dates, train_dates))
            test_idx = np.flatnonzero(np.isin(all_dates, test_dates))
            yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Metrics — tiny, no sklearn dependency.
# ---------------------------------------------------------------------------

def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# The evaluation harness — runs a model through the folds, in time order.
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """One fold's boundaries and scores — kept as DATA so it can be asserted on."""

    fold: int
    n_train_dates: int
    n_test_dates: int
    n_train_rows: int
    n_test_rows: int
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date
    mae: float
    rmse: float


# A model is supplied as a callable so validation stays model-agnostic (the
# model is the commodity part). Signature: given the fold's train X/y and the
# test X, return predictions aligned to the test rows.
#   For a sklearn estimator `est` (after encoding the categorical features,
#   which is the modeller's job, not validation's):
#       fit_predict = lambda Xtr, ytr, Xte: est.fit(Xtr, ytr).predict(Xte)
FitPredict = Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]


def walk_forward_evaluate(
    df: pd.DataFrame,
    splitter: WalkForwardSplit,
    fit_predict: FitPredict,
    *,
    date_col: str = DATE_COL,
) -> tuple[list[FoldResult], pd.DataFrame]:
    """Run `fit_predict` across the splitter's folds; collect scores + OOF preds.

    Returns
    -------
    results : one FoldResult per fold (boundaries + MAE/RMSE).
    oof : the out-of-fold prediction series — every test row's date, true label
        and prediction, concatenated across folds in time order. Because the
        test blocks are disjoint and walk forward, this IS the honest
        walk-forward prediction track you would plot or score as a whole.
    """
    results: list[FoldResult] = []
    oof_chunks: list[pd.DataFrame] = []

    for k, (tr, te) in enumerate(splitter.split(df, date_col=date_col)):
        train_df = df.iloc[tr]
        test_df = df.iloc[te]

        X_train, y_train = split_features_label(train_df)
        X_test, y_test = split_features_label(test_df)

        y_pred = np.asarray(fit_predict(X_train, y_train, X_test), dtype=float)
        if len(y_pred) != len(y_test):
            raise ValueError(
                f"fold {k}: model returned {len(y_pred)} predictions for "
                f"{len(y_test)} test rows."
            )

        results.append(
            FoldResult(
                fold=k,
                n_train_dates=train_df[date_col].nunique(),
                n_test_dates=test_df[date_col].nunique(),
                n_train_rows=len(train_df),
                n_test_rows=len(test_df),
                train_start=train_df[date_col].min(),
                train_end=train_df[date_col].max(),
                test_start=test_df[date_col].min(),
                test_end=test_df[date_col].max(),
                mae=mae(y_test, y_pred),
                rmse=rmse(y_test, y_pred),
            )
        )

        oof_chunks.append(
            pd.DataFrame(
                {
                    date_col: test_df[date_col].to_numpy(),
                    "y_true": np.asarray(y_test, dtype=float),
                    "y_pred": y_pred,
                    "fold": k,
                }
            )
        )

    oof = (
        pd.concat(oof_chunks, ignore_index=True).sort_values(date_col).reset_index(drop=True)
        if oof_chunks
        else pd.DataFrame(columns=[date_col, "y_true", "y_pred", "fold"])
    )
    return results, oof


# ---------------------------------------------------------------------------
# Baselines — the bar a real model has to clear. Persistence ("tomorrow looks
# like the last business day") is the natural one here: it just echoes the
# lagged FDIC balance the feature builder already provides.
# ---------------------------------------------------------------------------

def persistence_fit_predict(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    """Predict the previous business day's FDIC balance (fdic_balance_lag1).

    No fitting at all — it returns a feature straight through. If a fitted model
    cannot beat this on walk-forward MAE, the model is adding nothing.
    """
    return X_test["fdic_balance_lag1"].to_numpy(dtype=float)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # -- Demo 1: it plugs straight into dataset.py's output -------------------
    # The synthetic fixture yields only 3 example dates, which is enough to show
    # the expanding window in miniature: 2 folds, train growing by one date.
    from fdicfs.dataset import build_dataset
    from fdicfs.sources import synthetic as src

    df_real, _ = build_dataset(src.D_TM3, src.T)  # Jun 25, 26, 27 (3 examples)
    print("=== Demo 1: real dataset.py output (3 dates) ===")
    splitter = WalkForwardSplit(n_splits=2, test_size=1)
    for k, (tr, te) in enumerate(splitter.split(df_real)):
        tr_dates = list(df_real.iloc[tr][DATE_COL])
        te_dates = list(df_real.iloc[te][DATE_COL])
        print(f"  fold {k}: train {tr_dates}  ->  test {te_dates}")

    # -- Demo 2: many rows per date (the multi-portfolio correctness property) -
    # Hand-built 12-business-day timeline with TWO portfolios per day (24 rows).
    # The real dataset.py is single-portfolio today, so this frame is illustrative
    # of the future shape — its only job is to show that whole DATES move to one
    # side of the boundary even when several rows share a date.
    from fdicfs import calendar as cal

    days: list[dt.date] = []
    d = dt.date(2025, 6, 2)  # a Monday
    while len(days) < 12:
        if cal.is_business_day(d):
            days.append(d)
        d += dt.timedelta(days=1)

    rows = []
    for i, day in enumerate(days):
        for port, offset in (("PORT_A", 0.0), ("PORT_B", 500.0)):
            label = 1000.0 + 10.0 * i + offset      # smooth +10/day series
            rows.append(
                {
                    DATE_COL: day,
                    "portfolio_id": port,            # illustrative only
                    "fdic_balance_lag1": label - 10.0,  # = previous day's balance
                    LABEL_COL: label,
                }
            )
    df_toy = pd.DataFrame(rows)

    print("\n=== Demo 2: 12 dates x 2 portfolios (24 rows), 3 expanding folds ===")
    splitter = WalkForwardSplit(n_splits=3)  # test_size derived = 12 // 4 = 3
    print(" ", splitter)
    results, oof = walk_forward_evaluate(df_toy, splitter, persistence_fit_predict)
    for r in results:
        # Prove whole dates stay together: each test fold spans 3 dates x 2 = 6 rows.
        print(
            f"  fold {r.fold}: train {r.train_start}..{r.train_end} "
            f"({r.n_train_dates}d/{r.n_train_rows}r)  ->  "
            f"test {r.test_start}..{r.test_end} ({r.n_test_dates}d/{r.n_test_rows}r) "
            f"| MAE {r.mae:.1f}  RMSE {r.rmse:.1f}"
        )

    # Cross-fold correctness checks (these are what test_validation.py would pin):
    # Dedupe within each fold first — a date legitimately repeats inside a fold
    # (one row per portfolio); what must NOT happen is a date crossing folds.
    fold_test_dates = [
        pd.unique(df_toy.iloc[te][DATE_COL]) for _, te in splitter.split(df_toy)
    ]
    all_test_dates = np.concatenate(fold_test_dates)
    assert len(all_test_dates) == len(set(all_test_dates)), (
        "a date appeared in two different test folds"
    )
    for tr, te in splitter.split(df_toy):
        tr_max = df_toy.iloc[tr][DATE_COL].max()
        te_min = df_toy.iloc[te][DATE_COL].min()
        assert tr_max < te_min, "train/test boundary violated"
    print("  checks: test folds disjoint; every fold trains strictly before it tests.")

    print("\n=== out-of-fold prediction track (persistence baseline) ===")
    print(oof.to_string(index=False))