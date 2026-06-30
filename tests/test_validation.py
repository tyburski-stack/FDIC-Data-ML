"""
test_validation.py — proves the walk-forward splitter never lets time leak.

validation.py is the evaluation-level echo of the T-1/T contract: every train
date must be strictly before every test date, and (the subtle one) whole DATES
must move to one side of the boundary even when many rows share a date. These
tests pin both, plus the expanding/rolling/gap mechanics, the LeakageError
guard's teeth, the metrics, and the harness.

Data-independent: the splitter tests build their own tiny time-ordered frames
with known values; only the final integration test touches the synthetic source
through dataset.build_dataset.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fdicfs import calendar as cal, contract
from fdicfs.dataset import DATE_COL, LABEL_COL, build_dataset
from fdicfs.sources import synthetic as src
from fdicfs.validation import (
    FoldResult,
    WalkForwardSplit,
    mae,
    persistence_fit_predict,
    rmse,
    walk_forward_evaluate,
)


# ---------------------------------------------------------------------------
# Fixtures — hand-built, time-ordered, known-answer frames.
# Series is +10/day so the persistence baseline (predict the previous day's
# balance = fdic_balance_lag1) has a known MAE of exactly 10 everywhere.
# ---------------------------------------------------------------------------

def _bdays(n: int, start: dt.date = dt.date(2025, 6, 2)) -> list[dt.date]:
    """The first `n` business days on/after `start`, using the project calendar."""
    out: list[dt.date] = []
    d = start
    while len(out) < n:
        if cal.is_business_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


@pytest.fixture
def single_df() -> pd.DataFrame:
    """12 business days, one row per date (the synthetic single-portfolio shape)."""
    days = _bdays(12)
    return pd.DataFrame(
        {
            DATE_COL: days,
            "fdic_balance_lag1": [1000.0 + 10 * i - 10 for i in range(12)],  # prev day's balance
            LABEL_COL: [1000.0 + 10 * i for i in range(12)],
        }
    )


@pytest.fixture
def multi_df() -> pd.DataFrame:
    """12 business days x 2 portfolios = 24 rows (the future multi-portfolio shape).

    portfolio_id is illustrative only — the splitter never looks at it; its only
    purpose here is to put TWO rows on every date so the date-grouping property
    has something to prove.
    """
    days = _bdays(12)
    rows = []
    for i, day in enumerate(days):
        for port, off in (("A", 0.0), ("B", 500.0)):
            rows.append(
                {
                    DATE_COL: day,
                    "portfolio_id": port,
                    "fdic_balance_lag1": 1000.0 + 10 * i - 10 + off,
                    LABEL_COL: 1000.0 + 10 * i + off,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# THE BOUNDARY — the anti-leak heart. Train strictly before test, always.
# ---------------------------------------------------------------------------

def test_every_fold_trains_strictly_before_it_tests(single_df):
    s = WalkForwardSplit(n_splits=3)
    folds = list(s.split(single_df))
    assert len(folds) == 3
    for tr, te in folds:
        train_max = single_df.iloc[tr][DATE_COL].max()
        test_min = single_df.iloc[te][DATE_COL].min()
        assert train_max < test_min


def test_boundary_holds_with_many_rows_per_date(multi_df):
    s = WalkForwardSplit(n_splits=3)
    for tr, te in s.split(multi_df):
        assert multi_df.iloc[tr][DATE_COL].max() < multi_df.iloc[te][DATE_COL].min()


def test_test_folds_are_disjoint_in_time(single_df):
    # No date may appear in two different test folds.
    s = WalkForwardSplit(n_splits=3)
    seen = np.concatenate([pd.unique(single_df.iloc[te][DATE_COL]) for _, te in s.split(single_df)])
    assert len(seen) == len(set(seen))


def test_whole_date_stays_on_one_side(multi_df):
    # The subtle correctness property: a date's rows are never split across the
    # train/test boundary. With 2 portfolios, every test date contributes 2 rows.
    s = WalkForwardSplit(n_splits=3)
    for tr, te in s.split(multi_df):
        train_dates = set(multi_df.iloc[tr][DATE_COL])
        test_dates = set(multi_df.iloc[te][DATE_COL])
        assert train_dates.isdisjoint(test_dates)
        counts = multi_df.iloc[te][DATE_COL].value_counts()
        assert (counts == 2).all()  # both portfolios of each test date, together


# ---------------------------------------------------------------------------
# Expanding vs rolling window
# ---------------------------------------------------------------------------

def test_expanding_train_grows_each_fold(single_df):
    s = WalkForwardSplit(n_splits=3)  # expanding (max_train_size=None)
    sizes = [single_df.iloc[tr][DATE_COL].nunique() for tr, _ in s.split(single_df)]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)  # strictly increasing


def test_rolling_caps_train_size(single_df):
    # max_train_size pins the train window to a fixed number of recent dates.
    s = WalkForwardSplit(n_splits=2, test_size=2, max_train_size=3)
    sizes = [single_df.iloc[tr][DATE_COL].nunique() for tr, _ in s.split(single_df)]
    assert sizes == [3, 3]


def test_rolling_differs_from_expanding(single_df):
    expanding = WalkForwardSplit(n_splits=2, test_size=2)
    rolling = WalkForwardSplit(n_splits=2, test_size=2, max_train_size=3)
    exp_sizes = [single_df.iloc[tr][DATE_COL].nunique() for tr, _ in expanding.split(single_df)]
    roll_sizes = [single_df.iloc[tr][DATE_COL].nunique() for tr, _ in rolling.split(single_df)]
    assert exp_sizes != roll_sizes


# ---------------------------------------------------------------------------
# The gap (embargo)
# ---------------------------------------------------------------------------

def test_gap_inserts_embargo_between_train_and_test(single_df):
    gap = 2
    s = WalkForwardSplit(n_splits=2, test_size=2, gap=gap)
    timeline = sorted(single_df[DATE_COL].unique())
    for tr, te in s.split(single_df):
        train_max = single_df.iloc[tr][DATE_COL].max()
        test_min = single_df.iloc[te][DATE_COL].min()
        # exactly `gap` dates sit in the held-out band between train and test
        between = [d for d in timeline if train_max < d < test_min]
        assert len(between) == gap


def test_gap_zero_leaves_no_embargo(single_df):
    s = WalkForwardSplit(n_splits=3, gap=0)
    timeline = sorted(single_df[DATE_COL].unique())
    for tr, te in s.split(single_df):
        train_max = single_df.iloc[tr][DATE_COL].max()
        test_min = single_df.iloc[te][DATE_COL].min()
        between = [d for d in timeline if train_max < d < test_min]
        assert between == []  # test starts on the very next date


# ---------------------------------------------------------------------------
# Counts, shape, positional indices, robustness to row order
# ---------------------------------------------------------------------------

def test_yields_exactly_n_splits_folds(single_df):
    assert len(list(WalkForwardSplit(n_splits=4, test_size=1).split(single_df))) == 4


def test_explicit_test_size_controls_block_size(single_df):
    s = WalkForwardSplit(n_splits=2, test_size=2)
    for _, te in s.split(single_df):
        assert single_df.iloc[te][DATE_COL].nunique() == 2


def test_indices_are_positional_and_align_with_iloc(single_df):
    # First fold of an n_splits=2/test_size=1 split on 12 dates:
    # first_test_start = 12 - 2 = 10, so fold 0 tests date index 10.
    s = WalkForwardSplit(n_splits=2, test_size=1)
    tr, te = next(iter(s.split(single_df)))
    timeline = sorted(single_df[DATE_COL].unique())
    assert list(single_df.iloc[te][DATE_COL]) == [timeline[10]]
    assert single_df.iloc[tr][DATE_COL].max() < timeline[10]


def test_split_is_robust_to_unsorted_input_rows(single_df):
    # The splitter sorts the timeline internally, so shuffling the rows must not
    # change which DATES land in each fold.
    s = WalkForwardSplit(n_splits=3)
    ordered = [set(pd.unique(single_df.iloc[te][DATE_COL])) for _, te in s.split(single_df)]
    shuffled = single_df.sample(frac=1, random_state=0).reset_index(drop=True)
    reshuffled = [set(pd.unique(shuffled.iloc[te][DATE_COL])) for _, te in s.split(shuffled)]
    assert ordered == reshuffled


# ---------------------------------------------------------------------------
# Configuration errors — fail loud and early, with a readable message.
# ---------------------------------------------------------------------------

def test_constructor_rejects_bad_n_splits():
    with pytest.raises(ValueError):
        WalkForwardSplit(n_splits=0)


def test_constructor_rejects_negative_gap():
    with pytest.raises(ValueError):
        WalkForwardSplit(n_splits=2, gap=-1)


def test_constructor_rejects_zero_test_size():
    with pytest.raises(ValueError):
        WalkForwardSplit(n_splits=2, test_size=0)


def test_too_many_splits_for_the_data_raises(single_df):
    # 12 dates can't be carved into 50 test blocks.
    with pytest.raises(ValueError):
        list(WalkForwardSplit(n_splits=50).split(single_df))


def test_gap_too_large_leaves_no_training_data(single_df):
    # n_splits=3, test_size=3 -> first test starts at date index 3; a gap of 3
    # would push the train window to nothing.
    with pytest.raises(ValueError):
        list(WalkForwardSplit(n_splits=3, test_size=3, gap=3).split(single_df))


def test_missing_date_column_raises(single_df):
    with pytest.raises(KeyError):
        list(WalkForwardSplit(n_splits=2).split(single_df.drop(columns=[DATE_COL])))


# ---------------------------------------------------------------------------
# The boundary guard has TEETH — force the timeline out of order and prove it
# raises the contract's own LeakageError (the same exception a feature leak
# throws). This is the defense against a future edit to the fold arithmetic.
# ---------------------------------------------------------------------------

def test_boundary_guard_raises_leakage_if_timeline_unsorted(monkeypatch, single_df):
    import fdicfs.validation as V

    # Make the internal timeline sort a no-op, then feed rows in DESCENDING date
    # order. Now the "front" of the timeline is later than the "back", so the
    # train slice is no longer strictly before the test slice — exactly the
    # mistake the guard exists to catch.
    monkeypatch.setattr(V.np, "sort", lambda a: a)
    descending = single_df.iloc[::-1].reset_index(drop=True)
    with pytest.raises(contract.LeakageError):
        list(WalkForwardSplit(n_splits=2, test_size=2).split(descending))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_mae_known_values():
    assert mae([0, 0, 0, 0], [2, 2, 2, 2]) == 2.0
    assert mae([1, 2, 3], [1, 2, 6]) == pytest.approx(1.0)


def test_rmse_known_values():
    assert rmse([0, 0, 0, 0], [2, 2, 2, 2]) == 2.0
    assert rmse([1, 2, 3], [1, 2, 6]) == pytest.approx(np.sqrt(3.0))


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 8]  # one big miss
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)


# ---------------------------------------------------------------------------
# The evaluation harness
# ---------------------------------------------------------------------------

def test_persistence_returns_lag1(single_df):
    X = single_df.drop(columns=[DATE_COL, LABEL_COL])
    y = single_df[LABEL_COL]
    pred = persistence_fit_predict(X, y, X)
    assert list(pred) == list(single_df["fdic_balance_lag1"])


def test_evaluate_one_result_per_fold(single_df):
    s = WalkForwardSplit(n_splits=3)
    results, _ = walk_forward_evaluate(single_df, s, persistence_fit_predict)
    assert len(results) == 3
    assert all(isinstance(r, FoldResult) for r in results)


def test_evaluate_fold_boundaries_are_ordered(single_df):
    s = WalkForwardSplit(n_splits=3)
    results, _ = walk_forward_evaluate(single_df, s, persistence_fit_predict)
    for r in results:
        assert r.train_end < r.test_start                # train before test
    # and successive folds walk forward in time
    for a, b in zip(results, results[1:]):
        assert a.test_start < b.test_start


def test_evaluate_persistence_mae_is_known(single_df):
    # The +10/day series means persistence is always off by exactly 10.
    s = WalkForwardSplit(n_splits=3)
    results, _ = walk_forward_evaluate(single_df, s, persistence_fit_predict)
    for r in results:
        assert r.mae == pytest.approx(10.0)
        assert r.rmse == pytest.approx(10.0)


def test_evaluate_oof_covers_every_test_row_once(single_df):
    s = WalkForwardSplit(n_splits=3)
    _, oof = walk_forward_evaluate(single_df, s, persistence_fit_predict)
    # one row per test date (single portfolio), each test date once, time-ordered
    expected_dates = sorted(d for _, te in s.split(single_df)
                            for d in pd.unique(single_df.iloc[te][DATE_COL]))
    assert list(oof[DATE_COL]) == expected_dates
    assert list(oof[DATE_COL]) == sorted(oof[DATE_COL])  # ascending


def test_evaluate_raises_on_wrong_length_predictions(single_df):
    def broken_fit_predict(X_train, y_train, X_test):
        return np.array([])  # wrong length on purpose
    s = WalkForwardSplit(n_splits=2, test_size=2)
    with pytest.raises(ValueError):
        walk_forward_evaluate(single_df, s, broken_fit_predict)


# ---------------------------------------------------------------------------
# Integration — it plugs straight into dataset.build_dataset's output.
# ---------------------------------------------------------------------------

def test_integrates_with_build_dataset():
    # The synthetic fixture yields 3 example dates (Jun 25, 26, 27); 2 folds of
    # test_size=1 give the expanding window in miniature.
    df, _ = build_dataset(src.D_TM3, src.T)
    s = WalkForwardSplit(n_splits=2, test_size=1)
    folds = [(list(df.iloc[tr][DATE_COL]), list(df.iloc[te][DATE_COL]))
             for tr, te in s.split(df)]
    assert folds == [
        ([dt.date(2025, 6, 25)], [dt.date(2025, 6, 26)]),
        ([dt.date(2025, 6, 25), dt.date(2025, 6, 26)], [dt.date(2025, 6, 27)]),
    ]
