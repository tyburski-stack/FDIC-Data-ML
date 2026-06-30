# fdic-feature-store

A **point-in-time-correct feature store** for daily FDIC cash-balance
forecasting, plus a **mechanical proof that its temporal boundary holds**.

The forecasting model is not the deliverable. The *proven feature store* is.
A model is a commodity — anyone can fit one. A feature store with a mechanical
guarantee that it never peeks at the future is the rare, valuable part, and it
is what this repository builds. The model exists only as the consumer that
demonstrates the store works.

---

## The one rule everything rests on

We forecast the FDIC cash balance for a given day, call it **T**.

> To predict day **T**, the model may use only information already known by
> the end of the prior business day (**≤ T-1**). The actual day-**T** balance
> is the *answer* we grade against — it is never an *input*.

Each training example is therefore:

```
( feature vector drawn strictly from snapshots ≤ T-1 ,
  label = the day-T FDIC balance )
```

Features are sealed at **T-1**. The label is the *only* legitimate read of the
day-**T** snapshot. These are two physically separate reads in the
architecture — which matters, because the leaky and correct versions of this
code look nearly identical. The leak happens when a feature-read and the
label-read share one query and silently both grab **T**.

---

## Why a "feature store," not "a model"

Per business input from the BI team, FDIC balance is *operational cash*: noisy,
and unlike a normal asset balance. Daily movement decomposes into three parts:

1. **Scheduled & dated** (e.g. end-of-month distributions, known in advance) —
   the structurally predictable part. These become honest features: "a
   distribution is scheduled for T" is knowable at T-1, so it is legal to use.
2. **Ad-hoc but recorded** — timing can't be predicted, but these explain past
   spikes, which keeps validation honest.
3. **Genuinely random** — sets the accuracy floor. No model beats this.

Because a large share of the movement is irreducible, *accuracy was never the
point*. A legitimate part of the deliverable is **quantifying how much daily
movement is structurally predictable versus irreducibly event-driven**. That
is a finding, not a flaw.

---

## Architecture — 7 pieces, in dependency order

| # | Module | Role | Status |
|---|--------|------|--------|
| 1 | `contract.py` | The temporal contract: which columns are legal features (≤ T-1), which is the label (T), which are forbidden. The single source of truth. | **Built**, self-checks pass |
| 2 | `snapshot.py` | Read interface over EOD snapshots. Answers "state as-of date D" only — never reaches past D. The single chokepoint for date logic. | **Built & tested** |
| 3 | `calendar.py` | Business-day calendar. Resolves what "T-1" means across weekends, holidays, and month boundaries. Uses a real calendar library — never hand-rolled. | **Built & tested** |
| 4 | `features.py` | `get_features_asof(T)` — the centerpiece. Pulls ≤ T-1 snapshots *through snapshot.py*, builds lagged + cyclical + scheduled-event features. Structurally incapable of touching the T snapshot. | **Built**, guarded by the poison test |
| 5 | `test_poison.py` | **The real deliverable.** Corrupts the T snapshot (balance *and* schedule), asserts feature output is byte-identical. If T can't move the features, the boundary provably holds. | **Built & passing — now guarding the real `features.py`** |
| 6 | `dataset.py` | Walks the date range, pairs each ≤ T-1 feature vector with its day-T label. The *only* place T is read — and there it is the label. | **Built & tested** |
| 7 | `validation.py` | Walk-forward / expanding-window splits, never random. Validation design is the hard part; the model through it is commodity. | **Built & tested** |

Pieces 1 and 5 were built as a **pair, early** — the test is how you know the
rest is correct as you build it. The **calendar + snapshot** cluster (pieces 3
and 2) was built and tested behind it, and **features.py (piece 4)** was built
on top of the snapshot chokepoint. The connecting move — re-pointing the poison
test's import from the deleted `features_stub.py` to the real `features.py` — is
done, so the boundary proof guards the real feature builder. The final cluster,
**dataset + validation (pieces 6 and 7)**, is now built and tested on top of
that. All seven pieces are complete and data-independent; the only work left is
the Snowflake swap, which stays blocked on access.

---

## Repository layout

```
fdic-feature-store/
├── README.md
├── pyproject.toml              # deps, ruff, pytest config
├── src/fdicfs/
│   ├── contract.py             # piece 1 — the rulebook (built, self-checks pass)
│   ├── calendar.py             # piece 3 — resolves T-1 across boundaries (built & tested)
│   ├── snapshot.py             # piece 2 — as-of read interface, the chokepoint (built & tested)
│   ├── features.py             # piece 4 — get_features_asof(T) (built; poison-guarded)
│   ├── dataset.py              # piece 6 — pairs features with label (built & tested)
│   ├── validation.py           # piece 7 — walk-forward splits (built & tested)
│   ├── sources/                # swappable backends behind snapshot.py
│   │   ├── synthetic.py        #   hand-built table, known answers (built)
│   │   └── snowflake.py        #   swap in when access lands
│   └── _config.py              # date ranges, entity slice
├── tests/
│   ├── test_poison.py          # piece 5 — THE deliverable (passing, guards real features.py)
│   ├── test_contract.py        # well-formedness, guards fire (passing, 24 tests)
│   ├── test_calendar.py        # boundaries, gaps (passing)
│   ├── test_snapshot.py        # chokepoint guarantees (passing)
│   ├── test_features.py        # hand-computed spot checks (passing, 19 tests)
│   └── test_validation.py      # walk-forward boundary, date-grouping, guard (passing, 30 tests)
├── notebooks/                  # exploration, predictability decomposition
├── docs/
│   ├── temporal_contract.md    # plain-language companion (built)
│   └── assumed_schema.md       # reconcile placeholder names vs Snowflake
└── data/synthetic/             # fixture tables
```

The **`sources/` split is the load-bearing design choice.** `snapshot.py`
defines one read interface; `synthetic.py` and `snowflake.py` are
interchangeable implementations behind it. The source is **passed into**
snapshot.py, never imported by it — so the swap is a call-site change
(`source=synthetic` → `source=snowflake`) and snapshot.py itself never names a
concrete table. The same poison test that runs against synthetic data today
runs unchanged against Snowflake later.

Every path into `features.py` runs *through* `snapshot.py`. There is no direct
route from a feature to a raw table, so the leak cannot hide in a stray join —
there is only one door, and the contract is enforced at it.

---

## The chokepoint (piece 2), in brief

`snapshot.py` exposes exactly two reads, mirroring the contract's
features-vs-label split:

- **`features_asof(T, portfolio_id, source, *, balance_source, schedule_source)`**
  → resolves `as_of = calendar.prior_business_day(T)`, asserts that date legal
  via the contract, reads ≤ T-1 balances and ≤ T-1 visible schedule, and returns
  an `AsOfState(as_of, balances, schedule)` window. **T is consumed into T-1 on
  the first line and never seen again** — so it physically cannot reach a source
  on the feature path. Raises if the ≤ T-1 balance window is empty (you cannot
  train on no data).
- **`label_asof(T, portfolio_id, source, *, label_source)`** → the one
  physically separate read of T, returning the day-T FDIC balance. Same shape of
  call as the feature read, opposite date, different function — the whole
  anti-leak design in two functions.

The per-read overrides (`balance_source` / `schedule_source` / `label_source`)
exist so the poison test can swap *one* read to a corrupted version while
leaving the others clean.

---

## The leakage traps this guards against

- **The sweep halves.** FDIC balance and same-day money-market (MMKT) balance
  are two halves of the same end-of-day sweep. Using same-day MMKT to "predict"
  same-day FDIC reconstructs a closed ledger — perfect in backtest, useless
  live. All allocation features (MMKT balance, remaining/total, other
  instruments) are lagged to ≤ T-1. Because the honest signal is weak and
  noisy, leaked same-day features look *especially* seductive — which is
  exactly why the poison test earns its place.
- **The raw day-index.** Feeding a tree an absolute day counter lets it hardcode
  a regime boundary that won't generalize. Forbidden outright. Use relative
  time instead — day-of-week, business-day-of-month, days-to-quarter-end. This
  is where the institutional cash seasonality lives.
- **"Static" attributes that aren't.** Portfolio Purpose, Strategy Flag, Bucket
  look immutable, but if ever reassigned and posted in the T snapshot, reading
  them from T leaks. Pull from ≤ T-1 regardless: if truly static, T-1 == T and
  nothing is lost; if reassigned, the hole is already closed.
- **The schedule is data too.** Scheduled distributions are a feature source, so
  they obey the same contract. A scheduled event is legal only if its record is
  visible *before* its date — gating is on `record_visible_date`, **not**
  `distribution_date`. If it only appears on the day, it is secretly a leak — so
  the poison test corrupts the schedule record at T as well as the balance
  snapshot.
- **"EOD ≠ safe to use T."** End-of-day describes data *resolution*, not causal
  *ordering*. The T snapshot is captured after day-T decisions resolve, so it is
  still off-limits for features. The test: at the moment the prediction is
  needed (before T's close), does the feature value exist yet? T values don't.

---

## Data

- **Grain:** strictly end-of-day (EOD) daily snapshots. Forecasting grain is
  daily EOD to match capture resolution — modeling finer than capture would
  itself be a form of leakage.
- **Source:** Snowflake (cloud SQL warehouse). History is preserved — historic
  day-of values are not overwritten, so state can be queried as-of a past date
  and point-in-time reconstruction is viable.
- **Window:** ~2 years of good accessible data to start. Caveat: only ~8
  quarter-ends and ~24 month-ends, so few examples of the rare large events;
  widen later if it proves easy.

---

## Build strategy (≈ 3 weeks)

1. **During the Snowflake-access wait** (now): build the data-independent
   pieces. This is done — all seven: the contract; the poison test against a
   hand-fabricated synthetic table with known answers; the calendar (T-1 across
   weekend/holiday/month boundaries); the snapshot chokepoint over it; the real
   feature builder (`features.py`) with the poison test re-pointed at it; the
   dataset assembler (`dataset.py`); and walk-forward validation
   (`validation.py`) — plus the full pytest suite (105 tests) behind them.
   Synthetic data is *better* than real data for testing logic — known answers,
   and it forces the assumed schema to be written down as a checklist.
2. **When access lands:** swap the synthetic source for the Snowflake snapshot
   layer on a narrow entity slice (a few portfolio purposes, bounded date
   range). Get all 7 pieces working end-to-end, then widen.

A complete narrow pipeline proven correct beats a broad one that couldn't be
validated — and the validation is the part worth anything.

---

## Current status

**Four clusters done, all data-independent — the build is feature-complete:**

- **Cluster 1 — contract + poison test + synthetic source.** `contract.py` (+
  its plain-language companion) self-checks pass; `sources/synthetic.py` is a
  hand-built fixture with seeded traps and `poisoned_*` hooks;
  `tests/test_poison.py` passes. Its import now points at the **real
  `features.py`** (not the deleted `features_stub.py`), and the "teeth" test —
  which proves the poison test catches a deliberate leak — is now collected by
  pytest (it was previously named without the `test_` prefix and silently
  skipped). Both loose threads are closed; the boundary proof now exercises the
  real feature builder.
- **Cluster 2 — calendar + snapshot.** `calendar.py` resolves real T-1 across
  weekends and Fed holidays (`tests/test_calendar.py`, verified against
  hand-checked 2025 dates). `snapshot.py` is the chokepoint
  (`tests/test_snapshot.py`, all green): proven at the value level that the trap
  value cannot enter a feature read, that poison on T does not leak through, that
  the schedule gates on visibility not distribution date, and that the empty-data
  guard fires.
- **Cluster 3 — features.** `features.py` (piece 4) is the real
  `get_features_asof(T)`: a thin builder over the snapshot chokepoint emitting
  lagged balance/allocation features, suspected-static attributes, a
  scheduled-event feature, and the three relative-time features (day-of-week,
  business-day-of-month, days-to-quarter-end). Two hardening choices landed with
  it: the T-1 row is selected by explicit `snapshot_date == as_of` match (not
  positional `iloc[-1]`, closing the row-order thread), and every output key is
  validated against the contract as a declared FEATURE column. The two
  relative-time helpers live in `calendar.py` (one home for date logic) with
  hand-verified tests. `features_stub.py` was deleted — its history lives in git.
- **Cluster 4 — dataset + validation.** `dataset.py` (piece 6) walks a date
  range business day by business day and turns each into one training example —
  features (≤ T-1, from `features.py`) paired with the day-T label (the one
  legitimate read of T, via `snapshot.label_asof`). Three outcomes per day:
  example built, ramp-up skip (no ≤ T-1 history yet), or missing-label skip (a
  day past the data / in the future). A `BuildReport` returns the skipped dates
  as data, not just log lines, and rows come out ascending by `prediction_date`
  — the time order walk-forward depends on. `validation.py` (piece 7) is the
  walk-forward / expanding-window splitter: it splits on **unique dates** (so
  many rows sharing a date — e.g. multiple portfolios — never straddle the
  train/test boundary), enforces *train strictly before test* mechanically
  (raising the contract's own `LeakageError` if violated), and supports rolling
  windows (`max_train_size`) and an embargo `gap` (0 here, correct for a
  point-in-time single-day label). A thin harness runs any model callable across
  the folds and returns per-fold MAE/RMSE plus the out-of-fold prediction track,
  benchmarked against a persistence baseline.

**Test coverage — the full suite is 105 tests, all green.** Every pipeline layer
has a dedicated file: `test_calendar.py` (boundary/gap math), `test_snapshot.py`
(chokepoint guarantees), `test_poison.py` (the boundary proof + its teeth test),
`test_contract.py` (24 — the `LeakageError` guards actually fire on FORBIDDEN /
bare-label / metadata / past-T inputs, and `assert_contract_is_wellformed` is
given teeth via monkeypatched broken specs), `test_features.py` (19 — exact
feature values pinned against the synthetic fixture; every `_lag1` value asserted
`== T-1` and `!= T` trap), and `test_validation.py` (30 — the strictly-before
boundary, whole-date grouping under multiple rows per date, disjoint test folds,
expanding-vs-rolling, the gap embargo, the `LeakageError` guard firing when the
timeline is forced out of order, the metrics, and the harness). Together: poison
proves the boundary holds; `test_features` proves the output inside it is right;
`test_contract` proves the rules stop you when you cross them; `test_validation`
proves the *evaluation* can't shuffle time either.

**Next:** nothing data-independent remains — the pipeline is complete end-to-end
against the synthetic source. The remaining work is the **Snowflake swap**: drop
`sources/snowflake.py` in behind the chokepoint (same two read signatures as
`synthetic.py`) on a narrow entity slice, flip the source at the `features.py`
and `dataset.py` call sites in lockstep, and run all 7 pieces against real data.
The same poison and validation tests run unchanged.

---

## Open items (to resolve when access lands)

- Confirm a balance can be queried "as-of" an arbitrary past date on the narrow
  slice.
- Locate the scheduled-distribution data and run the PIT-safety check: does the
  schedule record appear *before* the distribution date, or only on the day?
  How far ahead is a distribution known? Is the amount known in advance, or only
  the date? (Date-only is still a useful feature.) **Note:** `features.py`
  currently emits `scheduled_distribution_amount` as the sum of visible
  amounts, on the optimistic assumption that amounts are known in advance. If
  the PIT check shows only the *date* is knowable, drop that feature to
  date-only and keep just `scheduled_distribution_flag`.
- Confirm whether the "static" attributes are ever reassigned.
- **Reconcile the holiday set.** Our calendar uses `USFederalHolidayCalendar`
  (counts the day after Thanksgiving as a business day; includes Juneteenth from
  2021). Confirm this matches the data's actual calendar — Fed/bank vs NYSE
  differ (e.g. Good Friday). The calendar tests record the verified facts about
  *our* calendar.
- Confirm the Snowflake source returns rows sorted ascending by snapshot_date.
  `features.py` no longer depends on this (it selects the T-1 row by explicit
  `snapshot_date == as_of` match), but `dataset.py`'s range walk and any future
  positional logic still assume the ascending sort contract.
- Widen beyond 2 years if it turns out to be easy (more month/quarter-end
  examples).

---

## Getting started

```bash
# scaffold (one time)
mkdir -p src/fdicfs/sources tests notebooks docs data/synthetic
touch src/fdicfs/__init__.py src/fdicfs/sources/__init__.py

# install the package so `from fdicfs...` imports resolve
pip install -e .

# self-check the contract
python -m fdicfs.contract

# run the test suite (the poison test is the one that matters)
python -m pytest tests/ -v
```