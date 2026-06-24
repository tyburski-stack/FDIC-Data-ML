# The Temporal Contract — in plain language

This document explains, without code, what `temporal_contract.py` guarantees
and why it exists. 
## The one rule everything rests on

We are forecasting tomorrow's FDIC cash balance. Call the day we are
predicting **T**.

> To predict day **T**, the model may only use information that was already
> known by the end of the day before (**T-1**). The actual day-**T** balance
> is the *answer* we are grading against — it is never allowed to be an
> *input*.

That is the entire project in one sentence. The contract is the written-down,
machine-enforced version of that rule.

## Why this is harder than it sounds

The correct version of this code and the broken ("leaky") version look almost
identical. The mistake happens quietly: a single database query accidentally
pulls a value from day **T** and feeds it to the model as if it were known in
advance. In testing, the model then looks fantastic — because it is secretly
peeking at the answer. In live use it is worthless, because that value does
not exist yet at the moment we need the prediction.

So the danger is not a dramatic bug. It is a silent one. The contract exists to
make the silent mistake *loud*: if any column is read from the wrong day, the
program stops with a `LeakageError` instead of producing a great-looking,
useless result.

## What the file contains

**A list of every column and its rule (the "spec").** Each column is tagged
as one of:

- **Feature** — something we are allowed to feed the model. Must come from
  **T-1 or earlier**.
- **Label** — the answer (the day-**T** balance). The *only* thing we are
  allowed to read from day **T**, and only as the answer, never as an input.
- **Metadata** — bookkeeping like IDs and dates. Never fed to the model.
- **Forbidden** — explicitly banned (see "raw day counter" below).

**Enforcement functions** that read from that list and refuse to break it.
These are what turn the list from a *description* into a *guarantee*.

## The specific traps the contract guards against

- **The sweep halves.** The FDIC balance and the same-day money-market
  (MMKT) balance are two halves of the same end-of-day movement. Using
  same-day MMKT to "predict" same-day FDIC is like using the second half of a
  receipt to predict the first half — perfect on paper, meaningless in
  practice. The contract forces every MMKT/allocation figure to come from
  **T-1**.

- **The raw day counter.** Feeding the model an absolute "this is day number
  517" value lets it memorize specific dates rather than learn real patterns.
  Banned outright. We use *relative* time instead — day of week, business day
  of the month, days until quarter-end — which carry the real seasonality
  without the memorization trap.

- **"Static" labels that aren't.** Fields like Portfolio Purpose look fixed,
  but if they are ever reassigned and stamped onto the day-**T** record,
  reading them from **T** leaks. We read them from **T-1** no matter what: if
  they really never change, we lose nothing; if they do change, the hole is
  already closed.

- **The schedule is data too.** Known-in-advance distributions are a feature
  source, so they obey the same rule: a scheduled event only counts if its
  record was *visible before the event's date*. If it only shows up on the
  day, it is secretly a leak — and the upcoming poison test checks exactly
  this.

## How this gets proven (the next piece)

The contract is the rulebook. The **poison test** (built next) is the proof.
It will deliberately corrupt the day-**T** data, then check that the features
the system produces do not change by a single byte. If poisoning day **T**
cannot move the features, then the features provably never looked at day
**T** — and the boundary holds, mechanically, not on trust.

## Open items (to confirm when data access lands)

- Confirm whether the "static" attributes are ever actually reassigned.
- Confirm a scheduled distribution's record appears *before* its date, and
  how far ahead — and whether the amount (not just the date) is known early.
- Reconcile the placeholder column names in the spec against the real tables;
  any real column missing from the spec is a gap to close.
