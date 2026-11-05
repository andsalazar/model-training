# Stability-based column selection

*The fix for the selection problem that made Stage 1 irreproducible.*

## The problem being fixed

Stage 1 chose the **top 12 columns** by alignment score. Stage 2 showed that cut
was arbitrary: at layer 12 the scores tie to within **0.0012** at rank 12, while
ranks 1–40 span only **0.049**. The four columns that differed between two runs
ranked 9, 11, 13 and 18 — all inside a flat region. Different hardware, slightly
different activations, different columns, opposite verdict.

A rank is the wrong selection primitive, because a rank is a statement about *how
many other columns happen to be slightly above*, which is exactly the quantity a
tie makes meaningless.

## The rule

Select by **score threshold**, not by count. Keep every candidate column whose
alignment is at least `ratio` times the best alignment:

```
threshold = max(score) * ratio
selected  = [col for col, score in spectrum if score >= threshold]
```

A column's membership now depends on its own score relative to the maximum, not on
how many columns sit just above it. Two runs with slightly different scores will
mostly agree, because a column only enters or leaves if it crosses a fixed
fraction of a smoothly-varying maximum.

This also makes the selection **self-reporting**. When the ranking is a plateau,
the threshold admits many columns and the output says so — "40 columns within 99%
of the top score" is an honest description of a tie. `top-k` would report "12"
either way and hide the problem.

## Measured effect

The right perturbation to model is the **direction**, not the weights. Stage 2's
runs differed because the captured direction differed — it comes from a
transformers forward pass whose activations shift with hardware, while the weight
tensor is fixed. Jittering the weights at any realistic scale changes nothing;
jittering the direction reproduces the observed failure.

Overlap between the selection made on the real direction and on a perturbed one
(Jaccard; 1.0 means identical):

| direction perturbation | rank cut (k=12) | threshold cut (ratio 0.99) |
|---|---|---|
| 1e-4 | 1.000 | 1.000 |
| **1e-3** | **0.846** | **1.000** |
| 5e-3 | 0.714 | 1.000 |
| 1e-2 | 0.714 | 0.750 |

At the perturbation size that broke Stage 2, the rank cut loses 15% of its
columns to noise while the threshold cut loses none.

## API

```python
from wsp.activations import select_stable_columns, alignment_spectrum, selection_overlap

# Look at the shape before choosing a cut.
for col, score in alignment_spectrum(weight, direction, depth=40):
    print(col, score)

sel = select_stable_columns(weight, direction, ratio=0.99, candidate_depth=200)
print(sel.describe())      # "20 columns within 99% of the top score (0.0992) ..."
print(sel.k, sel.columns)

# Stability is quotable, not just claimed.
overlap = selection_overlap(sel_run_a, sel_run_b)
```

`min_columns` backfills from the spectrum when a threshold is too strict to yield
a usable edit, so a degenerate distribution cannot produce a one-column edit.

## Why this is the right fix and not just a different heuristic

Three properties matter, and the rank cut has none of them:

1. **Invariance to population.** Adding or removing near-tied columns does not
   change which columns are selected, only how many.
2. **Threshold is meaningful.** "Within 99% of the best" is a statement about the
   score distribution. "Top 12" is a statement about nothing when 40 columns are
   tied.
3. **Failure is visible.** A plateau produces a large selection and a large `k`,
   which is a signal to the operator. A rank cut returns the requested number
   regardless of whether the ranking supported it.

## What this does *not* fix

The threshold makes the *selection* reproducible. It does not make the *effect*
reproducible. Stage 2's edit also pushed unrelated behaviour to the token cap,
which is a property of the perturbation size (12–20 columns of 3584, i.e. 0.3–0.6%
of the matrix) rather than of which columns were chosen.

So the re-test has two questions, not one:

- does the threshold selection pick the same columns across runs? (it should)
- does the resulting edit behave the same way, across a *broad* probe set? (open)

If selection is stable and the effect still is not, then the method's problem is
not the column choice and the honest conclusion is that a directional edit of this
size is too blunt for the claim being made.

## Tests

`tests/test_stability.py` pins the property, and deliberately includes a test that
**the fixture fails before the fix is applied**:

- `test_direction_perturbation_reproduces_the_real_instability` — asserts the rank
  cut *does* lose columns under 1e-3 direction jitter. Without this, the stability
  tests would be measuring nothing.
- `test_threshold_selection_beats_rank_cut_under_direction_perturbation` — the
  comparison the fix rests on, at the observed perturbation size.

14 tests, all property-based on synthetic rankings with a real plateau.
