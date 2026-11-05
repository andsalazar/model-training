# Stability re-test: the selection is fixed, the method is not

## Summary

I built the stability fix the Stage 2 analysis called for, and re-tested it on the
real model. The result splits cleanly:

| question | answer |
|---|---|
| Does threshold selection pick the same columns across perturbations? | **yes** — overlap 1.000 at every ratio tested |
| Does the resulting edit behave safely? | **no** — every ratio was harmful |

**Every threshold ratio produced a harmful edit**, including the tightest
(4 columns, 0.11% of the matrix). The selection problem is solved; the method
still does not work. That is the honest conclusion, and the re-test was worth
running precisely because it distinguishes those two things.

## What was tested

`select_stable_columns` keeps every candidate column whose alignment is at least
`ratio` times the best, instead of taking the top *k*. Three ratios, one edit each,
measured against a **broad** probe set — the one change that would have caught
Stage 1's error. 9 probes: 4 arithmetic variants, logic, factual, and three
unrelated tasks including two long-form ones.

## Result: selection is stable

| ratio | columns selected | stability overlap |
|---|---|---|
| 0.95 | 4 | **1.000** |
| 0.90 | 4 | **1.000** |
| 0.80 | 12 | **1.000** |

The threshold rule is doing exactly what it was designed to do. The tight ratios
admit only columns `102, 987, 431, 914` — the four strongest, which are separated
from the rest by a real gap rather than a tie. Those four are also the ones that
appeared in *both* Stage 1's and Stage 2's rankings, which is the stability
property the fix was for.

Note also what the ratio reveals: at 0.80 the selection jumps to 12 columns and
picks up `1176, 2107, 2177, 178` — the plateau. The rule makes the plateau
*visible* as a jump in `k`, where `top-k=12` would have silently taken those four
and said nothing. That visibility is worth keeping regardless of the outcome.

## Result: the edit is harmful at every ratio

Baseline → edited token counts. `H` marks a probe the rule calls harmful.

**ratio 0.95 — 4 columns**

| probe | base | edited | state |
|---|---|---|---|
| arith_target `12×11` | 1869 | **323** | improved |
| arith_17x23 | 676 | **2048** | **H regressed (cap)** |
| arith_347x28 | 2048 | 1798 | unchanged |
| arith_9x8 | 181 | 181 | unchanged |
| logic | 621 | **298** | improved |
| fact_planet | 482 | **354** | improved |
| guard_capital | 232 | 231 | unchanged |
| guard_hammer | 1204 | **2048** | **H regressed (cap)** |
| guard_poem | 1971 | **2048** | **H regressed (cap)** |

**ratio 0.90 — 4 columns**: identical to 0.95 (same four columns).

**ratio 0.80 — 12 columns**

| probe | base | edited | state |
|---|---|---|---|
| arith_target `12×11` | 1869 | **791** | improved |
| arith_17x23 | 676 | **278** | improved |
| arith_347x28 | 2048 | **1375** | improved |
| arith_9x8 | 181 | 220 | unchanged |
| logic | 621 | **436** | improved |
| fact_planet | 482 | **738** | **H regressed** |
| guard_capital | 232 | 201 | unchanged |
| guard_hammer | 1204 | **2048** | **H regressed (cap)** |
| guard_poem | 1971 | **1474** | improved |

Tally:

| ratio | columns | improved | harmful |
|---|---|---|---|
| 0.95 | 4 | 3 | **3** |
| 0.90 | 4 | 3 | **3** |
| 0.80 | 12 | 5 | **2** |

## What this means

**The edit is not behaviour-selective, at any size.** A 4-column edit — 0.11% of
the matrix — improves the target by 83% while pushing two unrelated long-form
tasks to the token cap. Editing 12 columns is not obviously worse and is better on
some probes, which rules out "the edit was just too large" as the explanation.

**Column count is not the controlling variable.** If damage scaled with edit size,
4 columns should be safer than 12. It is not: 4 columns breaks `arith_17x23`,
`guard_hammer` and `guard_poem`; 12 columns breaks `fact_planet` and
`guard_hammer`. The damage is *different*, not less. That points at the direction
itself rather than at how many columns carry it.

**The direction is the problem, not the selection.** Stage 2 already showed the
captured direction is behaviour-general (+0.429 across behaviours at a layer)
while the edit is behaviour-specific in effect. Putting those together: we are
subtracting a vector that encodes a *shared* "keep checking" state, and that state
is used by arithmetic, logic, and long-form writing alike. Suppressing it makes
short answers shorter and long answers fall apart. The `guard_poem` result is the
clearest evidence — a 2-line poem is a task where "keep checking" is *useful*, and
removing it produces a 2048-token failure.

## Honest status

Two things are now established and one is refuted:

- **Established:** the threshold selection is reproducible (overlap 1.000), and it
  makes the ranking's plateau visible via `k`.
- **Established:** a directional edit derived this way is harmful on a broad probe
  set at every size tested (4, 12 columns) and every ratio (0.95, 0.90, 0.80).
- **Refuted:** that Stage 1's "validated" result indicated a working method. It
  was a narrow probe set meeting an unstable selection. Both are now fixed, and
  the result is negative.

So the sequence of this project is:

| stage | claim | outcome |
|---|---|---|
| 0 | the paper's result reproduces | **holds** |
| 1 | an edit can be derived from measurement | **holds** — it can be derived, and it works on a narrow probe set |
| 2 | it reproduces across hardware | **fails** — selection was unstable |
| 3 | fixing the selection fixes the method | **fails** — the edit is harmful at every size |

The method needs a *behaviour-selective* mechanism, not a better column choice.
The obvious candidate is the one the original brief pointed at: a trained adapter
(LoRA) whose capacity is spent learning which activations to suppress, rather than
a single fixed direction subtracted from a handful of columns.

## Cost note

The re-test ran three full edit-and-measure cycles (27 generations across 3
checkpoints) plus 2 captures. Roughly 40 minutes of pod time. Everything is
scripted in `scripts/pod_stability.py`, so re-running on other hardware is cheap.
