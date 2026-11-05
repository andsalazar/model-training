# Stage 2 findings

Stage 2 was meant to test whether Stage 1's method transfers. It found something
more important first: **Stage 1's headline verdict does not survive a second
run.** This document records what failed, why, and what is still standing.

## Summary

| Claim | Result |
|---|---|
| A — transfer to a second behaviour | **failed** — the edit is not behaviour-general |
| B — transfer to a second model | **not reached** — Stage 1 had to be re-examined first |
| C — the direction is shared or local | **answered**: layer-local, behaviour-general *in direction* only |
| Stage 1 reproducibility | **failed** — same procedure, opposite verdict |
| Measurement determinism | **confirmed** — stdev 0.0 over 7 repeats on fixed hardware |

The load-bearing finding is the reproducibility failure. It is not a bug in the
tooling; it is a property of the method, and it is now measured rather than
suspected.

## 1. Claim C — the direction is layer-local, not global

Grid: 3 behaviours × 9 layers (0, 3, 6, 9, 12, 15, 18, 21, 23), 27 directions,
378 pairwise cosines. Each behaviour is the *same question* phrased verbosely and
tersely, so the contrast is over-thinking rather than topic.

| Comparison | Mean cosine | n |
|---|---|---|
| Same behaviour, **different layers** | **+0.006** | 108 |
| Different behaviours, **same layer** | **+0.429** | 27 |

Within a behaviour, across layers:

| Behaviour | Mean | Min | Max |
|---|---|---|---|
| arithmetic | −0.000 | −0.106 | +0.170 |
| logic | +0.014 | −0.092 | +0.181 |
| factual | +0.003 | −0.089 | +0.117 |

Across behaviours, by layer:

| Layer | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 23 |
|---|---|---|---|---|---|---|---|---|---|
| Mean cosine | **+0.709** | +0.581 | +0.359 | +0.338 | +0.266 | +0.377 | +0.358 | +0.333 | +0.540 |

**There is no single "over-thinking direction" in the model.** A direction at
layer 12 is unrelated to one at layer 6 even for the same question. What Stage 1
found was *a* working direction at a layer, twice — not two measurements of one
canonical thing. Any phrasing like "we located the model's over-thinking
direction" is wrong.

The direction *is* behaviour-general though: arithmetic, logic and astronomy
share a direction at each layer (+0.429). What is being captured is a general
"keep checking" state, not an arithmetic feature. Similarity is strongest at the
ends (+0.709 at layer 0, +0.540 at 23) and weakest in the middle (+0.266 at 12),
consistent with middle layers doing task-specific work.

## 2. Claim A — the direction is general, but the edit is not

Prediction from Claim C: if behaviours share a direction at a layer, an
arithmetic-derived edit should suppress over-thinking in other behaviours.

Direction captured from **arithmetic only** at layer 12, 12 ranked columns, scale
0.5, then five probes:

| probe | role | baseline | edited | state |
|---|---|---|---|---|
| arith_target `12×11` | source | 1869 | **642** | improved |
| arith_heldout `17×23` | heldout | 676 | **278** | improved |
| logic_transfer | other behaviour | 621 | 492 | unchanged |
| fact_transfer | other behaviour | 482 | **2048** | **regressed (cap)** |
| guard_hammer | unrelated | 1204 | **2048** | **regressed (cap)** |

**The prediction failed.** The edit helped arithmetic and damaged a factual
question and an unrelated task. Sharing a direction at a layer (+0.429) is not
sharing causation: editing along that direction perturbs arithmetic in the
intended way and perturbs other behaviours in unintended ones.

So the honest scope of the method is narrower than Stage 1 implied: **a
direction-derived edit is behaviour-specific in its effect, even though the
captured direction is not behaviour-specific.**

## 3. Stage 1 does not reproduce

Claim A used layer 12, scale 0.5, top_k 12, and the same capture prompts as Stage
1's validated ranked-column sweep. 10 of the 12 columns matched. Yet the verdict
inverted:

| Run | GPU | Columns that differed | Verdict |
|---|---|---|---|
| Stage 1 ranked sweep | RTX 4000 Ada | `46`, `1575` | validated |
| Claim A / probe-set test | RTX PRO 4500 | `1176`, `2177` | harmful |

Both runs also **rejected Stage 1's own probe set**. Measured on one identical
checkpoint, the edit that Stage 1 reported as VALIDATED:

| probe | baseline | edited | state |
|---|---|---|---|
| s1_target `12×11` | 1869 | 642 | improved |
| s1_control | 727 | 272 | improved |
| s1_17x23 | 676 | 278 | improved |
| s1_13x14 | 259 | **1167** | **regressed** |
| s1_9x8 | 181 | **302** | **regressed** |
| s1_guard_capital | 232 | 201 | unchanged |

Two of Stage 1's own arithmetic held-outs regressed. On a broader probe set the
same checkpoint also pushed `fact_planet` (482 → 2048) and `guard_hammer`
(1204 → 2048) to the cap.

This is not sampling noise. Measurement on this pod is **perfectly deterministic**:
7 repeats per prompt, 5 prompts, `stdev 0.0`, spread `1.00x`, no wrong answers.
Every difference above is a real difference between checkpoints, not run-to-run
jitter.

## 4. Why it does not reproduce: the top-k cut is a plateau

The ranking has no meaningful twelfth column. Alignment cosines at layer 12:

| rank | 10 | 11 | 12 | 13 | 14 | 18 | 20 | 40 |
|---|---|---|---|---|---|---|---|---|
| cosine | .1124 | .1119 | **.1113** | .1101 | .1081 | .1034 | .1023 | .0904 |

Ties at the cut are **0.0012**, while the full range from rank 1 to rank 40 is
only **0.049**. The four columns whose membership differed between runs rank 9,
11, 13 and 18 — all inside a plateau where the score is flat to three decimals.

Consequences, in order of severity:

1. **"Top 12" is an arbitrary choice inside a tie.** A selection that changes with
   the last digit of a float is not a reproducible method. This alone explains why
   two runs on different GPUs produced different edits.
2. **The ranking provides far less discrimination than Stage 1 assumed.**
   Stage 1 section 10 concluded "the alignment ranking identifies the right
   columns". The evidence there was a contiguous window failing while a ranked set
   succeeded — but `900:1000` contained 97 *unranked* columns, so that comparison
   mostly demonstrated "editing 100 columns breaks things, editing 12 does not".
   It never isolated ranking quality from edit size.
3. **A 12-column edit is a real, large perturbation.** 12 of 3584 columns is 0.33%
   of the matrix, and the observed effects swing from −60% to +700% tokens on
   behaviour the capture never looked at. Whatever the edit is doing, it is not a
   surgical adjustment.

## 5. What is still standing

These survive, and they are worth keeping:

- **Measurement determinism.** The verifier's fixed rule is sound and its inputs
  are reproducible on fixed hardware. The problem is the method, not the harness.
- **The verifier itself.** `validated` / `harmful` with `harmful` checked first is
  exactly what caught this. Without it, the Stage 1 result would still be standing
  as fact.
- **Claim C's structure result.** Layer-local, behaviour-general — measured on 378
  pairs, far too large a split to be noise.
- **The negative results.** Contiguous window harmful; `hit_cap` ≠ looping;
  high `contrast_cosine` ≠ noise; capture past truncation.
- **Stage 0.** Unaffected — that reproduced a published edit with a fixed
  checkpoint, which is a different claim.

## 6. What must change before any further claim

1. **Select columns by a criterion that is stable across hardware**, or state the
   result as an interval rather than a point. A top-k inside a 0.0012 tie cannot
   support a reproducibility claim.
2. **Report knee and plateau, not a magic k.** The spectrum should be published
   with the edit, so a reader can see whether the selection was meaningful.
3. **Use a broad probe set by default.** Stage 1's near-pure-arithmetic set
   licensed "VALIDATED" for an edit that regressed two arithmetic held-outs on a
   different checkpoint and pushed unrelated tasks to the cap. Held-out instances
   of the *same* behaviour are not enough; the guard must include a long-form task,
   because a global verbosity change disguises itself as a targeted fix.
4. **State the operating point.** Layer, scale, column count, column-*selection
   rule*, and checkpoint hash, all of them, because the verdict depends on all of
   them.

## 7. Honest status

The Stage 1 claim was **too strong**. It said the method produces a validated
edit; in fact it produces an edit whose verdict depends on a column selection that
is not stable across hardware, and whose effect is narrower and more damaging
than the probe set revealed.

What Stage 1 actually established is narrower and still real: **a
direction-derived edit at one layer can cut a specific over-thinking behaviour by
55–90% on a fixed checkpoint measured on fixed hardware — while perturbing other
behaviours unpredictably, and not reproducing across GPUs.**

That is a useful negative-plus-positive result. It is not a method, and it should
not be published as one until the selection problem is solved.
