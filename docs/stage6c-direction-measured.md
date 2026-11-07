# Stage 6c: the direction, measured

*What the drifting-versus-stable contrast actually looks like in Ovi's residual stream,
and what it says about whether this instrument can support the hypothesis.*

## The pair

Two rollouts of one prompt and one seed, differing only in step count:

| Rollout | Steps | Colour verdict from the earlier sweeps |
|---|---|---|
| A (drifting) | 20 | drifted in 2 of 3 seeds (R² 0.59, 0.68) |
| B (stable) | 30 | stable in 3 of 3 seeds (oscillates, R² low) |

Same prompt, same seed 42, 960×960, 5s model. Captured at 5 layers (0, 2, 4, 8, 11) ×
10 step indices × both streams, patch stride 8 within each of 121 frames.

Both rollouts completed with no notes:

- 20 steps: 400 hook firings, **40 model calls** for 20 steps, 100 captures, 230.1s
- 30 steps: 600 firings, **60 model calls** for 30 steps, 100 captures, 344.7s

`model_calls == 2 × steps` in both, which is classifier-free guidance calling the model
twice per step — confirmed from behaviour a second time.

## Aligning by noise level

The two runs visit different timestep grids (20 entries versus 30). Pairing is by nearest
timestep value, and the recorded tables make it exact — every pair matched on the same
value:

```
20 steps: 999, 989, 978, 965, 952, 937, 920, 902, 882, 859, 833, 803, 768, 728, 681, 624, 555, 468, 356, 208
30 steps: 999, 992, 985, 978, 969, 961, 952, 942, 931, 920, 908, 895, 882, 867, 850, 833, 813, 792, 768, 742, 713, 681, 644, 603, 555, 499, 434, 356, 262, 146
```

The matched pairs are 999/999, 952/952, 882/882, 768/768 — no tolerance needed. This is
also a check that index pairing would have been wrong: index 4 of 20 is 952, index 4 of
30 is 969.

## The measured result

| Layer | pairs | relative delta | cosine | state norm |
|---|---|---|---|---|
| 0 | 3 | **4.68%** | 0.998802 | 4.19 |
| 2 | 3 | **2.54%** | 0.999723 | 30.61 |
| 4 | 3 | 4.02% | 0.999465 | 27.76 |
| 8 | 3 | 3.89% | 0.999338 | 27.82 |
| 11 | 3 | 4.03% | 0.999219 | 26.53 |

15 directions from 20 pairs; 5 were identical.

`relative delta = ‖stable − drifting‖ / mean(‖stable‖, ‖drifting‖)`, computed on the
mean over all 3,388 patches.

### Step 0 is exactly zero, and that is the correctness check

All five layers at step 0 report `delta_norm: 0.0` — bit-identical states, cosine
1.00000008. That is what should happen: same seed, same prompt, and step 0 of a 20-step
and a 30-step run are the *same timestep* (999.0) starting from the *same* noise. If this
had come back non-zero, either the pairing or the capture was wrong.

It is the opposite of the failure mode this project keeps hitting: here an empty result
is the correct answer, and it validates the instrument.

### What the small cosine means

A cosine of 0.9992–0.9997 means the two states point in almost the same direction. The
divergence is a 2.5–4.7% relative perturbation, not a different state. Two readings:

1. **Encouraging for the hypothesis.** The difference between a drifting and a stable
   rollout is *small* — which is what a surgical, low-parameter edit should target. TBC's
   adapter is under 0.1% of the base model; a direction this close to the original state
   is consistent with an edit of that scale being able to move the outcome.
2. **A caution about signal.** At 4% relative, the direction is a small offset on a large
   vector. Whether that offset is *causal* for the colour drift, or an incidental
   difference that accompanies any step-count change, is not established by this. A
   direction that is merely correlated with stability would produce a unit vector and
   perfectly plausible alignment scores.

Nothing here yet distinguishes those two readings. That is what the edit test is for.

### Where the divergence is largest

Layer 0 has the largest relative delta (4.68%) but the smallest state norm (4.19) — it is
an early, low-magnitude layer, so the same absolute change is a bigger fraction. Layer 2
diverges least (2.54%) despite a large state norm.

The magnitude ordering is not the same as the relative ordering, which matters for
choosing where to edit: a column's alignment score depends on the direction's *direction*,
not its size, but a damping edit's effect scales with the column's own norm.

## Projection onto the real weights — and what it says

The direction was scored against every column of the two matrices that write to the
residual stream, read directly from the 11.66B bfloat16 checkpoint (memory-mapped, no GPU
needed).

| Site | best cosine | 2nd | 10th | spread | selected |
|---|---|---|---|---|---|
| blocks.0.self_attn.o | +0.0621 | +0.0609 | +0.0512 | 0.0109 | 4 |
| blocks.0.ffn.2 | +0.0752 | +0.0731 | +0.0663 | 0.0089 | 6 |
| blocks.2.self_attn.o | +0.0607 | +0.0548 | +0.0497 | 0.0110 | 4 |
| blocks.2.ffn.2 | +0.1116 | +0.1054 | +0.0639 | 0.0477 | 4 |
| blocks.4.self_attn.o | +0.0896 | +0.0734 | +0.0540 | 0.0356 | 4 |
| blocks.4.ffn.2 | **+0.1290** | +0.1059 | +0.0741 | 0.0549 | 4 |
| blocks.8.self_attn.o | +0.0592 | +0.0528 | +0.0453 | 0.0138 | 4 |
| blocks.8.ffn.2 | +0.0827 | +0.0802 | +0.0640 | 0.0187 | 4 |
| blocks.11.self_attn.o | +0.0648 | +0.0506 | +0.0420 | 0.0229 | 4 |
| blocks.11.ffn.2 | +0.0699 | +0.0629 | +0.0549 | 0.0150 | 4 |

### The alignment is weak

The best cosine anywhere is **0.129**, and the mean best is 0.081. In a 3072-dimensional
space the expected |cosine| between two unrelated unit vectors is about
$1/\sqrt{3072} \approx 0.018$, so these scores are meaningfully above chance — but they
are nowhere near the 0.5+ that would indicate a weight column writing along the captured
direction.

### `selected = 4` is a backfill, not a finding

Every site reported 4 selected columns (6 in one case), which is exactly `min_columns`.
That means the 0.9-of-best threshold selected **zero** columns everywhere. The score
distributions are plateaus: at blocks.0.self_attn.o the top score is 0.0621 and the
second is 0.0609 — a 0.0012 gap, the same magnitude as the Stage 2 tie that made a
top-k cut arbitrary.

So the honest reading of `selected = 4` is "the threshold rule found no distinctive
column, and the backfill supplied four". Reporting that as "4 targeted weights per site"
would be exactly the error Stage 2 made.

### What this means for the hypothesis

Three interpretations, and this data does not choose between them:

1. **The direction is real but the target is wrong.** A residual-stream direction is a
   sum of contributions from *every* component in the block, not just the two that write
   to it. The components that best distinguish drifting from stable might be in the
   attention pattern itself (which keys attend to which), which no column of `o` or
   `ffn.2` represents.
2. **The direction is dominated by a shift, not a feature.** A 4% relative delta on a
   large vector is mostly a small rotation of everything, not a sparse set of features.
   A sparse feature direction would align strongly with a few columns.
3. **The measured contrast is not the mechanism.** Two rollouts at different step counts
   differ in many ways beyond colour stability; the direction may capture the gross
   difference rather than the causal one.

The observation that argues against "nothing here" is the ordering: `ffn.2` aligns
consistently more strongly than `self_attn.o` (0.075–0.129 versus 0.059–0.090), and
block 4 is the strongest site in both. That is a weak but coherent signal, and it is the
first evidence in this project about *where* in Ovi a stability-relevant direction lives.

## What an edit from this would look like — and why it is not obviously worth running

`damping_vector` would scale 4–6 columns per site by `1 - strength`. With a cosine of
0.06–0.13, the selected columns overlap the direction by at most 13%, so damping them
changes the state along the direction by roughly `strength × cosine ≈ 0.02 × 0.1 = 0.2%`
of the state norm. That is far below the 2.5–4.7% difference between the drifting and
stable rollouts — an edit of that size cannot plausibly move 20 steps toward 30-step
behaviour.

Recording that arithmetic before running the edit is the point. A null result from a
0.2%-of-norm edit would say nothing about the hypothesis, so running it would produce an
uninterpretable answer rather than a cheap one.

## Code and data

| Item | Where |
|---|---|
| Capture | `/local/captures/pair_20v30.npz` (2.8 GB), `pair_20v30.json` |
| Direction report (with vectors) | `/local/captures/direction_full.json` |
| Projection | `/local/captures/projection.json` |
| Derivation | `scripts/pod_derive_direction.py` |
| Projection | `scripts/pod_project_direction.py` |
| Summaries | `scripts/pod_direction_summary.sh`, `scripts/pod_score_distribution.sh` |

## Not yet done

- **No edit has been applied**, deliberately: see the arithmetic above.
- One seed. The colour verdict it relies on came from 3 seeds; this direction is from a
  single pair.
- The audio direction has not been projected. The capture holds audio states
  (157 tokens, same 3072 width) and the projection script accepts `--stream audio`, but
  only video was run.
- Prompt adherence is still unmeasured anywhere in the project.
