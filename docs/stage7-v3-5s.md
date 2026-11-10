# Stage 7 v3 at 5 s: broader evaluation, on-policy training, and a noise floor

*The plan after the WSP stage: train on the adapter's own outputs (parts 1 and 3 of the
"move it forward" list), try an adapter at blocks 0–3 (the audio block 3 finding), and
broaden the evaluation to 6 unseen prompts with a prompt-adherence score. The most
important result is none of those: it is that a single generated clip is too noisy a
measurement for the differences this project has been reporting.*

> **Recheck, 2026-11-09.** The paired statistics were recomputed from the committed JSON
> with an exact test, a symmetric distance and a per-metric breakdown; see
> [Statistical recheck](#statistical-recheck). Two readings changed: "the adapter is
> better than plain 12 steps" is supported in direction but not settled on unseen
> prompts, and most of that gain is the sharpness term. "30 steps is better than every
> adapter" is unchanged.

Driver: `scripts/pod_adapter_run.py` (`onpolicy` and multi-adapter `eval` stages),
`scripts/pod_v3_5s.sh`; analysis: `scripts/analyse_v3.py` (per-variant means) and
`scripts/analyse_v3_paired.py` (paired statistics); adherence:
`scripts/pod_adherence.py`. Results: `artifacts/v3_5s/`. Hardware: one H100 SXM
(no A100 stock), $29.09 in total.

## What was run

| | |
|---|---|
| Evaluation | 9 clips: river seeds 100/200/300 (a training prompt, held-out seeds) + 6 unseen prompts (violin, kitchen, storm, skater, campfire, piano), seed 100 |
| Reference | 48-step UniPC teacher, regenerated on the H100 |
| Variants | UniPC-30, UniPC-12, and four adapters at 12 Euler steps: **v2** (the Stage 7 v2 5 s adapter), **v2fix** (v2 with the audio adapter off for steps 0–1), **onpol** (v2 fine-tuned on its own rollouts, 2 epochs), **onpol03** (blocks 0–3, block 3 new, 1 epoch on onpol's epoch-0 data) |
| On-policy targets | for each state the adapter actually reaches, the frozen base model takes 4 guided Euler substeps on the 48-step grid; the target is the velocity covering that in one step (`Trainer.collect`) |
| Adherence | CLIP ViT-L/14 (frames vs the prompt's visual part) and CLAP (audio vs the audio part), as the margin over the other 6 prompts |

## The noise floor

The same prompt, seed and sampler settings produce **different clips on an H100 than on
an A100**. Two A100 pods had reproduced each other exactly
([stage7-adapter-step0.md](stage7-adapter-step0.md)), so the difference tracks the GPU
type. **The cause was not tested.** One candidate is that the two GPUs' bf16 kernels
differ slightly and sampling amplifies the difference. Another is that the starting noise
itself differs: it is drawn with a generator on the GPU (`Trainer.initial_noise` in
`scripts/pod_adapter_run.py` mirrors Ovi's draw). Comparing the starting-noise tensors
across GPU types would separate the two, and has not been done. Distance (the Stage 7
four-metric measure) of each H100 clip from the A100 clip of the *same* variant:

| | river 100 | river 200 | river 300 | violin | mean |
|---|---|---|---|---|---|
| teacher-48 | 0.094 | 0.476 | 0.093 | 0.196 | **0.215** |
| UniPC-12 | 0.128 | 0.081 | 0.385 | 0.201 | 0.199 |
| UniPC-30 | 0.157 | 0.175 | 0.325 | 0.192 | 0.212 |

**Re-drawing a clip moves its distance by about 0.2.** That is as large as most of the
differences between variants reported in Stages 7 v2 and the WSP step, which rested on 3–4
single clips. Those conclusions need re-reading (see "Corrections" below).

## Results (all on the H100, same clips for every variant)

Same-pod wall clock: 12 steps **57.5 s**, 30 steps **124.5 s** (2.17×).

| variant | video distance, river | unseen | all 9 | flash, worst clip | CLIP margin | CLAP margin | audio brightness error |
|---|---|---|---|---|---|---|---|
| teacher-48 | 0 | 0 | 0 | — | **19.45** | 28.84 | 0 |
| UniPC-30 | 0.412 | **0.235** | **0.294** | 1.3 | 19.44 | 29.99 | **2.6%** |
| UniPC-12 | 0.622 | 0.557 | 0.578 | 1.1 | 16.40 | 29.58 | 4.4% |
| v2 | 0.401 | 0.439 | 0.426 | 1.2 | 18.74 | 29.97 | 5.0% |
| v2fix | 0.308 | 0.569 | 0.482 | 1.3 | 18.46 | 29.41 | 6.0% |
| onpol | 0.351 | 0.509 | 0.457 | 1.1 | 18.55 | 28.62 | 5.1% |
| onpol03 | **0.223** | 0.542 | 0.436 | 1.4 | 18.57 | 30.26 | 5.1% |

Paired differences in video distance across clips (negative = first is closer to the
teacher), mean and 95% bootstrap interval:

| comparison | all 9 | unseen 6 |
|---|---|---|
| v2 − UniPC-12 | **−0.152 [−0.306, −0.023]**, 6/9 better | −0.117 [−0.349, +0.033] |
| v2 − UniPC-30 | **+0.132 [+0.020, +0.260]** | **+0.204 [+0.063, +0.354]**, 0/6 better |
| onpol − v2 | +0.030 [−0.063, +0.119] | **+0.070 [+0.002, +0.147]** |
| onpol03 − v2 | +0.009 [−0.144, +0.139] | **+0.103 [+0.010, +0.208]** |
| v2fix − v2 | +0.056 [−0.071, +0.186] | **+0.130 [+0.009, +0.278]** |

What survives the noise:

1. **The adapter is better than plain 12 steps in direction** (v2, 6 of 9 clips, bootstrap
   interval excludes 0), at the same speed. The recheck below qualifies this: an exact
   test gives p = 0.074, only 3 of the 6 unseen prompts improve, and most of the gain is
   one metric.
2. **30 steps is better than every adapter on unseen prompts**, on all 6 clips.
3. **On-policy training did not generalise.** It helps on the training prompt (river:
   0.401 → 0.351, and 0.223 with block 3), within the noise floor, and is worse than v2
   on unseen prompts. Its own validation loss fell 13% (0.0330 → 0.0287), on prompts it
   also never trained on, while rollout quality on unseen prompts got worse: the
   one-step loss is again not a reliable guide to rollout quality.
4. **No flash in any of the 54 clips** (worst last-8-frame brightness jump 1.4; the A100
   v2 violin clip had 17.8). The flash was a property of particular A100 draws, not a
   reliable adapter defect, and the fix for it only costs quality here.
5. **Prompt adherence is the clearest positive.** 12 steps loses CLIP margin (16.4 vs
   19.45 for the teacher and 19.44 for 30 steps); every adapter recovers most of it
   (18.5–18.7). v2 beats UniPC-12 on 8 of 9 clips (mean +2.4; +1.4 without the storm
   clip, where UniPC-12 collapsed to 4.4). Audio adherence (CLAP) does not change with
   any variant: 12 steps does not lose it, and no adapter adds to it.
6. **Audio brightness is still not fixed.** Every adapter is slightly further from the
   teacher's spectral centroid than plain 12 steps (5.0–6.0% vs 4.4%). Audio loudness
   (RMS) is too noisy per clip to rank: one kitchen clip gives UniPC-30 a 3.3× error.

## Statistical recheck

The intervals above were computed ad hoc; no committed script produced them.
`scripts/analyse_v3_paired.py` now reproduces them from `artifacts/v3_5s/eval.json` and
`adherence.json` (output: `artifacts/v3_5s/paired.json`, pinned by
`tests/test_analyse_v3_paired.py`) and adds four checks. Bootstrap bounds move in the
third decimal with the random seed.

### 1. An exact test next to the bootstrap

A percentile bootstrap from 6 to 9 clips gives intervals that are too narrow to lean on.
The exact sign-flip test asks how often the observed mean difference would arise if each
clip's difference were equally likely to have either sign. Video distance, first minus
second (negative = first is closer to the teacher):

| comparison | all 9 | unseen 6 |
|---|---|---|
| v2 − UniPC-12 | −0.152, 6/9 closer, p = 0.074 | −0.117, 3/6 closer, p = 0.44 |
| v2 − UniPC-30 | +0.132, 2/9 closer, p = 0.074 | +0.204, 0/6 closer, p = 0.031 |
| UniPC-30 − UniPC-12 | −0.284, 9/9 closer, p = 0.004 | −0.321, 6/6 closer, p = 0.031 |

With 6 clips the smallest possible p is 0.031, so "0 of 6" is the strongest result six
clips can give. The three river clips share one prompt, so the 9 clips are not 9
independent samples.

### 2. A symmetric distance

The Stage 7 measure, mean `|a − b| / b`, is asymmetric: a metric at twice the teacher's
value scores 1.0 and one at half scores 0.5. A variant that undershoots is therefore
favoured over one that overshoots by the same ratio. With mean `|log(a / b)|`, which
treats both alike:

| comparison | all 9 | unseen 6 |
|---|---|---|
| v2 − UniPC-12 | −0.131, 7/9 closer, p = 0.016 | −0.136, 4/6 closer, p = 0.125 |
| v2 − UniPC-30 | +0.135, 1/9 closer, p = 0.016 | +0.201, 0/6 closer, p = 0.031 |
| UniPC-30 − UniPC-12 | −0.266, 9/9 closer, p = 0.004 | −0.337, 6/6 closer, p = 0.031 |
| onpol − v2 | +0.092, 4/9 closer, p = 0.19 | +0.144, 2/6 closer, p = 0.125 |
| v2fix − v2 | +0.051, 2/9 closer, p = 0.26 | +0.111, 0/6 closer, p = 0.031 |

The ordering does not change. The adapter's gain over plain 12 steps is slightly firmer
on the symmetric measure, and still unsettled on the unseen prompts alone.

### 3. Which metric the gain comes from

Mean absolute relative error against the teacher, per metric, over the 9 clips:

| | sharpness | flicker | motion coherence | colour stability | sharpness, signed | clips softer than the teacher |
|---|---|---|---|---|---|---|
| UniPC-30 | 0.392 | 0.200 | 0.090 | 0.496 | +0.051 | 5/9 |
| UniPC-12 | 0.892 | 0.573 | 0.224 | 0.625 | +0.311 | 4/9 |
| v2 | 0.404 | 0.475 | 0.140 | 0.686 | −0.235 | 7/9 |

- **Most of v2's gain is the sharpness term** (0.892 to 0.404). Averaged over the other
  three metrics the two are close: UniPC-12 0.474, v2 0.434.
- **The adapter over-corrects.** Plain 12 steps is on average 31% sharper than the
  teacher; v2 is on average 23.5% softer, and softer on 7 of 9 clips. The blur reported
  at 10 s has a milder counterpart at 5 s. This is what regressing onto a teacher's
  velocity with a squared error would be expected to do.
- **Plain 12 steps is not uniformly over-sharp.** It is softer than the teacher on the
  four low-texture unseen prompts (campfire, kitchen, piano, storm) and sharper on the
  river and violin clips, so "few steps leave excess texture" depends on the prompt.

### 4. Prompt adherence: margin against raw score

| v2 − UniPC-12, all 9 | mean | clips higher | p |
|---|---|---|---|
| CLIP margin (matched − mean of mismatched) | +2.34 | 8/9 | 0.047 |
| raw CLIP score | +1.06 | 6/9 | 0.23 |

The margin result holds; the raw score does not separate the two. Part of the margin
gain is lower similarity to the *other* prompts rather than higher similarity to the
right one. The same is true of UniPC-30 against UniPC-12 (margin +3.04, p = 0.047; raw
+1.37, p = 0.15), so this is a property of the measure at 9 clips, not of the adapter.

### What the recheck leaves standing

- **Firm:** 30 steps is closer to the teacher than 12 steps (9 of 9 clips) and than the
  adapter on unseen prompts (6 of 6), on both distances.
- **Supported in direction, not settled:** the adapter improves on plain 12 steps. It
  holds on the training prompt and on the storm clip, where plain 12 steps collapsed;
  across the other five unseen prompts the mean difference is near zero on the original
  distance.
- **Not measured:** UniPC at 16 or 20 steps on the same pod. v2's distance (0.426) sits
  between UniPC-12 (0.578) and UniPC-30 (0.294), so whether the adapter beats simply
  running a few more steps is unknown. Base Euler-12 and adapter + UniPC were not in
  this evaluation either.

## Corrections to earlier Stage 7 docs

| earlier claim | status after v3 |
|---|---|
| v2: "12 steps + adapter is closer to the teacher than 30 steps on the river prompt" (0.193 vs 0.248, 3 clips) | **not established**: within the 0.2 noise floor; on the H100 the same comparison gives 0.401 vs 0.412 |
| v2 / WSP: "the adapter introduces an end-of-clip flash" | **not reproducible**: 0 of 54 clips on the H100; a feature of particular A100 samples |
| WSP: "holding the audio adapter off for 2 steps removes the flash" | the ablation was correct for those A100 clips, but the defect itself is not robust, so the fix is not needed and slightly hurts unseen prompts |
| WSP: audio block 3 "moves away from the teacher in every clip" | one A100 read on 4 clips; untested here; the block 0–3 adapter built on it did not help on unseen prompts |
| Pilot / v2: "the adapter halves the 12-step gap" | **supported in direction**: v2 beats UniPC-12 on 6 of 9 clips, by less (0.426 vs 0.578) |
| 10 s: blur | not re-tested; it was a large effect (sharpness 6.7 vs 14.1) and probably exceeds the noise, but it is one hardware's draws |

## What would make the next result trustworthy

- **Several samples per prompt.** With a noise floor of ~0.2 per clip, comparing two
  variants needs on the order of 3 seeds × 10 prompts to resolve differences of ~0.1.
- **Many more training prompts.** 10 training prompts (16 trajectories) let on-policy
  fine-tuning overfit; distillation methods typically use thousands.
- **A metric with less variance.** The four-metric distance to a single teacher sample is
  dominated by which sample was drawn; a distribution-level measure (e.g. FVD-style
  feature statistics over many clips) would compare variants rather than draws.
