# Stage 7 v4: the deciding experiment. Does the adapter beat running a few more steps?

*Stage 7 v3 left the adapter's value open. 12 steps with the v2 adapter sits between plain
12 steps and 30 steps, but UniPC at 16 or 20 steps was never run on the same pod, so
nothing said whether the adapter does more than four or eight extra steps would. This
experiment runs that comparison, with enough prompts and seeds to test it.*

**Result: the adapter does not beat running more steps, and on ten unseen prompts it does
not beat plain 12 steps either.** 12 steps with the adapter cannot be told apart from
plain UniPC at 12 or 16 steps, and plain UniPC at 20 steps is closer to the reference
(p = 0.020) for 29 more seconds per clip. The adapter's one measurable effect is against
the Euler sampler it has to run with: it makes up most of what switching from UniPC to
Euler loses. For generating clips, plain UniPC at 20 or 30 steps is the better choice and
the adapter is not worth using.

Everything under "Design", "Measures" and "How the result will be read" was committed and
pushed before the pod was created (commit `8bcb23f`). Results are underneath.

Driver: `scripts/pod_adapter_run.py` (sets `decide_s100/200/300`), pipeline
`scripts/pod_decide.sh`, analysis `scripts/analyse_decide.py`. Evidence:
`artifacts/v4_decide/`.

## Design

| | |
|---|---|
| Prompts | 10, none used to train or select any adapter: the six unseen prompts of v3 (violin, kitchen, storm, skater, campfire, piano) and four new ones (harbor, horse, typist, fountain) |
| Seeds | 100, 200, 300 for every prompt: 30 prompt-seed pairs |
| Reference | 48-step UniPC teacher for each pair, generated on the same pod |
| Base model | UniPC at **12, 16, 20 and 30** steps; Euler at 12 steps |
| Adapter | the v2 5 s adapter (`artifacts/adapter5s_v2/adapter_best.pt`, blocks 0–2, rank 8, 313k parameters) with 12 Euler steps |
| Model | Ovi 960x960, 5 s, text-to-video, the config's sampler settings |
| Hardware | one pod, one GPU type, for every clip |

210 clips. By the v3 timings on an H100 (about 3.7 s per step plus 13 s), one pair costs
about 655 s across all seven variants, 5.5 hours for the 30.

The run goes one seed at a time (teacher, variants, prompt adherence), so if it is cut
short it still holds all ten prompts at the seeds it finished.

**What is new against v3.** UniPC at 16 and 20 steps, and base Euler at 12, on the same
pod as everything else; ten unseen prompts instead of six; three seeds per prompt instead
of one. The on-policy and "fix" adapters are not included: v3 found them no better than
v2 on unseen prompts.

## Measures

The **prompt** is the unit: a prompt's three seeds are averaged before any test, giving
ten paired differences per comparison. Tests are exact sign-flip tests over the ten
prompts (1,024 sign patterns; the smallest possible p is 0.002), with a bootstrap
interval alongside.

- **Primary: symmetric distance to the teacher.** Per clip, the mean of `|log(a / b)|`
  over sharpness, flicker, motion coherence and colour stability, against the teacher
  clip of the same prompt and seed. The v3 recheck showed the older `|a − b| / b` measure
  favours a variant that undershoots; it is reported too, as `relative`.
- **Pooled distance.** Each metric is averaged over a prompt's three seeds first, for the
  variant and for the teacher, and then compared. This does not rest on one teacher clip
  per comparison, which v3 found moves by about 0.2 when merely re-drawn.
- **Prompt adherence.** CLIP margin (video against the prompt's visual part, minus the
  mean over the other nine prompts), raw CLIP, and CLAP margin for audio.
- **Audio.** Loudness (RMS) and brightness (spectral centroid) from each clip's MP4 audio
  track, as `|log(variant / teacher)|`.
- **Per-metric breakdown**, signed, to see whether the adapter is still softer than the
  teacher.
- **Wall-clock time** per variant, from this pod.
- **Placement.** The adapter's mean distance, located on the base model's
  distance-against-steps curve (12, 16, 20, 30) by linear interpolation: "12 steps with
  the adapter is worth about N plain steps".

Also recorded, not analysed here: a hash of the starting noise per seed
(`scripts/pod_noise_hash.py`), so that a later run on another GPU type can test whether
the cross-GPU difference seen in v3 starts in the noise.

## How the result will be read

Decided before the run, on the primary measure at prompt level:

| outcome | condition | what it means |
|---|---|---|
| **The adapter beats more steps** | adapter closer to the teacher than UniPC-16, p ≤ 0.05 | 12 steps with the adapter gives quality that plain sampling needs more than 16 steps for: at least 1.24× faster at equal quality (72 s against 58 s). If it also beats UniPC-20, at least 1.5×. Worth publishing as a fast mode. |
| **More steps beat the adapter** | UniPC-16 closer than the adapter, p ≤ 0.05 | four extra steps do more than the adapter does. The adapter is not a useful speed-up; publish the finding, not the weights as a recommendation. |
| **Not separated from UniPC-16** | neither of the above | the adapter is worth somewhere around 12 to 20 plain steps and this design cannot say where. Its gain at equal quality is then at most a few seconds per clip. |

Two checks decide whether the table can be used at all:

- **Does the adapter help at equal cost?** Adapter against UniPC-12 and against base
  Euler-12. v3 left this "supported in direction, not settled" on unseen prompts.
- **Can the design resolve the base curve?** UniPC-16 against UniPC-12, and UniPC-20
  against UniPC-16. If four extra plain steps do not show up as a difference at ten
  prompts, the measure is too coarse to place an adapter between them, and the result is
  reported as that.

Five adapter comparisons are run and no correction for multiple tests is applied. The two
that the outcome rests on are adapter against UniPC-16 and adapter against UniPC-12.

## Limits known in advance

- One GPU type. v3 found clips differ between an A100 and an H100; every clip here is
  from one pod, so comparisons are internal, but absolute numbers will not match v3's.
- The four video metrics are simple signal measures. They track softness, flicker and
  colour drift; they are not a perceptual score. A subset of clips is copied back for
  viewing.
- 5 s only. The 10 s adapter, which v2 found visibly softer, is not re-tested.
- One adapter, one training run. A better-trained adapter could move the result; this
  tests the one that exists.

## Results

Run on 2026-11-09/10 on one H100 SXM (driver 595.91.07, torch 2.6.0, Ovi at commit
`5b69b25`). All 210 clips were generated, none failed. The first clip (harbor, seed 100)
came from the smoke test, which runs the same script with the same settings. The pod ran
for 6 h 52 min at $3.99 an hour, about $27.40.

Reproduce every number below with
`python scripts/analyse_decide.py artifacts/v4_decide`; the quoted ones are pinned in
`tests/test_analyse_decide.py`.

### Time and distance to the 48-step reference

| variant | seconds per clip | distance, symmetric (primary) | distance, relative | distance, pooled |
|---|---|---|---|---|
| UniPC 30 | 122.3 | 0.238 | 0.238 | 0.174 |
| UniPC 20 | 85.7 | 0.378 | 0.339 | 0.338 |
| UniPC 16 | 71.1 | 0.449 | 0.441 | 0.373 |
| UniPC 12 | 56.5 | 0.471 | 0.458 | 0.381 |
| **adapter + Euler 12** | 56.5 | 0.478 | 0.406 | 0.457 |
| Euler 12 | 56.4 | 0.570 | 0.494 | 0.513 |

The 48-step reference itself takes 192.8 s. The adapter adds no measurable time.

### The comparisons, by prompt (symmetric distance; negative = first is closer)

| comparison | mean difference | 95% interval | prompts where first is closer | p |
|---|---|---|---|---|
| adapter − UniPC 12 | +0.006 | [−0.078, +0.089] | 5 of 10 | 0.89 |
| adapter − Euler 12 | −0.092 | [−0.174, −0.008] | 7 of 10 | 0.076 |
| adapter − UniPC 16 | +0.028 | [−0.064, +0.111] | 2 of 10 | 0.56 |
| adapter − UniPC 20 | +0.100 | [+0.036, +0.159] | 1 of 10 | 0.020 |
| adapter − UniPC 30 | +0.240 | [+0.170, +0.311] | 0 of 10 | 0.002 |
| UniPC 16 − UniPC 12 | −0.022 | [−0.061, +0.014] | 6 of 10 | 0.31 |
| UniPC 20 − UniPC 16 | −0.072 | [−0.130, −0.023] | 7 of 10 | 0.023 |
| UniPC 30 − UniPC 20 | −0.140 | [−0.197, −0.087] | 10 of 10 | 0.002 |

### Read against the rule fixed in advance

- **Outcome: "not separated from UniPC-16".** The adapter is neither closer nor further
  than UniPC-16 at p ≤ 0.05 (+0.028, p = 0.56).
- **Does the adapter help at equal cost? No.** Against UniPC-12 the difference is +0.006
  with the ten prompts split five to five (p = 0.89). Against base Euler-12, the sampler
  it actually runs with, it is closer on 7 of 10 prompts (−0.092, p = 0.076; p = 0.020 on
  the relative measure). So the adapter repairs most of what Euler loses to UniPC at 12
  steps, and ends level with UniPC-12, which needs no adapter.
- **Can the design resolve the base curve? From 16 steps up, yes; between 12 and 16,
  no.** UniPC-20 is closer than UniPC-16 (p = 0.023) and UniPC-30 closer than UniPC-20 on
  all ten prompts, but UniPC-16 is not separable from UniPC-12 (p = 0.31). By the check
  written in advance, the measure is too coarse to place anything between 12 and 16
  steps. It is fine enough to say the adapter is behind 20 steps.

On placement: the adapter's distance is "no better than 12 steps" on the primary and
pooled measures, and "17.4 steps" on the older relative measure. The relative measure is
the one the v3 recheck showed to favour undershoot, and the adapter undershoots (below).
On the pooled measure, which does not depend on single reference clips, the adapter is
*further* from the reference than UniPC-12 (+0.076, closer on 3 of 10 prompts, p = 0.041).

### Why the measures disagree about the adapter: it is softer

Signed error against the reference, over the 30 clips:

| | sharpness: mean signed error | clips softer than the reference | sharpness: mean absolute error | flicker: mean absolute error |
|---|---|---|---|---|
| UniPC 12 | +0.8% | 18 of 30 | 0.620 | 0.408 |
| UniPC 30 | −1.6% | 21 of 30 | 0.301 | 0.212 |
| adapter + Euler 12 | **−19.9%** | **24 of 30** | 0.391 | 0.503 |

Plain 12 steps misses the reference's sharpness in both directions, by a lot, with no
bias. The adapter misses by less and almost always on the soft side, and it changes less
from frame to frame than the reference (flicker −13.6% on average, lower on 23 of 30
clips). A measure that forgives undershoot scores that as a gain; a symmetric one does
not. This is the same softening seen at 10 s in v2 and at 5 s in the v3 recheck.

### Prompt adherence and audio

| | CLIP margin (video) | against the reference, p | CLAP margin (audio) | audio loudness error | audio brightness error |
|---|---|---|---|---|---|
| reference, 48 steps | 19.30 | | 26.78 | 0 | 0 |
| UniPC 30 | 19.21 | 0.58 | 26.70 | 0.190 | 0.027 |
| UniPC 20 | 19.38 | 0.82 | 27.03 | 0.256 | 0.053 |
| UniPC 16 | 19.03 | 0.55 | 31.99 | 0.223 | 0.045 |
| UniPC 12 | 17.96 | 0.008 | 27.75 | 0.254 | 0.041 |
| adapter + Euler 12 | 18.59 | 0.27 | 27.98 | 0.366 | 0.066 |
| Euler 12 | 17.87 | 0.049 | 27.07 | 0.425 | 0.071 |

- **Plain 12 steps loses video prompt adherence** (−1.34 against the reference, p =
  0.008), as v3 found. **Sixteen plain steps recover it** (+1.07 over UniPC-12, p =
  0.021; not different from the reference).
- **The adapter's adherence gain is not confirmed.** +0.62 over UniPC-12, on 6 of 10
  prompts, p = 0.35. v3 reported +2.34 (p = 0.047) on 9 clips.
- **Audio.** The adapter's loudness error is larger than UniPC-12's (0.366 against 0.254,
  smaller on 3 of 10 prompts, p = 0.10) and larger than UniPC-30's (p = 0.018). It is a
  little better than base Euler-12 (0.425). Audio prompt adherence does not separate any
  variant from the reference; UniPC-16's high CLAP mean is not significant (p = 0.09).

### Corrections to earlier Stage 7 documents

| earlier claim | status after v4 |
|---|---|
| v3 recheck: the adapter is better than plain 12 steps "in direction, not settled" | **not supported.** On ten unseen prompts at three seeds: +0.006, five prompts each way. |
| v3: the adapter "recovers most of the prompt adherence 12 steps loses" | **not confirmed** (+0.62, p = 0.35). Running 16 plain steps does recover it. |
| pilot, v2: the adapter "halves the 12-step gap" | held for base **Euler**-12, the sampler the adapter uses, in direction (p = 0.076). Not for UniPC-12, the sampler anyone would otherwise use. |
| v3: 30 steps beats the adapter on every unseen prompt | **confirmed**, 10 of 10. |
| v3 recheck: the adapter over-corrects sharpness | **confirmed**: softer than the reference on 24 of 30 clips. |

### An observation outside the plan: old prompts against new ones

Not planned in advance, and a split of ten prompts into six and four, so it is a lead
and nothing firmer. On v3's six prompts the adapter is closer than UniPC-12 on five
(mean −0.074), and at seed 100 alone the mean is −0.136, the same figure the v3 recheck
gave for those prompts. On the four prompts added here (harbor, horse, typist, fountain)
it is further than UniPC-12 on all four. None of the ten was used in training. The
adapter's effect depends strongly on the prompt, and v3's six happened to be favourable.

### Reproducibility across pods, and the starting noise

24 clips here repeat clips from v3 (six prompts at seed 100, four variants), which ran on
a different H100 SXM pod. **All 24 have identical quality metrics to v3's, to the last
digit.** Generation is reproducible across pods of the same GPU type, as two A100 pods
were in Stage 7 step 0. So the ~0.2 "re-draw" difference v3 measured is a difference
between GPU types, not run-to-run noise: on one GPU type a clip can be regenerated
exactly.

The starting noise was recorded for a later cross-GPU test
(`artifacts/v4_decide/noise.json`). On this H100:
every prompt at a given seed starts from identical noise; redrawing it with
`torch.randn` on the GPU reproduces Ovi's draw exactly; a CPU draw with the same seed
does not, as expected. Whether an A100 draws the same noise is still untested.

## What this means

- **For generating clips with Ovi:** use plain UniPC. 30 steps (122 s on an H100) is the
  closest to the 48-step reference on every prompt. 20 steps (86 s) is measurably closer
  than 16 and keeps prompt adherence at the reference's level. 16 steps (71 s) keeps
  adherence but its video statistics are no closer to the reference than 12 steps'. 12
  steps loses adherence.
- **For the adapter:** it is not a speed-up. It should not be published as a recommended
  model. What it does show is that 313k parameters on a frozen 11.66B model can move a
  12-step Euler sampler most of the way to a 12-step UniPC one, at the price of softer
  video and a larger audio level error.
- **For the project's hypothesis:** the TBC-sized adapter, trained by trajectory matching
  on 16 trajectories, did not make Ovi faster at equal quality. A better-trained adapter
  (hundreds of prompts, an objective that does not reward the average) is untested.
- **Not shown.** The measures are distances to a 48-step reference and CLIP/CLAP
  adherence, not a perceptual score, and nobody has compared the clips by eye. The 70
  seed-100 clips are in `artifacts/v4_decide/clips/` on the owner's machine for that.
