# Stage 7 v4: the deciding experiment. Does the adapter beat running a few more steps?

*Stage 7 v3 left the adapter's value open. 12 steps with the v2 adapter sits between plain
12 steps and 30 steps, but UniPC at 16 or 20 steps was never run on the same pod, so
nothing said whether the adapter does more than four or eight extra steps would. This
experiment runs that comparison, with enough prompts and seeds to test it.*

**Status: design fixed, not yet run.** Everything under "Design", "Measures" and "How the
result will be read" was committed and pushed before the pod was created. Results are
added underneath.

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

Not yet run.
