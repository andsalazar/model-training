# Stage 7 v2 at 5 s: a scaled-up adapter matches 30 steps on the river prompt

> **Overturned by [stage7-v4-decide.md](stage7-v4-decide.md).** On ten unseen prompts at
> three seeds this adapter is level with plain UniPC at 12 steps and behind plain 20
> steps; 30 steps is closer to the reference on every prompt.

> **Read with [stage7-v3-5s.md](stage7-v3-5s.md).** A re-run on different hardware showed
> that re-drawing the same clip moves the quality distance by ~0.2, as large as most
> differences reported here from 3–4 single clips. The "closer to the teacher than 30 steps" result and the end-of-clip flash did not reproduce (0 of 54 clips showed a flash); "better than plain 12 steps" did.

*The pilot's adapter halved the 12-step quality gap. This run scales it up and adds an
unseen prompt to the evaluation. Driver: `scripts/pod_adapter_run.py --set v2_5s`.*

## What changed from the pilot

| | pilot | v2 |
|---|---|---|
| training trajectories | 6 (4 prompts) | **16 (10 prompts)** |
| validation | the eval seeds (monitoring only) | **2 separate trajectories, unseen prompts**; best epoch selected on them |
| updates | 144 at lr 3e-4 constant | **576**, cosine 1e-3 → 1e-4 |
| eval | river × 3 seeds | river × 3 seeds **+ violin (unseen prompt)** |

Same adapter (rank 8, blocks 0–2, both streams, 313k params), same teacher (48-step
UniPC) and student (12 Euler steps on the UniPC grid). Training took 4.6 h at 27.6 s per
update.

## Validation (unseen prompts: market, rainforest)

| | video | audio |
|---|---|---|
| untrained | 0.0531 | 0.0294 |
| epoch 0 / 1 / 2 | 0.0503 / 0.0464 / **0.0454 (−15%)** | 0.0259 / 0.0239 / **0.0224 (−24%)** |

The last epoch was best and loss was still falling, though the gain had slowed.

## Held-out rollouts

Same-pod wall clock is unchanged: 12 steps 113 s, 30 steps 245 s.

### Video, mean |relative difference| from the 48-step teacher

Sharpness, flicker, motion coherence and colour stability, averaged:

| | UniPC 30 | UniPC 12 | Euler 12 | pilot adapter | **v2 adapter** |
|---|---|---|---|---|---|
| river (3 seeds) | 0.248 | 0.750 | 0.940 | 0.387 | **0.193** |
| violin (unseen prompt) | 0.211 | 0.448 | 0.593 | — | **3.041** |

Motion coherence (teacher 0.556 / 0.653 / 0.587 on the river seeds):

| | UniPC 30 | UniPC 12 | pilot | **v2** |
|---|---|---|---|---|
| river s100 / s200 / s300 | 0.491 / 0.527 / 0.555 | 0.494 / 0.387 / 0.506 | 0.450 / 0.423 / 0.489 | **0.540 / 0.528 / 0.588** |
| violin | 0.526 | 0.430 | — | 0.529 (teacher 0.562) |

On the river prompt the 12-step adapter is now **closer to the 48-step teacher than
30-step UniPC is**, at 2.17× the speed of 30 steps. Motion coherence, the pilot's main
shortfall, now matches or beats 30 steps on every seed.

### An adapter-caused artifact: brightness jumps in the last frames

The violin score is dominated by colour stability (5.40 against the teacher's 0.45)
with a low trend R² (0.095), so it is not drift. Per-frame mean brightness shows what it
is:

| clip | largest frame-to-frame brightness jump | where |
|---|---|---|
| violin, **adapter** | **17.84** | frames 114–119 (the last 7) |
| river s100, **adapter** | **3.42** | frame 114 |
| river s200 / s300, adapter | 0.98 / 0.95 | — |
| every baseline and teacher clip (16) | 0.63–1.51 | no pattern |

The adapter introduces a **brightness flash in the final ~0.3 s** (roughly the last two
latent frames) on 2 of 4 held-out clips, severe on the unseen prompt. Nothing else
produces it. It is localised in time and attributable to the adapter, which makes it the
kind of regression the WSP second stage is meant to trace.

### Audio (same tool for every clip: MP4 audio track)

| mean \|rel diff\| from teacher, 4 clips | UniPC 30 | UniPC 12 | **v2 adapter** |
|---|---|---|---|
| loudness (RMS) | 7.5% | 34% | **19%** |
| brightness (spectral centroid) | 3.0% | 6.4% | 6.0% |

Loudness is improved again (river 18%, pilot 23%), but the slight brightening at 12
steps is **not** corrected, and on violin the adapter is quieter than the teacher (−22%)
where UniPC-12 was louder (+28%). Audio is still the weaker stream for the adapter.

## Reading

| | |
|---|---|
| Speed | 2.17× faster than 30 steps (unchanged) |
| Video, trained-like prompt | **at or beyond 30-step quality** (0.193 vs 0.248), motion coherence fixed |
| Video, unseen prompt | coherence matches 30 steps, but an **end-of-clip brightness flash** |
| Audio | loudness gap roughly halved vs UniPC-12; brightness unchanged; still short of 30 steps |

## Targets for the WSP second stage

1. **The end-of-clip flash**: which block's adapter, at which timestep, writes the large
   change into the last latent frames? Candidate fix: suppress the adapter on the final
   latent frames, or only at the final interval.
2. **Audio brightness**: the adapter does not touch the 12-step brightening; WSP should
   locate where the audio stream's spectral tilt diverges from the teacher.
3. **Generalisation**: one unseen prompt is a thin test; the flash needs checking on more.

## Not established

One unseen prompt, three river seeds, one training run, one placement. Prompt adherence
is still unmeasured. The 10 s run (in progress) starts from this adapter.
