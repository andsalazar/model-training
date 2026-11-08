# Stage 7 pilot: a TBC-style adapter on Ovi 5 s, 12 steps

*First test of the adapter-first plan: train a small adapter so 12 sampling steps
follow a 48-step run, then check held-out seeds for speed and for quality in both
streams. One prompt for evaluation, three seeds, one training run.*

## Setup

| | |
|---|---|
| Adapter | `wsp.adapter`: `x + up(gelu(down(layernorm(x))))`, rank 8, before fusion blocks 0–2 of **both** streams, `up` zero-initialised. 313,392 parameters (TBC: ~156k). Ovi frozen. |
| Teacher | 48-step UniPC (Ovi's default sampler settings from the config). The 12-step UniPC grid is exactly every 4th 48-step timestep, so the teacher passes through every student state. |
| Student | 12 Euler steps on the UniPC-12 grid. Trained teacher-forced: the guided velocity (CFG 4.0/3.0, SLG block 11) is regressed onto the velocity that carries the teacher across each interval in one step. Loss is per-stream normalised MSE, video + audio weighted 1:1. |
| Data | 6 trajectories (river × 2 seeds, fan, coast, fox, city) × 12 intervals = 72 samples |
| Training | 2 epochs, 144 AdamW updates, lr 3e-4, grad clip 1.0. 27.6 s per update, 80 min total. |
| Held out | River prompt, seeds 100/200/300 (not trained on). Used for teacher-forced validation (not for model selection; the final epoch is evaluated) and for full rollouts. |

## Teacher-forced validation (held-out seeds)

| | video loss | audio loss |
|---|---|---|
| untrained (= base model) | 0.0661 | 0.0173 |
| epoch 0 | 0.0628 | 0.0167 |
| epoch 1 | **0.0584 (−12%)** | **0.0158 (−9%)** |

Still falling at the end, so the adapter is under-trained. Most of the gain is on the first
interval (video 0.428 → 0.360) and the last (0.104 → 0.082), where one Euler step spans
the largest change in noise level.

## Full rollouts on held-out seeds

Same-pod wall clock: 48 steps 377 s, 30 steps 245 s, **12 steps 113 s**, the same with or
without the adapter. So the adapted 12-step sampler is **2.17× faster than the current
30-step setting** and 3.3× faster than 48.

### Video, against the 48-step teacher

| metric (mean of 3 seeds) | teacher 48 | UniPC 30 | UniPC 12 | Euler 12 | **adapter + Euler 12** |
|---|---|---|---|---|---|
| sharpness | 11.58 | 17.91 | 31.69 | 37.17 | **17.10** |
| flicker | 0.169 | 0.188 | 0.228 | 0.202 | 0.241 |
| motion coherence | 0.599 | 0.524 | 0.462 | 0.348 | 0.454 |
| colour stability | 2.90 | 3.02 | 1.67 | 1.38 | 1.84 |
| **mean \|relative difference\| from teacher** (the four above) | 0 | **0.248** | 0.750 | 0.940 | **0.387** |

Few-step sampling produces **excess high-frequency texture**: sharpness 32 at 12 steps
against 12 for the teacher. The earlier sweeps read rising sharpness as more detail; next
to a 48-step reference it is clearly residual noise. The adapter removes most of it. On
the combined distance it halves UniPC-12's gap to the teacher (0.750 → 0.387). It does
not reach UniPC-30 (0.248).

It also recovers most of what Euler loses in motion coherence (0.348 → 0.454), but not
more than UniPC-12 already has (0.462). Flicker is worse on one seed (s300: 0.377).
The colour trend test is not informative at 5 s: the teacher itself scores R²
0.27–0.46, and the student variants scatter from 0.04 to 0.87.

### Audio, against the 48-step teacher

| | teacher 48 | UniPC 30 | UniPC 12 | Euler 12 | **adapter + Euler 12** |
|---|---|---|---|---|---|
| RMS, mean \|rel diff\| | 0 | 0.062 | 0.359 | 0.403 | **0.234** |
| centroid, mean \|rel diff\| | 0 | 1.8% | 5.9% | 8.4% | **5.2%** |
| final audio latent, rel. error (s100/200/300) | 0 | .054/.081/.162 | .158/.364/.254 | .182/.362/.339 | **.129/.257/.202** |

This is the same audio regression found at 10 s (quieter and brighter with fewer steps),
and stronger at 5 s: UniPC-12 is 36% quieter than the teacher. The adapter cuts that to
23% and is closer to the teacher's audio latent than UniPC-12 on **all three** seeds. On
seed 100 it matches the teacher almost exactly (RMS 0.00567 vs 0.00546, centroid 2459 vs
2461 Hz).

Final *video* latents are 0.88–1.09 away from the teacher for every 12-step variant,
and still 0.67–0.73 away at 30 steps. A few-step run lands on a different sample, not a
noisier copy of the teacher's, so latent distance is not a usable video-quality measure
here. The metric comparison above is.

## Reading

| Question | Answer from the pilot |
|---|---|
| Can a TBC-sized adapter be trained on Ovi? | **yes**: 80 min on one A100, both streams learn, held-out loss falls |
| Does it make 12 steps better? | **yes**: video distance to the teacher halved, audio level error cut from 36% to 23%, excess texture largely removed |
| Is 12 steps + adapter as good as 30 steps? | **no, not yet**: 0.387 vs 0.248 on video, 23% vs 6% on audio level, motion coherence 0.454 vs 0.524 |
| Speed | **2.17× faster than 30 steps**, measured on the same pod |

That is the first result in this project in favour of the TBC half of the hypothesis: an
adapter recovers a real part of the quality lost to step cutting, in both streams, at no
measurable cost per step. It does not yet recover enough to replace 30 steps. Training was
under-converged, so more data and epochs are the obvious next lever before any
WSP-guided correction is worth trying.

## What WSP is for next

The adapter's remaining errors are now concrete and stream-specific: motion coherence
(video), loudness (audio), flicker on one seed. That is the contrast the WSP second stage
needs: adapter-12 versus teacher-48 at matched timesteps (the 12-step grid is on the
48-step grid). The instruments from Stage 6 (residual capture and per-head attention
contrast, both streams, conditional pass, null baseline) apply directly.

## Not established

- One evaluation prompt, three seeds, one training run, one adapter placement (blocks
  0–2). Whether placement matters is exactly the WSP question.
- Prompt adherence (unmeasured anywhere in the project).
- 10 s, the target horizon: a 10 s training step is ~40 s and fits (Stage 7 step 0).
- Teacher-forced training ignores exposure bias: at inference the student sees its own
  states, not the teacher's. The full rollouts above include that effect.
