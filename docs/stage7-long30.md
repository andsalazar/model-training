# Stage 7: 30 s on Ovi, the 10 s model run at 3× its trained length

*Ovi ships 5 s and 10 s checkpoints only. This runs the 10 s model with a 3× longer
latent (181 video latent frames = 721 frames, 942 audio tokens) to see whether, and how,
it degrades past its trained length. That decides whether a TBC-style adapter has a job
at 30 s. Method A from the plan; chaining 10 s clips (method B) was not run.*

Tool: `scripts/pod_ovi_long.py`. River prompt, seed 100, 30 UniPC steps, the config's
sampler settings. A 10 s run with the same tool and settings is the baseline.

## Cost and memory

| | 10 s | 30 s |
|---|---|---|
| generate (incl. decode) | 634 s | **3,705 s (62 min)** |
| peak GPU memory | 60.4 GB | 66.2 GB |

Three times the length costs **5.8×** the time: attention is quadratic in tokens
(54,900 → 162,900). Decoding 721 frames fitted. So at 30 s, step count matters even more
than at 10 s: 12 steps would be roughly 25 min instead of 62.

## Quality per 10 s window

| | 10 s baseline | 30 s: 0–10 s | 10–20 s | 20–30 s |
|---|---|---|---|---|
| motion coherence | 0.672 | 0.168 | 0.156 | 0.133 |
| temporal flicker (frame-to-frame change) | 0.322 | 0.076 | 0.077 | 0.078 |
| sharpness | 23.6 | 13.8 | 14.1 | 14.7 |
| colour stability | 5.40 | 3.95 | 0.35 | 0.35 |
| colour trend R² / drift | 0.27 / 7.3 | 0.08 / 1.4 | 0.23 / 0.3 | 0.07 / 0.2 |
| audio RMS | 0.0108 | 0.0083 | 0.0083 | 0.0082 |
| audio centroid | 2,483 Hz | 2,476 | 2,468 | 2,469 |

Whole-clip delta entropy (the Stage 6 collapse metric): mean 2,992 against 3,929 at 10 s,
with no downward trend (slope +0.24, R² 0.20, `collapsing: false`).

## Reading

**The video does not collapse; it holds one scene with local motion throughout.**
Frame-to-frame change is about a quarter of the 10 s baseline in *every* window, motion
coherence is a quarter, and there is no trend across the three windows. The first draft
of this section read that as "spreads or freezes its motion". **Watching the clip
corrected it:** the 30 s video is not frozen. Water ripples and leaf motion continue for
the full 30 s on an unchanging scene. What is missing is large-scale motion: the 10 s
clip's camera movement, which is what drives its much larger frame-to-frame change and
motion-coherence scores. The metrics measured the magnitude of motion correctly but cannot
tell "frozen" from "steady local motion", and the first reading should not have been
written without looking.

**Both clips look low-resolution.** Viewing showed softness in the 10 s clip as well as
the 30 s one, so it is not a length effect. None of the Stage 6 metrics measure
resolution; see "Resolution" below.

**Audio holds up much better.** Level is 23% below the 10 s clip but steady across all
three windows, and timbre is unchanged (centroid within 1%). As in the step-count work,
the audio stream degrades less than the video stream.

## Resolution: inherent to Ovi's output, slightly worsened by the encode

`scripts/pod_effective_resolution.py` measures the share of spectral power finer than a
480-px frame can hold, and an *effective width*: the frame width whose Nyquist limit
contains 99% of the spectral power. On synthetic frames upscaled from a known width it
reads about 1.3× high (120 px reads as 155, 60 as 85), so read it as a ranking, not an
exact width.

| video | effective width (of 960) | power finer than 480 px |
|---|---|---|
| 10 s, 30 steps, **raw VAE decode** of the saved latents | 597 | 0.0194 |
| the same 10 s clip, **as written to MP4** | 557 | 0.0161 |
| 30 s, 30 steps (MP4) | 551 | 0.0226 |
| 5 s, 48 steps (MP4) | 597 | 0.0173 |
| 5 s, 30 steps / 12 steps (MP4) | 718 / 703 | 0.0312 / 0.0355 |
| 5 s, 12 steps + adapter (MP4) | 632 | 0.0258 |

- **The softness is mostly Ovi's own output.** The raw VAE decode already holds detail
  equivalent to roughly a 600-px frame (≈450 px after calibration), not 960. That fits
  the architecture: Ovi's video VAE (from Wan 2.2 TI2V-5B) compresses 16× spatially, so
  a 960–960 frame is decoded from a 60–60 latent grid.
- **The MP4 encode takes a further ~17% of the fine detail** (PSNR raw vs MP4 32.4 dB).
  Ovi's `save_video` writes libx264 at ffmpeg's default quality (CRF 23). Saving at a
  higher quality setting is a free improvement.
- **It is not a length effect**: 30 s and 10 s are the same (551 vs 557).
- **More steps make it softer, not sharper**: 48 steps has less fine detail than 30 or
  12. The extra high-frequency energy at low step counts is residual noise (as the
  pilot found with sharpness), so these measures cannot be read as "higher is better" on
  their own.

So resolution is not something a step-count adapter fixes. The levers are a
higher-quality encode (free), a super-resolution pass after generation, or a different
VAE, none of which is in scope for the TBC + WSP question.

## What this means for TBC + WSP at 30 s

- The 30 s defect is a **length-extrapolation failure** (no large-scale motion), not the
  step-count failure the 5 s adapter was trained on. An adapter trained at 5 or 10 s to
  make 12 steps follow 48 has no reason to restore motion at 30 s. TBC's Oasis work
  targeted long horizons in an autoregressive model; Ovi generates the whole clip at
  once, so its long-horizon failure is a different one.
- A fair test needs **training at 30 s**: a 10 s training step already peaks at 67 GB,
  so 30 s does not fit on one A100-80GB. It would need multi-GPU sequence parallel
  (Ovi supports `sp_size`) or a larger card.
- **Chaining three 10 s clips** (method B) keeps each clip in distribution and is the
  practical route to 30 s today. The 10 s adapter would apply to each clip; audio would
  need stitching at the joins.
- For WSP the 30 s run has the strongest defect seen in the project. Contrasting 30 s
  against 10 s states at matched timesteps would localise where motion is lost, but it
  would diagnose extrapolation, not the adapter.

## Not established

One seed, one prompt, 30 steps only. Visual inspection by the project owner (one viewing,
both clips). Chaining (method B) not run. Whether the prompt's "camera slowly panning"
actually happens in the 10 s clip was not checked frame by frame.
