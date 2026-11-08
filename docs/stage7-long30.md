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

**The video does not collapse progressively; it is low-motion from the start.**
Frame-to-frame change is about a quarter of the 10 s baseline in *every* window, motion
coherence is a quarter, and there is no trend across the three windows. In the second
and third windows per-frame colour barely moves at all (stability 0.35). A clip that was
fine for 10 s and then froze would show a healthy first window; this one does not. The
most likely reading is that the model, given 3× the temporal positions it was trained on,
spreads or freezes its motion rather than generating 30 s of it. That is an inference
from metrics. **Watch the clip to confirm it:** `artifacts/long30/30s_30steps_s100.mp4`
(the 10 s baseline is beside it).

**Audio holds up much better.** Level is 23% below the 10 s clip but steady across all
three windows, and timbre is unchanged (centroid within 1%). As in the step-count work,
the audio stream degrades less than the video stream.

## What this means for TBC + WSP at 30 s

- The 30 s defect is a **length-extrapolation failure** (missing motion), not the
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

One seed, one prompt, 30 steps only. No visual inspection yet. Chaining (method B) not
run.
