# Stage 7: WSP as a second step, reading what the adapter does to Ovi

*The adapter-first plan: put a TBC-style adapter on Ovi for speed, then use WSP-style
reads to find and fix what it breaks. This is the second step, run on the 5 s v2 adapter
(the one with the end-of-clip flash) and on the 10 s-trained adapter.*

Tools: `scripts/pod_wsp_adapter.py` (read), `scripts/analyse_wsp_adapter.py` (summary),
`scripts/pod_flash_ablation.py` (causal test). All captures are of the conditional CFG
pass, with blocks labelled by stream (the Stage 6e fixes).

## The read

Four rollouts per prompt/seed: teacher-48, base Euler-12, adapter Euler-12, base UniPC-12.
Attention summaries are taken at the 12-step grid's timesteps, which the teacher visits
exactly (every 4th of 48 steps), at blocks 0, 3, …, 27 of both streams. For each site
the largest per-head effect against the teacher is averaged over timesteps, and
`ratio = adapter / base Euler-12`: **below 1 the adapter moved that site toward the
teacher, above 1 it moved it away.**

### Attention, ratio adapter / base Euler-12

| stream | block | violin (flash) | river 100 (flash) | river 200 (clean) | 10 s river 100 |
|---|---|---|---|---|---|
| video | 0 | 0.40 | 0.79 | 0.57 | 0.73 |
| video | 3 | **1.97** | 1.17 | 0.72 | 0.96 |
| video | 6 | 1.04 | 0.92 | 0.62 | **1.84** |
| video | 12 | 1.24 | 0.84 | 0.84 | **2.19** |
| video | 15 | 1.30 | 1.21 | 0.53 | 1.23 |
| video | 18 | 1.54 | 1.13 | 0.52 | 0.96 |
| video | 21 | **2.16** | 0.99 | 0.68 | 1.30 |
| video | 24 | 0.88 | 1.32 | 0.63 | 1.00 |
| video | 27 | 0.87 | 1.40 | 0.45 | 0.61 |
| audio | 0 | 1.31 | 0.73 | 0.77 | 0.43 |
| audio | **3** | **1.60** | **1.72** | **1.48** | **2.11** |
| audio | 6 | 1.42 | 1.10 | 1.00 | 1.74 |
| audio | 9 | 1.57 | 1.14 | 1.05 | 1.92 |
| audio | 18 | 1.09 | 0.66 | 1.68 | 1.62 |
| audio | 21 | 0.98 | 1.17 | 1.59 | 1.79 |

(Blocks 9 video and 12, 15, 24, 27 audio omitted for width; full tables in
`artifacts/wsp_stage/summary_*.json`.)

What the read shows:

- **On the clean clip the adapter pulls every sampled video block toward the teacher**
  (0.45–0.84). On the two clips that flashed it pushes several mid/late video blocks
  away (violin 3, 18, 21 at 1.5–2.2).
- **Audio block 3 is the one site that moves away from the teacher in every clip**
  (1.48–1.72 at 5 s, 2.11 at 10 s). Block 3 is the first block after the adapters
  (0–2). That is the most consistent target the read produced, and it lines up with the
  defect the adapter has never fixed: audio brightness.
- **At 10 s, video blocks 6 and 12 move away** (1.84, 2.19), alongside the blur.

### Where the adapter writes

Each adapter's own change `|adapter(x) − x| / |x|`, averaged over steps, per latent frame
(video) or time bin (audio):

| | last 2 bins / rest | first bin / rest |
|---|---|---|
| video adapters, blocks 0–2, all four clips | **0.97–1.02** | 0.76–1.08 |
| audio adapter, block 0 | 1.45–1.59 | 0.28–0.86 |
| audio adapters, blocks 1–2 | 0.44–0.73 | 0.66–0.91 |

The video adapters write **uniformly across latent frames**. So the end-of-clip flash is
*not* written directly into the last frames; it forms downstream. The audio block-0
adapter does write more toward the end of the clip, but it does so on the clean clip too,
so this alone does not explain the flash.

## The causal test: which part of the adapter makes the flash?

The read narrowed the question; an ablation answers it. Largest frame-to-frame brightness
change in the last 8 frames:

| variant | violin | river 100 |
|---|---|---|
| full adapter | **17.79** | **3.82** |
| audio adapters off | 1.12 | 1.12 |
| video adapters off | 1.96 | 0.97 |
| adapter off for the **last** 2 steps | 18.18 | 3.73 |
| adapter off for the **first** 2 steps | **0.57** | **0.78** |
| writes zeroed on the last 2 latent frames | 4.25 | 1.94 |
| no adapter | 0.70 | 0.87 |

1. **The flash is a cross-modal interaction.** It needs both streams' adapters; either
   alone does not produce it. In a fused audio-video model, adapting each stream
   independently can create a defect neither stream makes alone, through the
   cross-attention between them.
2. **It is created at the start of sampling and shows at the end of the clip.** Turning
   the adapter off for the two noisiest steps removes it; turning it off for the last
   two does nothing. Zeroing the writes on the final frames only partly helps, which
   agrees with the read: the writes there are not special.

## The fix, evaluated

Four candidate fixes, scored on all four held-out clips against the 48-step teacher
(video: mean |relative difference| over sharpness, flicker, motion coherence and colour
stability; audio: the MP4 audio track, same tool for every clip):

| variant | river × 3 | violin | all 4 | flash, worst clip | audio loudness error |
|---|---|---|---|---|---|
| UniPC 30 (current setting) | 0.248 | 0.211 | **0.239** | 0.5 | **7.5%** |
| UniPC 12 | 0.750 | 0.448 | 0.675 | 1.1 | 34% |
| adapter, as trained | **0.193** | 3.041 | 0.905 | **17.8** | 19% |
| adapter off for the first step | 0.816 | 0.503 | 0.737 | 1.5 | 29% |
| adapter off for the first 2 steps | 0.850 | 0.565 | 0.779 | 0.8 | 33% |
| **audio adapter off for the first 2 steps** | 0.252 | 0.760 | 0.379 | **1.5** | 26% |
| video adapter only | 0.255 | 0.747 | 0.378 | 1.1 | 25% |

- **Switching the whole adapter off early removes the flash and almost all the benefit.**
  The adapter does its useful work in the same first, noisiest steps where it creates
  the flash.
- **The targeted fix the ablation pointed to works for the flash.** Keeping the video
  adapter on and holding the audio adapter off for the first two steps removes the flash
  on every clip (worst 1.49) and cuts the unseen-prompt distance from 3.04 to 0.76.
- **It costs part of the gain.** On the river prompt the fixed adapter (0.252) is back
  to 30-step quality rather than beyond it (0.193 unfixed), and the audio loudness error
  rises from 19% to 26%. The early audio writes help both streams on most clips and
  break the video on some. That trade-off is what a fused audio-video model's
  cross-attention makes of two independently trained adapters.
- **Over all four clips, 30 steps is still the best quality** (0.239 vs 0.379), while
  12 steps with the fixed adapter is 2.17× faster and far better than plain 12 steps
  (0.675).

## What this says about the TBC + WSP hypothesis

| question | answer from Stage 7 |
|---|---|
| Can a TBC-style adapter speed Ovi up? | **yes**: 12 steps is 2.17× (5 s) / 2.24× (10 s) faster than 30 |
| At what quality? | 5 s, trained-like prompt: **beyond 30 steps**; unseen prompt: an artifact; 10 s: audio level at 30-step parity, video **softer** |
| Can WSP-style reads find what the adapter breaks? | **partly**: the write map ruled out the obvious explanation (writes into the last frames); attention flagged audio block 3 in every clip; the causal localisation came from the ablation the read motivated |
| Can a targeted fix repair it? | **the artifact, yes; without cost, no**: the fix removes the flash and gives back part of the gain |

The project's central idea survives in a narrower form than the brief. The speed-up is
real, and the adapter can match or beat the slow setting on familiar content. Both
defects found, the 5 s cross-modal flash and the 10 s blur, come from one cause: the
adapter is trained teacher-forced on one interval at a time, so it never sees how its
early writes compound through a rollout, or through the other stream. The next lever is
the objective, not the placement:

1. **Rollout-aware training**: train on the student's own states (or unroll 2–3
   steps), so a compounding defect is in the loss.
2. **A distribution-matching or sharpness term**: pure MSE lands on the average of the
   plausible outcomes, which is the 10 s blur.
3. **Cross-modal awareness**: train the two streams' adapters jointly with a loss that
   sees their interaction; audio block 3, the first block after the adapters, is the
   WSP-flagged site to watch.

## Not established

Three river seeds and one unseen prompt at 5 s; one seed at 10 s for the read. Prompt
adherence unmeasured throughout. The fix was evaluated at 5 s only.
