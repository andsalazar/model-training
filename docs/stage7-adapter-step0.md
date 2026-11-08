# Stage 7, step 0: can an adapter be trained, and what must it fix?

*The plan changes order. Stage 6 used WSP to choose where to put an adapter and found no
target. Stage 7 puts a TBC-style adapter on Ovi first, which is close to what TBC did on
Wan 2.2, and then uses WSP to locate whatever quality the adapter loses. Step 0 answers
the two questions that decide whether that is affordable and what the adapter is for.*

## (a) Training fits on one A100-80GB

`scripts/pod_adapter_probe.py` inserts a residual adapter
`x + up(gelu(down(layernorm(x))))` (rank 8, `up` zero-initialised, so the adapted model
starts identical to the base) before fusion blocks 0, 1 and 2 of both streams: 313,392
parameters, about 2× TBC's 156k. Ovi is frozen. One optimiser step is a full forward and
backward through the 11.66B model, because the gradient has to flow through every block
after the adapter.

| case | video / audio tokens | forward only | train step | peak allocated | fits |
|---|---|---|---|---|---|
| 5s, checkpointing | 27,900 / 157 | 3.8 s | **13.9 s** | 52.3 GB | yes |
| 10s, checkpointing | 54,900 / 314 | 9.8 s | **39.7 s** | 67.3 GB (73.0 reserved) | yes, ~6 GB spare |
| 5s, no checkpointing | 27,900 / 157 | — | — | >79 GB | **OOM** |

36.7 GB is resident before any training (DiT, T5 text encoder, both VAEs). The T5
encoder is only needed to precompute prompt embeddings, so offloading it after that
buys ~11 GB of margin at 10 s. Every adapter's `up` weight received a non-zero gradient
in both streams; the audio gradients are ~30× smaller than the video ones, which a
training run will need to rebalance (per-stream loss weights or learning rates).

The 10 s case ran on the 5 s checkpoint's weights. The two checkpoints share one
architecture, so memory and time carry over; only the values differ.

## (b) The 10 s quality-loss curve

Same river prompt and settings as the Stage 6 speed sweep (`inference.py`, UniPC, shift
5, guidance 4.0/3.0, `slg_layer` 11, the config's negative prompts). 16 and 12 steps at
seeds 100/200/300 are new. 50/30/20 are from
[stage6-speed-findings.md](stage6-speed-findings.md). A 20-step seed-100 anchor was
re-run on this pod and reproduced the earlier pod **exactly**: colour R² 0.589, drift
12.28, sharpness 23.29, audio RMS 0.00981 and centroid 2522.5 Hz. So generation is
deterministic across pods and the documented baselines can be compared directly.

### Colour: drift appears at 20 steps and does not get worse below it

Trend R² of per-frame mean colour (>0.5 means drift) and mean absolute drift:

| steps | seed 100 | seed 200 | seed 300 | seeds drifting |
|---|---|---|---|---|
| 50 | 0.130 / 1.81 | 0.119 / 2.17 | 0.146 / 1.52 | 0 / 3 |
| 30 | 0.274 / 7.24 | 0.383 / 5.23 | 0.090 / 1.95 | 0 / 3 |
| 20 | **0.589 / 12.28** | **0.677 / 7.87** | 0.353 / 5.40 | 2 / 3 |
| 16 | **0.525 / 11.24** | **0.617 / 7.74** | 0.481 / 6.75 | 2 / 3 |
| 12 | 0.395 / 8.76 | **0.537 / 6.28** | **0.640 / 7.93** | 2 / 3 |

Colour drift is a step-count threshold effect somewhere between 30 and 20 steps, not a
smooth decline. At 12–20 steps two of three seeds drift by a similar amount. If the
adapter can fix the drift at 20 steps, it is plausibly fixing the same defect at 12.

### Audio: a small, consistent quietening at 12 steps

| steps | seed 100 RMS / centroid | seed 200 | seed 300 |
|---|---|---|---|
| 50 | 0.01214 / 2475 Hz | 0.00448 / 2663 Hz | 0.02298 / 2320 Hz |
| 20 | 0.00981 / 2523 | 0.00425 / 2688 | 0.02334 / 2312 |
| 16 | 0.00949 / 2538 | 0.00396 / 2719 | 0.02232 / 2316 |
| 12 | 0.00919 (−24%) / 2564 (+3.6%) | 0.00368 (−18%) / 2775 (+4.2%) | 0.01926 (−16%) / 2351 (+1.4%) |

At 20 steps the change is mixed by seed. At 12 steps **all three seeds are quieter**
(−16 to −24% RMS) and slightly brighter (+1.4 to +4.2% centroid). There is still no
timbre trend within a clip (R² ≤ 0.003). That is the first audio regression measured in
the project: small, but consistent in direction.

### Video detail and motion (seed 100, the only seed with a 50-step reference)

| steps | sharpness | flicker | motion coherence | colour stability |
|---|---|---|---|---|
| 50 | 14.76 | 0.198 | 0.615 | 2.83 |
| 20 | 23.29 | 0.344 | **0.670** | 6.77 |
| 16 | 28.54 | 0.312 | 0.535 (−13%) | 6.45 |
| 12 | 27.70 | 0.304 | 0.515 (−16%) | 6.17 |

Motion coherence held at 20 steps and falls below baseline at 16 and 12. Sharpness keeps
rising, which at this level is as likely to be retained high-frequency noise as real
detail; there is no reference-based metric to tell them apart.

### Speed on this pod

Wall clock is linear in steps: 12 → 383 s, 16 → 459 s, 20 → 534 s (18.9 s per step plus
a ~156 s fixed decode/setup cost). 30 and 50 steps were not timed on this pod, so the
estimates below use that linear fit. They are not measurements.

| steps | measured or estimated | vs 50 | vs 30 |
|---|---|---|---|
| 50 | ~1,101 s (est.) | 1.00× | — |
| 30 | ~723 s (est.) | ~1.52× | 1.00× |
| 20 | 534 s | ~2.06× | ~1.35× |
| 16 | 459 s | ~2.40× | ~1.58× |
| 12 | 383 s | ~2.87× | ~1.89× |

(The earlier pod measured 1.53× at 30 steps and 1.76× at 20. Its 20-step runs were
slower, so speedups should only be compared within one pod.)

## What the adapter has to fix

At **12 steps** (≈1.9× faster than the current 30-step setting), relative to 50 steps:

1. **colour drift**, in 2 of 3 seeds, and the same as at 20 steps;
2. **audio quietening**, −16 to −24% RMS in all 3 seeds;
3. **motion coherence**, −16% on the one seed with a reference.

Those are three measured defects in two streams, which is the setting the WSP second
stage needs: the adapter-on vs 50-step comparison will contain real regressions to
locate.

## Not established

- Prompt adherence, still unmeasured anywhere in the project.
- One prompt. Every 10 s result is the river prompt.
- Whether sharpness gains are detail or noise.
- Motion coherence at seeds 200/300 against a 50-step reference (not recorded earlier).
