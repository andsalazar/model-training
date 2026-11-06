# Speeding up Ovi's longest sequence without losing quality

*The plan for the speed objective, and what makes it different from the stability work
that preceded it.*

## The goal, stated precisely

Speed up the **longest** video Ovi can produce, without losing quality on the
**shorter** ones either.

That is two claims, and they need separate tests:

1. **Long horizon gets faster.** Measured as wall-clock seconds per generation at a
   fixed prompt and seed.
2. **Quality holds at both horizons.** Measured per horizon against that horizon's own
   baseline — never 10s against 5s, because those are different models with different
   frame counts.

The second claim is the one that can fail silently. "Faster" is trivially achievable by
degrading output; the whole difficulty is showing that didn't happen.

## What changed from the stability work

The previous stage asked whether entropy collapse could be measured on Ovi. Answer: the
metric works on constructed degradation, but **no real 5-second rollout collapses.** So
Ovi has no stability problem at these lengths for an adapter to fix.

This stage asks a different question with a different metric. The collapse metric is
still useful — it is a cheap detector for "the rollout degenerated" — but it is no
longer the verdict. Quality is.

| | Stability stage | Speed stage |
|---|---|---|
| Objective | prevent information collapse | reduce wall-clock time |
| Primary metric | delta entropy trend | seconds per generation |
| Constraint | — | four quality metrics within tolerance |
| Failure mode | rollout degenerates | rollout looks worse |

## The tunable surface, and why steps is the lever

Ovi's inference config exposes:

| Knob | Affects compute? | Notes |
|---|---|---|
| `sample_steps` | **yes, linearly** | 50 is the default. 30–40 is what the README suggests for speed. |
| `solver_name` | yes, indirectly | `unipc` is the default; alternatives change steps needed |
| `shift` | no | timestep scheduling |
| `video_guidance_scale` | no | quality/trajectory, not cost |
| `audio_guidance_scale` | no | same |
| `slg_layer` | no | skip-layer guidance target layer |
| `fp8` | memory only | README: "will not have speed up in inference time as it still uses bf16 matmuls" |
| `cpu_offload` | slows down | trades VRAM for time |
| `sp_size` | **yes, via multi-GPU** | sequence parallelism; needs N GPUs |

So on one GPU, **`sample_steps` is the only knob that buys real time.** `fp8` is
explicitly documented as *not* a speed win, and `cpu_offload` is a slowdown. `sp_size`
needs multiple GPUs, which multiplies cost rather than reducing work.

That makes the first experiment simple: sweep steps, measure seconds and quality, and
find where each horizon breaks.

## Hardware reality

The 32GB GPU from the previous stage **cannot run 10 seconds at all**:

- bf16 10s OOMs at 241 frames (`rope_apply_3d` needed 1.26 GiB with 730 MiB free)
- fp8 is restricted by an assertion to `720x720_5s` only

So the speed work runs on an **A100 80GB** (`8tug831argzfn0`, CA-MTL-3, $1.79/hr). 80GB
runs 10s in bf16 with no fp8 and no CPU offload, which also removes those as confounds.

Cost note: $1.79/hr is 2.5× the 32GB pod. A full sweep of 5 step values × 2 horizons is
10 generations; at roughly 2 minutes (5s) and 8 minutes (10s) per generation that is
about **50 minutes of GPU, or $1.50** plus setup.

## The experiment

### Phase 1 — step sweep at both horizons

```
scripts/pod_ovi_speed.py --steps 50 40 30 20 12
```

For each of `960x960_5s` and `960x960_10s`:

- generate at each step count, same prompt, same seed
- record wall-clock seconds and peak VRAM
- measure the four quality metrics on the decoded video
- compare each condition against the **same horizon at 50 steps**

Output is a table: steps → speedup → quality ratios → acceptable yes/no.

### Phase 2 — find each horizon's floor

Phase 1 gives a coarse map. Phase 2 bisects each horizon's floor: the lowest step count
that stays acceptable. The interesting result is if the two horizons have **different
floors** — that would mean the long sequence is more step-sensitive, which is a
non-obvious finding and directly relevant to the goal.

### Phase 3 — the WSP edit, if and only if steps are not enough

Only worth doing if Phase 1 shows a real quality floor before the speed target. Then the
question becomes whether a targeted edit can raise that floor, which is where WSP's
surgical placement applies. **Do not start here.** Step reduction is free, needs no
model edit, and may already deliver the goal.

## Why the quality metrics are reference-free

There is no ground-truth video for a text-to-video prompt, so FVD and PSNR are out. CLIP
score would work but needs a large model on the pod and is too slow to run on every
generation.

The four metrics in `wsp.quality` are cheap, deterministic, and each maps to a visible
defect:

| Metric | What it catches |
|---|---|
| `sharpness` | over-smoothing / mush |
| `temporal_flicker` | frame-to-frame jitter |
| `motion_coherence` | incoherent change instead of motion |
| `colour_stability` | hue or exposure drift |

Each is judged as a **ratio** against the baseline, because absolute values are not
comparable across prompts — a foggy scene is legitimately less sharp than a detailed
one. The tolerance is ±10%, chosen because these metrics vary by a few percent between
seeds of the same prompt.

## The comparison rule

"Acceptable" means every metric's ratio is within ±10% of 1.0. Three deliberate
choices:

- **Direction does not earn a pass.** A rollout that got *sharper* is still a change.
  Calling it acceptable without looking would hide a real difference.
- **A zero baseline is handled, not ignored.** `colour_stability` is exactly 0.0 for a
  static rollout, which is a good result; the ratio is 1.0 when both are zero and
  infinite when only the candidate is non-zero.
- **The baseline is always the same horizon.** 10s is never compared to 5s.

## What would make this fail

Stated up front, because the previous stages each failed on a precondition nobody wrote
down:

1. **No speed/quality trade-off exists.** If quality is already flat across 30–50 steps,
   there is nothing to trade and the answer is "just lower the default", not a model
   edit.
2. **The metrics do not track perceptual quality.** They are proxies. A step reduction
   could pass all four while looking visibly worse in a way none of them capture — for
   example, worse prompt adherence, which is not measured here at all.
3. **Timing is dominated by something other than steps.** The 5-second generation took
   ~12s/step for 30 steps plus a 446s VAE decode tail. If decode dominates, cutting
   steps buys less than expected.

Point 3 is worth checking first from Phase 1's own timings, and point 2 is the honest
caveat on any "quality held" claim from this design.
