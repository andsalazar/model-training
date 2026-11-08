# Stage 6e: review corrections

*A full review of the Stage 6 measurement pipeline found nine defects. Six changed what
the Stage 6c/6d numbers mean; three were cosmetic or confined to docs. This records each
one, how it was confirmed, its effect on the published result, and the fix.*

The rule that caught most of them is the project's own: **a tool that cannot look must
say so**. Several of these were silent — the run completed, the numbers looked plausible,
and nothing in the report said a measurement had been taken somewhere other than where
the doc said.

## Summary

| # | Defect | Confirmed by | Effect on published result | Fixed in |
|---|---|---|---|---|
| 1 | Attention "layer" was a per-forward call counter, interleaving audio and video | `queries` field in the v1 JSON | **6d result mislabelled**: "layers 0, 2, 4, 8" were audio blocks 0, 1, 2, 4; "layer 11" was video block 5 | `attention_hooks.py` |
| 2 | CFG pass overwrite: the unconditional pass replaced the conditional one at every key | `ovi_fusion_engine.py` call order + `tests/test_hooks.py` | **every 6c/6d capture is of the unconditional (SLG) branch** | `hooks.py`, `attention_hooks.py` |
| 3 | Capture drivers called `generate()` without the config's sampler settings | `inference.py` vs `generate()` defaults | **captured trajectory ≠ the one the speed/quality sweeps measured** | `dit.ovi_sampler_kwargs`, both drivers |
| 4 | `effect_size` always fell back to relative change | `np.var` of a scalar is 0 | 6d effect values are `|Δ|/mean`, not pooled-SD effects | `attention.py` |
| 5 | Contrast and derive silently dropped captured steps paired with uncaptured ones | v1 JSON: 10 steps requested, 4 timesteps compared | 6d compared 4 timesteps, not 10 | `pod_ovi_attention.py`, `pod_derive_direction.py` |
| 6 | Timestep tables in docs came from the Euler solver; Ovi uses UniPC | `pod_timestep_schedule.sh` vs recorded tables | wrong numbers quoted (t=233/756, gaps 111/264/522) | docs, script |
| 7 | "Exactly zero / bit-identical" for heads below the 0.05 reporting threshold | `min_effect=0.05` in the driver | overclaim | 6d |
| 8 | Residual capture aligned on 4 timesteps because both runs captured even indices | recorded tables | 6c direction rests on 4 of 10 possible matched timesteps | `--capture-steps-for` |
| 9 | Windows `create` writes CRLF; bash rejects it on the pod | first chain launch | none (caught at launch) | convert before `scp` |

## 1. The attention layer label was a call counter

`AttentionHooks` incremented `_layer` on every intercepted `flash_attention` call and
reset it per model forward. In the fused path each fusion block calls
`audio_block.self_attn` **then** `vid_block.self_attn`, both of which reach
`ovi.modules.model.flash_attention`. Cross-attention lives in `ovi/modules/fusion.py`,
which imports `flash_attention` from `ovi.modules.attention` and is not intercepted. So
call index $2i$ was audio block $i$ and $2i+1$ was video block $i$, and every record was
filed as `stream="video"`.

The v1 JSON settles it — the token counts are the two streams' sequence lengths:

| v1 "layer" | queries = keys | row entropy | actually |
|---|---|---|---|
| 0 | 157 | 4.66 (log 157 = 5.06) | audio block 0 |
| 2 | 157 | 4.25 | audio block 1 |
| 4 | 157 | 4.31 | audio block 2 |
| 8 | 157 | 3.86 | audio block 4 |
| 11 | 27,900 | 8.40 (log 27,900 = 10.24) | video block 5 |

Fired counts agree: 2,360 calls for 40 forwards is 59 per forward — 30 blocks × 2 streams
in the conditional pass (60) and one block fewer in the SLG-skipping negative pass (58).

**Fix.** Forward pre/post hooks on every `blocks[i].self_attn` of both stacks set the
`(stream, block)` in flight; the intercept reads it. A call outside any `self_attn` is
counted as `unlabelled` and not recorded. The test fake now has two stacks called
audio-then-video, and a regression test gives the streams different token counts so a
mislabel shows up in the shape.

## 2. Every capture was of the unconditional pass

`StepTracker` maps a repeated timestep to the same step, so Ovi's two CFG passes share a
capture key. That was intended (it stops one denoising step consuming two indices), but
the capture dict is keyed by `(stream, layer, step)`, so the **second** pass overwrote the
first. Ovi runs the conditional pass first and the negative pass — negative prompt, block
`slg_layer` skipped — second. Every residual direction and every attention summary in
6c/6d is therefore of the unconditional branch. The existing test asserted the overwrite
in a comment and nobody followed the consequence.

**Fix.** `StepTracker.last_pass` numbers passes within a timestep. `DitHooks` and
`AttentionHooks` take `cfg_pass` (0 = conditional, the default; 1 = unconditional;
`None` = the old last-wins behaviour) and report how many firings they declined. Both
drivers expose `--cfg-pass`.

## 3. The capture drivers sampled a different trajectory

Ovi's `inference.py` reads its sampler settings from the config; `OviFusionEngine.
generate()` has different defaults, and the capture drivers called `generate()` with
neither:

| setting | `inference.py` / config | `generate()` default (used by v1 captures) |
|---|---|---|
| `video_guidance_scale` | 4.0 | 5.0 |
| `audio_guidance_scale` | 3.0 | 4.0 |
| `slg_layer` | 11 | 9 |
| `video_negative_prompt` | "jitter, bad hands, blur, distortion" | "" |
| `audio_negative_prompt` | "robotic, muffled, echo, distorted" | "" |

The speed and quality sweeps went through `inference.py` with the config, so they are
unaffected. The captures were taken on a differently guided rollout whose colour
behaviour was never measured — the "20 drifts, 30 is stable" premise of the contrast was
established under different settings.

**Fix.** `wsp.dit.ovi_sampler_kwargs(config)` returns exactly the keys and fallbacks
`inference.py` uses; both drivers pass it and record it in the report.
`--negative-prompt` now defaults to the config.

## 4. `effect_size` was never a pooled-SD effect

`attention_contrast` divided by `sqrt(mean(var(a.row_entropy[head]), var(b...)))`, but
`row_entropy[head]` is the per-head mean — a scalar — so the variance was always 0 and
the "degenerate head" fallback `|Δ| / mean entropy` always ran. The 6d values (peak
0.1286) are relative changes of about 1–13%, not effect sizes.

**Fix.** Every summariser now accumulates the per-query entropy variance per head
(`AttentionMap.row_entropy_var`); the contrast uses it and labels each result with
`effect_kind` (`pooled_sd` or `relative`). A test checks all three summarisers agree with
a direct computation.

## 5. Silent drops in pairing

Both the attention contrast and the residual derive paired each step of run A with the
nearest timestep in run B's **full** schedule, then `continue`d without a record when that
step had not been captured. Both runs captured even indices only, and the 30-step run's
nearest step to a 20-step timestep is often odd. In the v1 attention run, 10 timesteps
were requested and 4 were compared (999, 952, 882, 768); nothing in the output said so.

**Fix.** Pairing is over captured steps only, and every captured step produces a row —
compared, or `skipped` with the reason. The attention payload reports `n_compared` and
`n_skipped`.

## 6, 8. The real UniPC schedule, and matched capture steps

`pod_timestep_schedule.sh` built the schedule with `solver_name="euler"`. Ovi uses UniPC.
The recorded tables (from the captures themselves) are:

```
20 steps: 999, 989, 978, 965, 952, 937, 920, 902, 882, 859, 833, 803, 768, 728, 681, 624, 555, 468, 356, 208
30 steps: 999, 992, 985, 978, 969, 961, 952, 942, 931, 920, 908, 895, 882, 867, 850, 833, 813, 792, 768, 742, 713, 681, 644, 603, 555, 499, 434, 356, 262, 146
```

At index 18 these are 356 and 768 — still a 412 gap, so the confound 6d removed was
real, but the numbers quoted for it (233 vs 756, gaps 111/264/522) were Euler's.

The tables also show something the v1 schedules wasted: the 20-step run's **even** indices
and the 30-step run's **every-third** indices land on identical values — 999, 978, 952,
920, 882, 833, 768, 681, 555, 356. Capturing `20=0,2,…,18` and `30=0,3,…,27` makes every
pair exact across the whole trajectory, including the low-noise end where colour drift
shows. Both drivers now take `--capture-steps-for`, and the attention driver defaults to
these schedules with `--max-gap 1.0`.

## 7. "Exactly zero" was a threshold

The driver reports heads with effect ≥ 0.05. "Layers 0 and 8 show exactly zero … bit-
identical" meant no head crossed 0.05, not that the states were identical. Corrected in
6d.

## What this means for the hypothesis

The speed result (30 steps, 1.42?1.53? faster with quality held) stands: it came from the
sweeps, which used the correct settings. The re-measured contrasts:

- **Residual (6c v2):** the conclusion survives ? best column alignment 0.112 video /
  0.077 audio against 0.018 chance, plateaus everywhere. But the captured 20-step rollout
  does **not** drift, so this is a step-count direction, not a drift direction.
- **Attention (6d v2 + null):** video's 20-vs-30 contrast is indistinguishable from a
  30-vs-40 null; audio shows a modest excess in blocks 0, 3 and 12 on one seed.

The v1 attention claims ("layer 11 head 17", "layers 0 and 8 exactly zero") are
retracted.
