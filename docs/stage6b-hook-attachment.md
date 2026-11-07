# Stage 6b: putting a hook into Ovi

*Where a capture can attach to Ovi's diffusion transformer, why the obvious site does
not work, and what the code now guarantees about an empty capture.*

## The problem this stage solves

A WSP direction is a contrast between two activation states. To get one from Ovi, a
hook has to read the residual stream during a real generation. Two things made that
harder than it looks on a standard transformer, and one of them would have failed
silently.

## Finding 1: the obvious hook site never fires

On a normal transformer you hook `blocks[i]`: a forward hook fires once per layer and
hands you the layer's output. On Ovi that hook fires **zero times**.

`FusionModel.forward` does not call `blocks[i](...)`. It calls

```python
self.single_fusion_block_forward(vid_block=vid_block, audio_block=audio_block, ...)
```

a bound method that reaches into each block and calls `modulation`, `norm1`,
`self_attn`, `norm3`, `cross_attn`, `norm2` and `ffn` *individually*. That is not
incidental — it is how video-to-audio cross-attention is implemented. The video block
cross-attends to the audio stream and the audio block cross-attends back to the
pre-cross video state (`og_audio`), so the two stacks cannot run as opaque modules.

Verified against a stand-in that reproduces the structure:

```
block-level hooks fired: 0 (expected 0)
norm1 pre-hooks fired: 4 (expected 4 — one per video block)
```

This is the most dangerous finding of the stage, because a capture with no hook
firings produces an empty result, and **an empty capture is indistinguishable from a
capture that found no structure**. It is the same shape as the four earlier failures
in this project: the ffprobe-missing "no audio", the inverted `direction_similarity`
sign, the unmeasured token counts, the substring compliance test. A tool that cannot
look was returning a verdict.

`norm1` is called exactly once per block, per stream, per model pass, in **both** the
fused and unfused paths, so a *pre*-hook there yields the residual stream — the
additive highway an adapter would modify — whichever path runs.

## Finding 2: Ovi is two stacks, not one

`video_model` and `audio_model` are separate `WanModel` instances that share a block
count and are fused block by block. Every block index exists twice, as two tensors of
different hidden widths.

A capture keyed on `(layer, timestep)` therefore lets the audio tensor overwrite the
video one. The stream is now part of the key and part of the schedule, so one rollout
records both stacks — which the project needs, since the question is whether one
method can be aimed at the two modalities separately.

## Finding 3: only video has a frame axis

`rope_apply_3d` flattens video as `[F, H, W]`, so a frame window is a contiguous slice
of the token axis. `rope_apply_1d` flattens audio with no such grid.

Slicing audio by "frames" would average a time range unrelated to the video's frame
boundaries and return a plausible number. `_windowed_mean` raises instead, naming the
reason.

## Finding 4: classifier-free guidance breaks call-counting

Ovi's sampler calls the model **twice per denoising step** at the same timestep —
positive then negative, for both video and audio:

```python
pred_vid_pos, pred_audio_pos = self.model(..., t=timestep_input, **pos_forward_args)
pred_vid_neg, pred_audio_neg = self.model(..., t=timestep_input, **neg_forward_args)
```

Numbering *calls* would file the conditional and unconditional passes as successive
denoising steps, and the schedule would select the wrong pair. `StepTracker` numbers by
distinct timestep value: a repeated timestep resolves back to the step that introduced
it.

The same loop also explains why raw patch capture is affordable: at two calls per step,
a 30-step run is 60 model passes, and a schedule naming 5 layers × 5 steps × 2 streams
records 50 tensors rather than everything.

## Finding 5: `slg_layer` makes one block one-sided

`slg_layer=9` (the default) skips block 9 **in the negative pass only**. A capture at
that layer sees the conditional state twice and the unconditional state never, so a
contrast there is not like-for-like. The default layer set stays off 9 for that reason.

## What an empty capture now reports

`DitHooks.report()` distinguishes the three ways the capture can come back empty:

| Situation | What the report says |
|---|---|
| No hook fired | `"no hook fired: the hook site was never called, so nothing was measured (this is not evidence that the states are identical)"` |
| Hooks fired, schedule declined | The firing count plus `"check that the schedule matches the observed steps (saw N)"` |
| Firings outside a numbered step | A count of firings that were not recorded |

Every case reports the firing count alongside the recorded count, so zero records can
never be read as a measured negative.

## Code

| File | What it does |
|---|---|
| `src/wsp/dit.py` | `CaptureSchedule`, `ResidualCapture`, `frame_window`, `_windowed_mean` — pure array logic, no torch |
| `src/wsp/hooks.py` | `DitHooks` (hook attach/detach, step tracking, report), `StepTracker` |
| `src/wsp/targeting.py` | `find_targets`, `build_plan`, `apply_plan` — direction to weight-column edits |
| `scripts/pod_ovi_capture.py` | Drives `OviFusionEngine` with hooks attached; runs the drifting-vs-stable pair |

`dit.py` deliberately does not import torch, so the reduction maths is testable
without a GPU. `hooks.py` imports torch only where it must.

## Finding 6: the FlashAttention fallback changes the answer, not just the speed

Ovi's README calls FlashAttention "optional", and `attention.py` does fall back to
`torch.nn.functional.scaled_dot_product_attention` when `flash_attn` is missing. But
the fallback path contains:

```python
if q_lens is not None or k_lens is not None:
    warnings.warn('Padding mask is disabled when using scaled_dot_product_attention.')
attn_mask = None
```

Ovi passes `seq_lens`/`k_lens` on every attention call because sequences are padded.
So the SDPA path **drops the padding mask and attends over the padding tokens**. That
is a different computation, not a slower one — and it would silently contaminate any
capture taken through it.

For the earlier speed experiments this did not matter: they measured wall clock and
output quality, and a wheel rendered in roughly the right time is still a valid timing
sample. For activation capture it matters, because the direction would be read from a
state the real model never occupies.

So FA2 is **required** for stage 6b rather than optional, a capture must record which
attention backend produced it, and the build is a source build on A100 (sm_80) because
no wheel matches this torch/Python combination.

### Torch version pinning

`torchaudio` unpinned resolved to 2.11.0, built against CUDA 13, and importing it
under `torch 2.6.0+cu124` fails with `libcudart.so.13: cannot open shared object file`.
That error names a missing library rather than a version mismatch, so it reads like a
broken container. All three of torch/torchvision/torchaudio are now pinned, and the
setup script imports them immediately after installing, so a mismatch fails at install
time rather than during the first capture run.

## Tests

73 tests across `tests/test_dit.py` (42), `tests/test_hooks.py` (19) and
`tests/test_ovi_capture.py` (12). Two are load-bearing for the findings above:

- `TestAttach::test_a_block_level_hook_would_not_fire` — pins *why* `norm1` is the hook
  site rather than trusting the claim.
- `TestHookSite::test_hook_records_the_residual_not_the_normalised_value` — asserts the
  capture matches the raw residual and demonstrably **not** `norm1`'s output. The first
  version of this test used an identity norm, which made the two indistinguishable and
  the test vacuous; it now uses a real scaling layer.

## First real capture — verified

The smoke run (8 steps, 2 layers, 3 capture steps, 1 seed) proves the pipeline on the
actual 11.66B model:

| Field | Value | What it means |
|---|---|---|
| `attention_backend` | `flash_attn_2` | the real path, not the padding-dropping SDPA fallback |
| `model_calls` | 16 for 8 steps | **CFG double-pass confirmed empirically**, not just read from source |
| `fired` | `{audio: 32, video: 32}` | 2 layers × 8 steps × 2 passes, exactly as designed |
| `recorded` | 12 | layers × steps × streams, after the schedule filtered |
| `decoded_declined` | 40 | 64 firings − 12 recorded − 12 overwritten by the second CFG pass |
| `steps_seen` | 8 | distinct timesteps, so guidance did not inflate the count |

And the arrays are real, not empty:

```
capture_8_video_0_0: shape=(27900, 3072) std=0.3862 finite=True
capture_8_audio_0_0: shape=(157, 3072)   std=6.7482 finite=True
capture_8_audio_2_4: shape=(157, 3072)   std=27.0953 finite=True
degenerate captures (non-finite or zero variance): 0
video_8: shape=(3, 121, 960, 960)  audio_8: shape=(80384,)
```

12 of 12 captures non-degenerate. The video residual width is 3072 and there are
27,900 patch tokens at 960×960 for a 121-frame rollout; audio is 157 tokens, which is
why Ovi's two stacks cannot share a capture key.

Timing: 92.7s for 8 steps at ~8.2s/step, plus load time. That matches the earlier
5s/step sweeps, so capture adds no measurable overhead.

## Not yet done

- The direction is computed and projected but not yet *applied*. Nothing has been
  edited, so the hypothesis that a stable-minus-drifting direction can raise the
  quality floor at aggressive step counts remains untested on this model. See
  [stage6c-direction-measured.md](stage6c-direction-measured.md) for the projection
  result and the arithmetic that argues against running the edit as designed.
- Prompt adherence is still unmeasured anywhere in the project.
- **The capture artifacts no longer exist.** The 2.8 GB `pair_20v30.npz` and the
  derived reports were written to the A100 pod's container disk — deliberately, because
  `/workspace` enforces a 640 MB quota — and that pod has since been terminated. The
  measurement is documented but not reproducible without re-running the capture
  (~20 min of generation plus a ~40 min reinstall). The network volume survives; the
  venv and captures do not.
