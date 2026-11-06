# Scoping idea (b) and (c): video + audio weight editing

*What the premises actually are, what the models actually are, and what the gap
between this project and them looks like.*

## The premises, checked

The brief carried three ideas. Two of them rested on claims I had taken on trust.
Checked against HuggingFace and the model cards:

| premise | status |
|---|---|
| MIT's Weight-Space Programming unblocks capability | **verified** — Stage 0 reproduced it |
| Wan 2.2 has an MoE architecture | **verified** — but it is timestep-based, not token-based |
| "Mischief 1" is a neuron-inspired LoRA on Wan 2.2 | **no evidence found** |
| MiniMax H3 is an open-weight joint audio+video model | **verified** — `MiniMaxAI/MiniMax-H3`, 6.3M downloads, tagged `text-to-audio-video` |
| Ovi is a joint audio+video model | **verified** — `chetwinlow1/Ovi`, Apache 2.0, arXiv 2510.01284 |

### "Mischief 1" — not found

Searches for "Mischief" on HuggingFace return only unrelated 2024 Mistral
finetunes (`OwOOwO/mistral_mischief1` through `mischief4`, 0 likes, 4–8 downloads).
A search for "mischief wan" returns an empty list. A web search was blocked by a
bot challenge, so I could not check non-HuggingFace sources.

**I cannot confirm this premise.** It may be a private model, a name from a talk or
a video rather than a published artifact, or a misremembering. Before any work on
idea (b) as stated — "neuron-inspired LoRA with a frame-similarity penalty as in the
Mischief 1 claim" — that claim needs a source. Building to an unverified
specification is how the first Stage 4 attempt wasted a training run.

### Wan 2.2 — real, and the MoE is not what "MoE routing" suggests

`Wan-AI/Wan2.2-T2V-A14B`, Apache 2.0, arXiv 2503.20314. The model card describes the
MoE as:

> "separating the denoising process cross timesteps with specialized powerful expert
> models"

The repository confirms it: `high_noise_model/` and `low_noise_model/`, each a full
6-shard checkpoint. And the config settles what kind of MoE it is:

```json
{"_class_name": "WanModel", "dim": 5120, "num_layers": 40, "num_heads": 40}
```

That is a **single dense DiT**. The "MoE" is two complete dense models, split by
denoising timestep — the high-noise expert handles early denoising, the low-noise
expert handles late denoising. There is no sparse per-token router anywhere.

This matters for idea (b). "Exploit MoE routing" in an LLM means per-token expert
selection: you can measure which expert processes which token and target it. In Wan
2.2 the "routing" is a hard split on the diffusion timestep, chosen before generation
starts. There is no per-token signal to exploit and no router to steer. The nearest
analogue would be "target the expert that handles the timestep range where the
artefact appears" — a much coarser idea, and not what "MoE routing" implies.

Available variants:

| model | params | resolution | runs on |
|---|---|---|---|
| `Wan2.2-T2V-A14B` | 14B MoE | 480P/720P, 5s | multi-GPU |
| `Wan2.2-I2V-A14B` | 14B MoE | 480P/720P, 5s | multi-GPU |
| `Wan2.2-TI2V-5B` | 5B | 720P@24fps | **a single 4090** |

All three are **video only**. None generates audio.

### Ovi — the actual joint audio+video target

`chetwinlow1/Ovi`, Apache 2.0, arXiv 2510.01284, "Twin Backbone Cross-Modal Fusion
for Audio-Video Generation", by Chetwin Low, Weimin Wang and Calder Katyal
(Character AI + Yale).

- **11.66B parameters** (BF16), 70GB of storage
- Built on `Wan-AI/Wan2.2-TI2V-5B`
- A 5B audio branch trained from scratch on in-house audio data
- 960×960, 24fps, 5s or 10s, synchronized audio+video from text or text+image
- ComfyUI integration via `ComfyUI-WanVideoWrapper`

This is the real target for idea (c). It is also the only model here that does what
the brief described.

### MiniMax-H3 — real, but licensed differently and already has a LoRA ecosystem

`MiniMaxAI/MiniMax-H3`, 6,619 likes, 6.3M downloads. Tags include
`text-to-audio-video`, `image-to-audio-video`, `audio-video-generation` and
`synchronized-audio-video`. The repository has a DAC-based audio VAE
(`FL2VA/audio_vae/`) alongside the video path, so it is genuinely joint.

Two things to note before planning around it:

- **License is `other`** — a "minimax-h3-community-license-agreement", not Apache 2.0.
  Ovi and Wan 2.2 are Apache 2.0. If the output is to be published or used
  commercially, the licence matters and H3's is more restrictive.
- **LoRA training on H3 already exists.** `akatz-ai/MiniMax-H3-Character-Swap-LoRA`
  has 722 likes and 142k downloads, built with `ai-toolkit` and targeting ComfyUI.
  That is direct evidence that adapter training on this model is feasible and that
  tooling exists — which is useful for idea (b), and means the training-loop problem
  below is less novel than it looked.

## The gap that matters more than the model choice

Everything in this project is an **autoregressive language model**. Every concept the
method rests on is autoregressive-specific:

| concept used throughout | exists in a diffusion model? |
|---|---|
| token | no — latents over timesteps |
| token cap / truncation | no |
| "thinking block" | no |
| "stop sooner" | no — there is no sequence to stop |
| over-thinking as repeated self-verification | no analogue |
| `hit_cap` as a failure signal | no |

The Stage 4 result — "suppress redundant self-verification and deliver short correct
answers" — has no direct translation. A diffusion model does not deliberate in text.
Its failure modes are visual: temporal inconsistency, motion artefacts, anatomical
errors, audio desync. Detecting those requires a *perceptual* metric, not a token
count.

So idea (b) and (c) are not "apply the Stage 4 recipe to a bigger model". They are
"build a measurement apparatus for a different modality, then find a behaviour in it
that is worth suppressing". That is a larger project than everything done so far.

## What would have to be built

1. **A behaviour metric for video.** Something that says "this generation is
   over-doing X" as a number. Candidates: temporal consistency (frame-to-frame
   similarity), motion magnitude, aesthetic score, CLIP alignment. None is as clean
   as "token count", and all are expensive to compute.
2. **A contrast pair.** The method needs a target and a control prompt whose
   activations differ in the behaviour and not in the content. For text this was
   "12×11 only the digits" vs "12×11 just the number". The video analogue is not
   obvious.
3. **Activation capture in a diffusion UNet/DiT.** The current tooling assumes a
   `transformers` causal LM with named linear layers. Wan 2.2 is a DiT with a
   different module structure. The capture layer would need rewriting.
4. **A training loop.** Diffusion LoRA training is a different shape from the
   supervised next-token loss used in Stage 4. The target is a denoising objective,
   and "suppress a behaviour" has to be expressed as a loss over generated samples —
   which means backpropagating through sampling or using a reward model.

Item 4 is the hard one. Stage 4's advantage was that "stop sooner" could be expressed
as a supervised target on text. There is no equivalent for "generate less of X" in
diffusion without a differentiable reward.

**But item 4 is less novel than it looked.** `akatz-ai/MiniMax-H3-Character-Swap-LoRA`
shows LoRA training on H3 already works via `ai-toolkit`, so the plumbing exists. The
open question is not "can we train a LoRA on a video model" — it is "can we define a
behaviour metric good enough to train against". That is item 2, and it is the real
risk.

## Compute

| task | model | rough requirement |
|---|---|---|
| Wan 2.2 TI2V-5B inference | 5B | 1× 4090 (24GB) — documented |
| Wan 2.2 A14B inference | 14B MoE | multi-GPU, 48–80GB |
| Ovi inference | 11.66B | 48GB+ (weights ~23GB + VAE + T5) |
| Ovi LoRA training | 11.66B | 80GB+ per GPU, likely multi-GPU |

The pods used so far were 24–32GB single GPUs at $0.28–0.89/hr. Ovi training is a
different class: an 80GB A100/H100 is roughly $2–3/hr, and multi-GPU more. A single
training run is plausibly $20–100 rather than the ~$1 the Stage 4 runs cost.

## Recommendation

**Do not start idea (b) as stated.** The "Mischief 1" premise has no source. Building
to it would repeat the Stage 4a mistake — a training run spent on a specification
nobody verified.

**The cheapest next step is a feasibility check on inference**, not training:

1. Run one model once on a 48GB GPU with a handful of prompts. Cost: ~$2–4. Ovi is
   the better choice than H3 here — Apache 2.0, smaller, and the brief's "joint
   audio+video" description matches it exactly.
2. Measure whether a behaviour metric separates good from bad generations. If no
   metric separates them, the method has nothing to target and the project stops
   there for the cost of one pod-hour.
3. Only if a metric works, scope the training loop — and reuse the `ai-toolkit`
   approach that already works on H3 rather than building one.

That ordering is deliberate: it tests the *measurement* before the *method*, which is
the lesson from Stages 1–3. Every failure in this project came from measuring the
wrong thing or measuring it badly. Doing that first on video is cheaper than doing it
after a training run.

### The specific metric question to answer first

The text method needed a number that said "this generation is over-doing X". For
video the candidates are temporal consistency, motion magnitude, aesthetic score and
CLIP alignment. The feasibility check should answer: **does any of these separate a
generation that over-does a behaviour from one that does not, on the same prompt?**

If yes, the method has a target. If no, there is nothing to derive an edit from, and
that is worth knowing for $4 rather than after a $100 training run.

## What this document does not claim

- That Wan 2.2's MoE cannot be exploited — only that its routing is timestep-based,
  so the LLM-MoE framing does not transfer.
- That Ovi is the right target — only that it is the real model matching the brief.
- Any estimate of whether the method works on video. Nothing here has been measured
  on video at all.
