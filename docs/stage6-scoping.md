# Scoping the Ovi + TBC-adapter project

*What TBC actually built, what is open and what is closed, and why WSP is the right
instrument for the "surgical targeting" question.*

## Correction to the earlier version of this document

An earlier draft concluded that "Mischief 1" did not exist, based on HuggingFace
searches returning nothing. That was wrong. Mischief 1 is real — it is a **hosted
model**, not a published checkpoint, which is why it is not on HuggingFace. The
source is <https://www.tbc.co/post/the-neural-dynamics-adapter> and the model is
served as `tbc/mischief-1` on OpenRouter at $0.005/second.

The earlier draft also described the TBC adapter as "a LoRA". It is not. That
distinction turns out to matter a great deal for this project.

## What TBC actually built

From the primary source (tbc.co, "The Neural Dynamics Adapter"):

**The base model** is the open-source **600M-parameter Oasis** Minecraft world
generation model — a Diffusion Transformer with a ViT spatial autoencoder, action
conditioning, and diffusion forcing (independent per-token noise levels during
training to mitigate error accumulation in autoregressive rollout).

**The failure being fixed** is long-horizon rollout degradation: as rollout length
increases, small errors accumulate, internal representations lose structure, and
outputs degrade. This is a *temporal stability* failure, not a quality failure.

**The adapter** is:

> "a shallow, lightweight module that modifies residual information in early DiT
> layers by incorporating localized spatial activations and a learned projection,
> adding approximately 156k parameters without altering the core model architecture"

Three things in that sentence matter here:

1. **It modifies residual information** — the residual stream. That is exactly the
   tensor WSP reads.
2. **In early DiT layers** — a placement decision. Which layers, and why those, is
   the question WSP is built to answer.
3. **156k parameters** — 0.026% of the 600M base. Tiny, like the 1.77M-parameter
   LoRA that worked in Stage 4.

**The "neuron" part** is the *source of the transformation*, not a runtime
component. The adapter is derived from living neural networks cultured on 4096-
electrode multi-electrode arrays: input is encoded as electrical stimulation, the
network's spatiotemporal response is recorded and decoded into numerical
representations, and those representations are projected into the DiT's
intermediate representations. At deployment it is **entirely in silico** — a fixed
transformation. No biological component runs at inference.

**The metric** is differential entropy across generated frames, as a proxy for
information collapse. Reported results: ~19% improvement over base Oasis, ~15% over
a parameter-matched fine-tuned control, ~5% over a LoRA-adapted model. The
parameter-matched control is the important one — it isolates the adapter's effect
from mere retraining.

## What is open and what is closed

This is the crux of the project's feasibility, and it is asymmetric:

| | Mischief 1 | Ovi |
|---|---|---|
| access | **API only** (`tbc/mischief-1`) | **open weights**, Apache 2.0 |
| weights readable | **no** | **yes** |
| size | undisclosed | 11.66B |
| modality | video | video **+ audio** |
| cost | $0.005/second | free to run locally |

**We cannot read Mischief 1's weights.** It is a hosted product. So WSP cannot be
applied to it directly — there is no checkpoint to measure.

**We can read Ovi's weights.** `chetwinlow1/Ovi`, Apache 2.0, arXiv 2510.01284,
"Twin Backbone Cross-Modal Fusion for Audio-Video Generation" (Character AI + Yale).
11.66B parameters, built on `Wan-AI/Wan2.2-TI2V-5B`, with a 5B audio branch trained
from scratch. 960×960, 24fps, 5 or 10 seconds, synchronized audio+video.

So the project is not "read Mischief 1". It is **"read Ovi, and use WSP to decide
where TBC's adapter idea should go"** — which is what the brief actually asked for.

## Why WSP is the right instrument here

The brief's instinct is sounder than it first appears. Consider what each side has:

- **TBC's adapter** needs a *placement decision*: which early DiT layers, and which
  residual channels within them. The published work says "early DiT layers" without
  saying how that was chosen.
- **WSP** is precisely a method for measuring, per layer and per channel, which
  directions carry a behaviour. Stages 1–4 of this project built and hardened exactly
  that instrument: activation capture, contrast directions, per-column alignment
  ranking, stability-based selection, and independent re-measurement.

So "surgically target where to implement the adapter" is not a metaphor — it is the
literal output of the tooling this project already has. The adapter is a residual
module; WSP measures residual directions. The mapping is direct.

This is a much better fit than the earlier draft assumed. The earlier draft framed
the gap as "everything is autoregressive-specific and nothing transfers". That is
true of the *behaviour* (over-thinking has no diffusion analogue) but false of the
*instrument* (residual-stream measurement transfers directly).

## The four things that would have to be built

### 1. A behaviour metric for video — the real risk

Stage 4's advantage was that "stop sooner" could be expressed as a supervised target
on text: truncate the trace after the answer. There is no equivalent for "generate
less of X" in diffusion without a differentiable reward.

TBC's answer is differential entropy across frames. That is measurable without
training a reward model, which makes it attractive. But it is a *proxy* for
information collapse, and the paper's own framing admits this. Whether it separates
a good rollout from a bad one on Ovi, at reasonable cost, is unknown.

**This is the item to test first, and it is cheap to test.**

### 2. Activation capture on a DiT

The existing `wsp.activations` code assumes a HuggingFace causal LM: `model.layers[i]`,
residual streams over token positions. A DiT has a different shape — patch tokens,
timestep conditioning, cross-attention to text. The capture hook needs rewriting, but
the *math* (contrast directions, cosine alignment, column ranking) is unchanged.

### 3. The audio branch

Ovi's audio branch is a separate 5B backbone. "Both video and audio" means running the
whole pipeline twice, on two different residual structures, and then reconciling them.
The cross-modal fusion means the two branches are not independent — an edit to one may
propagate. That interaction is unmeasured and is the most novel part of the brief.

### 4. The adapter itself

156k parameters in the residual stream is small enough to train on a single 48GB GPU.
The hard part is not the training loop — it is knowing what to train it *against*.

## Recommended first step

**Do not start with training.** The cheapest decisive experiment is a measurement
feasibility check on Ovi:

1. Run Ovi once on a 48GB GPU with 5–10 prompts at 5 seconds. Cost: ~$2–4.
2. Compute differential entropy across frames for each generation.
3. Ask: **does the metric separate a rollout that degrades from one that doesn't, on
   the same prompt?**

If yes, the method has a target and WSP has something to measure. If no, there is
nothing to derive an edit from — and that is worth knowing for $4 rather than after a
$100 training run.

This is the lesson from Stages 1–3: every failure in this project came from measuring
the wrong thing or measuring it badly. Testing the measurement before the method is
the cheapest possible insurance.

## What this document does not claim

- That WSP will find a useful direction in Ovi. Nothing here has been measured on
  video at all.
- That TBC's adapter is reproducible. The biological derivation is not available; only
  the architectural idea (residual module, early layers, ~156k params) is.
- That the audio branch can be edited the same way. It is a different backbone with a
  different residual structure.
- That Mischief 1's weights are obtainable. They are not, as far as this document can
  establish.
