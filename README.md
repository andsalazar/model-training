# Weight-Space Programming experiments

A reproduction and extension harness for **Weight-Space Programming** — having a
language model read and write another network's weights directly, with no
gradients and no training loop.

Based on [Sparks of Weight-Space Programming: LLMs can natively read & write
neural network weights](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab — MIT CSAIL, November 2026).

## Why this exists

The paper's headline result: GPT-6 Astra read Qwen3.5-0.8B's raw weights,
diagnosed reasoning loops caused by a direction written by MLP layers 15 and 19,
and subtracted that direction from 64 output columns in each layer — 131,072 of
~800M weights (0.016%). MATH-500 went from 7.9% to 48.1%, GSM8K from 13.8% to
67.1%, and IFEval from 21.0% to 62.3%.

The open question this repo is built to answer: **does that capability depend on
the editor being a frontier model?** The paper used Astra. If a cheap open-weight
editor can do the same edit, the technique becomes reproducible by anyone.

## Design

The editor is a pluggable backend, so the comparison is a config flag rather
than a rewrite:

| Alias | Model | $/M in | $/M out |
|---|---|---|---|
| `astra` | `openai/gpt-6-astra` | 10.00 | 50.00 |
| `kimi` | `moonshotai/kimi-k3.1-pro` | 2.00 | 10.00 |
| `muse` | `meta/muse-glimmer-30b` | 0.30 | 1.20 |
| `deepseek` | `deepseek/deepseek-v4.1-flash-1017` | 0.03 | 0.48 |

All four are reached through one OpenRouter key, so transport, retries, and
accounting are identical across arms.

## Setup

Secrets live in Doppler, never in files:

```bash
doppler run --project model-training --config dev -- wsp doctor
```

Required: `OPENROUTER_API_KEY`. Optional: `HF_TOKEN`, `WANDB_API_KEY`,
`RUNPOD_MODEL_TRAINING_API_KEY`.

```bash
uv venv
uv pip install -e ".[dev]"
```

## Usage

```bash
# Verify secrets are wired up
doppler run --project model-training --config dev -- wsp doctor

# List editor candidates and pricing
wsp editors

# Send one prompt to an editor, report latency and cost
doppler run --project model-training --config dev -- wsp ping --editor deepseek

# Summarize a checkpoint without loading the weights
wsp inspect models/Qwen3.5-0.8B/model.safetensors

# Render a tensor slice as text for an editor
wsp slice models/Qwen3.5-0.8B/model.safetensors \
    --tensor model.layers.15.mlp.down_proj.weight --cols 0:64
```

## Safety

`wsp.runpod` refuses to modify any pod not recorded in `runs/pods.json`. The
account holds a pod this project did not create; the guard rejects it before any
network call is made, and that behaviour is pinned by tests.

## Status

Stage 0 — reproduction harness. The weight-slicing, editor, and guard layers are
built and tested. Scoring requires a GPU and runs on rented pods.

## License

Apache 2.0
