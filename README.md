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

Stage 0 (reproduction) is complete and verified: the paper's headline result
reproduces, and its arithmetic matches exactly. See
[docs/stage0-findings.md](docs/stage0-findings.md).

Stage 1 produced a working edit, but **Stage 2 showed it does not reproduce**, so
it is not yet a method. The honest current status:

| claim | status |
|---|---|
| An edit can be derived from activation measurement | **yes** |
| That edit cut over-thinking, on one checkpoint and one GPU | **yes** — 2048 → 164 tokens, 1869 → 210 |
| It reproduces on the same procedure, different hardware | **no** — verdict inverted |
| It transfers to other behaviours | **no** — pushed a factual probe and a long-form task to the cap |
| Measurement is deterministic on fixed hardware | **yes** — 7 repeats, stdev 0.0 |
| The over-thinking direction is global | **no** — layer-local (+0.006 across layers) |

The reproducibility failure has a specific cause: the "top 12 columns" cut falls
inside a plateau where alignment scores tie to within 0.0012, so a selection that
changes with the last digit of a float decides the verdict. Two runs with 10 of 12
columns in common got opposite results.

What survives is worth keeping: the activation tooling, the verifier that caught
this, and a set of negative results (contiguous windows are harmful, `hit_cap` is
not looping, a high `contrast_cosine` is not noise). Full analysis in
[docs/stage2-findings.md](docs/stage2-findings.md); Stage 1's evidence, with
corrections, in [docs/stage1-findings.md](docs/stage1-findings.md).

## Verification

An edit only counts if it survives re-measurement. `wsp.verify` compares a
baseline reading against an edited one under a fixed rule and returns
`validated`, `partial`, `no-effect` or `harmful`, with `harmful` checked first so
a helped target cannot mask a broken held-out probe. The `baseline` and
`verify_edit` tools expose it to the editor, so a run cannot report post-edit
behaviour it never measured.

The comparison is a fixed rule rather than a judgement call:

| State | Condition |
|---|---|
| `broken` | the expected answer was present before the edit and is gone after |
| `regressed` | now hits the token cap, or grew past 1.25× baseline |
| `improved` | fell to 0.75× baseline or below |
| `unchanged` | moved less than the 25-token noise floor |

An edit is `validated` only when every target improved and nothing regressed or
broke. A target that improves while a held-out question hits the cap is
`harmful` — that is redistributing the behaviour, not suppressing it.

### Running a verified exploration

`scripts/run_explore_verified.py` runs the editor and then re-measures its edits
itself, so the verdict does not depend on the editor's prose:

```bash
python scripts/run_explore_verified.py --editor deepseek \
    --model Qwen/Qwen3.5-0.8B \
    --checkpoint /workspace/models/Qwen3.5-0.8B/model.safetensors \
    --out /workspace/explore15.json
```

It reads the checkpoints the editor actually wrote from the tool log, reloads
each one, and measures the probe set against a baseline taken on the original
checkpoint. The result lands in the `verification` key of the output JSON,
alongside the transcript and the audit findings.

This matters because both Stage 1 runs that got far enough to matter fabricated
numbers: one inverted a `direction_similarity` sign, the other reported post-edit
token counts without ever calling `measure_loop` after its edits. The auditor
catches those after the fact; re-measurement catches them regardless.

## License

Apache 2.0
