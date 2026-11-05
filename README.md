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

Everything after that is a negative result, and an instructive one. The current
claim-by-claim status:

| claim | status |
|---|---|
| An edit can be derived from activation measurement | **yes** |
| That edit cut over-thinking on one checkpoint, one GPU | **yes** — 2048 → 164, 1869 → 210 |
| It reproduces across hardware | **no** — the top-12 cut sat in a 0.0012 tie |
| Threshold selection fixes the reproducibility | **yes** — overlap 1.000 at every ratio |
| A stable selection makes the edit safe | **no** — harmful at 4 and 12 columns |
| It transfers to other behaviours | **no** — pushes long-form tasks to the cap |
| The over-thinking direction is global | **no** — layer-local (+0.006 across layers) |

The sequence:

1. **Stage 1** derived a working edit from measurement, twice, and validated it on
   a near-pure-arithmetic probe set.
2. **Stage 2** re-ran the same procedure on different hardware, got the opposite
   verdict, and found why: alignment scores tie to within 0.0012 at the cut, so
   the "top 12 columns" was an arbitrary point inside a plateau. It also found
   the edit damages behaviour Stage 1 never tested — including two of its own
   arithmetic held-outs.
3. **Stage 3** replaced the rank cut with a score threshold. The selection became
   perfectly reproducible (overlap 1.000). The edit remained harmful anyway: a
   4-column edit (0.11% of the matrix) improved the target by 83% while pushing
   two unrelated long-form tasks to the token cap.

The conclusion is that the problem is the **direction**, not the column choice. It
encodes a shared "keep checking" state that arithmetic, logic and long-form
writing all use, so suppressing it makes short answers shorter and long answers
fall apart. A behaviour-selective mechanism is needed — which points at a trained
adapter rather than a single subtracted vector.

What is reusable: the activation tooling, the verifier that caught the whole
thing, and a set of negative results worth having (`hit_cap` is not looping, a
high `contrast_cosine` is not noise, contiguous windows are harmful, direction is
layer-local).

- [docs/stage1-findings.md](docs/stage1-findings.md) — the derived edit, with
  corrections applied after Stage 2
- [docs/stage2-findings.md](docs/stage2-findings.md) — the reproducibility failure
- [docs/stability-selection.md](docs/stability-selection.md) — the fix
- [docs/stability-retest.md](docs/stability-retest.md) — the re-test, and why a
  stable selection still fails
- [docs/reproducibility-failure.md](docs/reproducibility-failure.md) — standalone
  write-up of the failure

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
