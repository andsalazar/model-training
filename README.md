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

Everything after Stage 0 is best read as a sequence of failures that each narrowed
the problem, ending in one mechanism that works. Claim-by-claim:

| claim | status |
|---|---|
| An edit can be derived from activation measurement | **yes** |
| That edit cut over-thinking on one checkpoint, one GPU | **yes** — 2048 → 164, 1869 → 210 |
| It reproduces across hardware | **no** — the top-12 cut sat in a 0.0012 tie |
| Threshold selection fixes the reproducibility | **yes** — overlap 1.000 at every ratio |
| A stable selection makes the edit safe | **no** — harmful at 4 and 12 columns |
| The over-thinking direction is global | **no** — layer-local (+0.006 across layers) |
| A trained adapter is behaviour-selective | **yes** — targets only, guards keep real content |
| The adapter delivers finished answers | **yes** — 6/6 targets, after targets include the closing marker |
| The adapter generalises to a held-out behaviour | **yes** — `Tokyo`, `Au` never trained on |
| The adapter recipe generalises to format compliance | **no** — 0/8 compliant; the base trace never contains the answer |
| Including guards in training protects them | **no** — it damages them; targets-only is better |

The sequence:

1. **Stage 1** derived a working edit from measurement, twice, and validated it on
   a near-pure-arithmetic probe set.
2. **Stage 2** re-ran the same procedure on different hardware, got the opposite
   verdict, and found why: alignment scores tie to within 0.0012 at the cut, so
   "top 12 columns" was an arbitrary point inside a plateau. The edit also damaged
   behaviour Stage 1 never tested.
3. **Stage 3** replaced the rank cut with a score threshold. Selection became
   perfectly reproducible (overlap 1.000). The edit stayed harmful at every size:
   a 4-column edit (0.11% of the matrix) improved the target 83% while pushing two
   long-form tasks to the token cap. The problem was the direction, not the column
   choice — it encodes a shared "keep checking" state that arithmetic, logic and
   writing all use.
4. **Stage 4** replaced the fixed direction with a LoRA adapter, which can
   *condition* on the input. The first attempt failed on a data error of mine — it
   trained on the answer string and learned to emit `132`, `yes`, and `ail` for a
   prompt whose answer was `nail`. Training on the model's **own truncated
   reasoning** fixed it: targets fell 60–90%, both held-out probes improved, and
   guards kept producing genuine poems and explanations.
5. **Stage 4b–4d** corrected two measurement bugs and one data-design error.
   Re-judged with the generated text preserved, 4b's targets were *unfinished* —
   every output ended mid-deliberation inside the thinking block. 4c appended the
   closing marker and bare answer to each target, which fixed all six targets but
   pushed two guards to the cap. 4d trained on **targets only**, which fixed the
   targets *and* the guards. The mixed set was the problem: including guards
   taught the adapter to shorten everything.
6. **Stage 5** tested whether the recipe generalises to format compliance. It does
   not — 0 of 8 targets compliant. The base model's trace never contains a
   compliant answer, so truncating it teaches "stop mid-sentence". The recipe has
   a precondition: the base trace must contain the desired output.

The honest headline is that four stages produced three negatives and one partial
positive, and every negative narrowed the next attempt. What is reusable: the
activation tooling, the verifier that caught each error, and the negative results
themselves.

- [docs/stage1-findings.md](docs/stage1-findings.md) — the derived edit, with
  corrections after Stage 2
- [docs/stage2-findings.md](docs/stage2-findings.md) — the reproducibility failure
- [docs/stability-selection.md](docs/stability-selection.md) — the selection fix
- [docs/stability-retest.md](docs/stability-retest.md) — why a stable selection
  still failed
- [docs/stage4-findings.md](docs/stage4-findings.md) — the adapter, its first
  failure, and the sampler experiment
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

## Next: the Ovi + TBC-adapter project

The original brief asked to combine MIT's Weight-Space Programming with TBC's
neuron-derived adapter on a joint video+audio model. Two documents scope that:

- [docs/stage6-scoping.md](docs/stage6-scoping.md) — what TBC actually built, what
  is open and what is closed. Mischief 1 is real but **API-only** (no readable
  weights); Ovi is Apache 2.0 with readable weights. The adapter is a 156k-parameter
  residual module in early DiT layers, **not a LoRA**.
- [docs/stage6-literature.md](docs/stage6-literature.md) — the neuroscience and ML
  literature that plausibly informed TBC's work, graded by how well each link is
  actually evidenced. TBC's post has **no bibliography**, so every link is inference.
- [docs/stage6-metric.md](docs/stage6-metric.md) — the collapse metric, built as pure
  math, tested against constructed collapses, then measured on four real Ovi rollouts.
  **Per-frame entropy does not separate a collapse; delta entropy does.** Three
  estimator bugs were caught by testing, and no real 5-second rollout collapses.
- [docs/stage6-pod-setup.md](docs/stage6-pod-setup.md) — getting Ovi running on a
  Blackwell pod: four blockers, three of which produced error messages pointing at the
  wrong thing.

Three premises in the brief did not survive contact with the primary source: TBC used
**Oasis**, not Wan 2.2; the adapter is **not a LoRA**; and the goal was **rollout
stability**, not speed. Wan 2.2's "MoE" is also a timestep-split dense two-expert
ensemble, not per-token routing — so there is no router to exploit. What does survive:
the mechanism (a small weight-space direction used to *suppress* a behaviour) has real
precedent, and WSP's residual-stream measurement maps directly onto TBC's unanswered
placement question.

## License

Apache 2.0
