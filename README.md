# Weight-Space Programming experiments: Qwen3.5-0.8B and Ovi

Two connected experiments on open-weight models, run on rented GPUs. Every result was
re-measured, and the ones that did not hold are kept in, with correction banners.

1. **Text (Stages 0–5).** Reproduce a published **Weight-Space Programming** edit on
   Qwen3.5-0.8B (a language model reading and writing another network's weights
   directly, with no gradients and no training loop), then try to derive such edits from
   measurement instead of copying them.
2. **Video + audio (Stages 6–7).** Port the tooling to **Ovi**, an 11.66B joint
   video-and-audio diffusion transformer, and test whether a tiny trained adapter, sized
   and placed like TBC's "neural dynamics adapter", together with WSP-style reads of the
   model's internals, can make generation faster without losing quality.
3. **Reading weights (Stage 8).** Repeat the paper's most direct "reading" experiment, a
   language model told to say what a small network does from its raw weights alone, with
   the paper's networks, fresh ones, a control, and five different readers.

Based on [Sparks of Weight-Space Programming: LLMs can natively read & write
neural network weights](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab — MIT CSAIL, November 2026) and on TBC's
[neural dynamics adapter post](https://www.tbc.co/post/the-neural-dynamics-adapter).

Most of what this repository found is negative, or narrower than hoped. It is published
as a record of how the claims were tested and where early results turned out to be
measurement error, not as a new method or a recommended model.

> **Picking this project up?** Start with [docs/HANDOFF.md](docs/HANDOFF.md).

## The two questions behind it

**Can one model "read" another model's parameters?** The project started from curiosity
about the WSP method itself: could the paper's reading of a network's weights be
replicated with a different, cheaper model as the reader?

- **Reading a small network from its raw weights: yes, with the paper's reader only.**
  The paper's Experiment 2 was repeated: all 23,824 weights of a small diffusion model as
  text, no tools, "what images does it generate?". GPT-6 Astra named the right shape for
  the paper's three networks, for three freshly trained ones, and for a plus-shaped cross
  the paper never used, and declined to name anything for a network with shuffled
  weights (0% confidence). Four cheaper readers could not: the best, Kimi, was right on 4
  of 8 and confidently described the shuffled network as a digit. One reply per network,
  nine networks ([stage8](docs/stage8-reading-weights.md)).
- **Reading a language model's weights from a static slice: no.** Given a slice of the
  weight matrix the paper edited in Qwen, all five editor models tried, including the
  paper's own, declined to name columns and said doing so would be guesswork
  ([stage0](docs/stage0-findings.md)).
- **With tools to run the model and measure activations: partly.** A cheap editor
  (DeepSeek, $0.03 per million input tokens) captured a direction from activations,
  edited 12,288 weights and cut an over-thinking prompt from 1,869 tokens to 210, with
  the result judged by an independent verifier ([stage1](docs/stage1-findings.md)). That
  is reading *behaviour through measurement*, with the weights as the place the edit
  lands; it is not reading meaning off raw parameters. The editor never saw a weight
  value, only tensor names and a tool-ranked list of columns.
- **It did not hold up as a method.** The same editor reported numbers it had not
  measured, the edit did not reproduce on other hardware, and a stabilised version
  harmed unrelated tasks ([WRITEUP](docs/WRITEUP.md)).
- **On Ovi** the reads were done by measurement scripts rather than an LLM editor, and
  they found no concentrated place to edit.

**What is TBC's adapter method, and does it carry over?** TBC's adapter could be scoped
but not replicated: there are two public documents, no code, no weights and no
bibliography, and the product is API-only ([scoping](docs/stage6-scoping.md),
[literature](docs/stage6-literature.md)). What could be tested is an adapter of TBC's
size and placement, trained by a standard method. See Part 2.

## Findings at a glance

| Finding | Evidence | Where |
|---|---|---|
| The paper's weight-reading result (Experiment 2) repeats with its own reader, and not with cheaper ones | GPT-6 Astra: 6 of 6 on the paper's and on fresh networks, a new shape named, a shuffled control declined; Kimi, GLM, DeepSeek, Qwen: 4, 2, 0, 0 of 8. One reply each | [stage8](docs/stage8-reading-weights.md) |
| The published WSP edit reproduces on Qwen3.5-0.8B | the paper's 25-problem practice subset: 4.0% → 64.0% (paper's released scores for it: 5.0% → 51.0%); the full MATH-500 was not run | [stage0](docs/stage0-findings.md) |
| An edit derived from activation measurement does not reproduce across hardware | the "top 12 columns" cut sits inside a 0.0012 tie in alignment score | [reproducibility-failure](docs/reproducibility-failure.md) |
| A fixed direction cannot be behaviour-selective; a LoRA trained on the model's own truncated reasoning can | 9 probes on one model; does not carry over to format compliance (0 of 8) | [WRITEUP](docs/WRITEUP.md) |
| LLM editors reported verification numbers they had not measured | a sign-inverted similarity, unmeasured token counts; the reason `wsp.verify` re-measures every edit | [WRITEUP](docs/WRITEUP.md) |
| Ovi at 30 steps instead of 50 is 1.42× (5 s) / 1.53× (10 s) faster with quality held | 9 clips: within re-draw noise of a 48-step reference, same prompt adherence. Ovi's README documents this knob; the measurement is what is added | [stage6-speed](docs/stage6-speed-findings.md), [stage7-v3](docs/stage7-v3-5s.md) |
| WSP-style reads find no concentrated edit target for step count in Ovi | best weight-column alignment 0.11 against 0.018 by chance; video attention differs no more than a null contrast | [6c](docs/stage6c-direction-measured.md), [6d](docs/stage6d-attention-contrast.md) |
| A 313k-parameter adapter (0.003% of Ovi) recovers part of what 12-step sampling loses | closer to the reference on 6–7 of 9 clips; not settled on unseen prompts; below 30 steps on all 6 of them | [stage7-v3](docs/stage7-v3-5s.md#statistical-recheck) |
| One generated clip is too noisy a measurement for the differences first reported | re-drawing a clip moves the quality distance by ~0.2; three earlier findings were overturned | [stage7-v3](docs/stage7-v3-5s.md) |
| One-step training loss does not predict rollout quality | it missed a blur at 10 s and overfitting in on-policy training | [stage7-v2-10s](docs/stage7-v2-10s.md), [stage7-v3](docs/stage7-v3-5s.md) |

## What was run: compute and training

Everything ran on rented Runpod GPU pods, driven from a Windows machine over SSH. Total
GPU spend was roughly **$207** (about $120 through Stage 6 from session notes, $87 for
Stage 7 from billing reads). Stage 8 used no GPU: its networks train on a CPU, and its 45
model calls cost $21.09 through OpenRouter.

| | |
|---|---|
| **Hardware** | A100 80 GB (Ovi sweeps, captures, adapter training), H100 SXM (the v3 evaluation and on-policy training), RTX PRO 4500 Blackwell 32 GB (first Ovi bring-up), smaller pods such as an RTX 4000 Ada for the Qwen stages |
| **Adapter training on Ovi** | 313,392 trainable parameters on a frozen 11.66B model; each update is a full forward and backward pass through the model with gradient checkpointing. 5 s clips: 52 GB peak (out of memory above 79 GB without checkpointing). 10 s clips: 58–67 GB, made to fit by back-propagating the two guidance passes one at a time and moving the T5 text encoder to the CPU |
| **Training runs** | Pilot: 144 updates, 80 min. v2 at 5 s: 576 AdamW updates with cosine decay, 4.6 h. v2 at 10 s: 5.6 h at 98 s per update. On-policy fine-tuning on the adapter's own rollouts (224 s per clip to collect targets on the H100) |
| **Generation at scale** | 48-step reference runs as distillation targets (197 s per 5 s clip on the H100; 986 s per 10 s clip); a 30 s clip from the 10 s model in 62 min at 66 GB |
| **Instrumenting an 11.66B DiT** | residual-stream and per-head attention captures in both streams at matched noise levels, with a null baseline; nine capture defects found in review and fixed ([6e](docs/stage6e-review-corrections.md)) |
| **Text model work** | LoRA fine-tuning of Qwen3.5-0.8B; vLLM serving at 1,539 tokens/s against 36 with `transformers` on the same GPU |
| **Environment work** | pinned torch 2.6.0 with a prebuilt FlashAttention 2.7.4 wheel on Ampere and Hopper; torch 2.7.1+cu128 with a source build on Blackwell ([pod setup](docs/stage6-pod-setup.md)) |
| **Pod lifecycle** | a REST client that refuses to touch pods this project did not create, a create-with-fallback script for tight GPU stock, and 190 scripts (155 of them `pod_*` drivers and checks run on the GPU) kept as the working record |
| **Small diffusion models (Stage 8)** | five DDPMs (an MLP of 23,824 weights, 40,000 Adam steps each, about 2.5 minutes on a CPU), each checked by generating 1,000 images and classifying them; prompts rebuilt byte for byte from the paper's released weights |
| **Tests** | 806 tests, no GPU needed (`pytest -q`; 2 of them need the paper's log release checked out and are skipped otherwise); the statistics quoted for Stages 7 and 8 are pinned to the committed evidence in `artifacts/` |

## Repository layout

```
src/wsp/     library: adapter, DiT and attention capture, quality metrics, verifier, Runpod guard
scripts/     pod_* run on the GPU pod; analyse_* run locally
tests/       unit tests for the library and the scripts' pure logic
docs/        one document per stage, with correction banners on superseded results
artifacts/   the JSON behind the tables in docs/ (adapters and clips are not in the repository)
```

## Part 1: Weight-Space Programming on Qwen3.5-0.8B (Stages 0-5)

### The question

The paper's headline result: GPT-6 Astra read Qwen3.5-0.8B's raw weights,
diagnosed reasoning loops caused by a direction written by MLP layers 15 and 19,
and subtracted that direction from 64 output columns in each layer — 131,072 of
~800M weights (0.016%). MATH-500 went from 7.9% to 48.1%, GSM8K from 13.8% to
67.1%, and IFEval from 21.0% to 62.3%.

The open question this repo is built to answer: **does that capability depend on
the editor being a frontier model?** The paper used Astra. If a cheap open-weight
editor can do the same edit, the technique becomes reproducible by anyone.

### Design

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

### Results

Stage 0 (reproduction) is complete and verified: the paper's edit reproduces on its
25-problem practice subset (4.0% → 64.0%, against the paper's released 5.0% → 51.0% for
that subset; the full MATH-500 was not run), and its parameter arithmetic matches
exactly. See [docs/stage0-findings.md](docs/stage0-findings.md). The full account of
Stages 0–5 is [docs/WRITEUP.md](docs/WRITEUP.md).

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

### Verification

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

#### Running a verified exploration

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

## Part 2: Ovi and a TBC-sized adapter (Stages 6-7)

The original brief asked to combine MIT's Weight-Space Programming with TBC's
neuron-derived adapter on a joint video+audio model.

**What "TBC-style" means here.** The adapter in this repository is a standard bottleneck
residual adapter, `x + up(gelu(down(layernorm(x))))`, placed before Ovi's first three
fusion blocks in both streams and sized like TBC's (313k parameters against their
~156k). It is trained on a frozen Ovi to make 12 sampling steps follow a 48-step run.
TBC's post describes a module built from "localized spatial activations and a learned
projection" derived from recordings of cultured neurons, trained alongside fine-tuning
of the base model, for rollout stability; they published no code or weights. So this is
a test of a TBC-*sized*, TBC-*placed* adapter, not a reproduction of TBC's method, and
there is no parameter-matched control to say the placement matters in itself.

The stage documents, in order:

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
- [docs/stage6-speed-plan.md](docs/stage6-speed-plan.md) — the plan for the speed
  objective, and why it needs a different metric from the stability work.
- [docs/stage6-speed-findings.md](docs/stage6-speed-findings.md) — **the speed result.**
  Denoising steps are 92% of wall clock, so cutting them is the lever: **30 steps is
  1.42× faster at 5s and 1.53× at 10s** with unchanged audio. 20 steps reaches 1.76× at
  10s but its colour drifts in 2 of 3 seeds. The goal is met by configuration alone, with
  no model edit. (The document's reading of the sharper output at fewer steps as "real
  detail" was later withdrawn; it carries a correction banner.)
- [docs/stage6b-hook-attachment.md](docs/stage6b-hook-attachment.md) — **getting a hook
  into Ovi.** The obvious site (a forward hook on `blocks[i]`) fires **zero times**,
  because Ovi's fused loop calls each block's submodules individually so video and audio
  can cross-attend. `norm1` is the site that works in both paths. Also: Ovi's SDPA
  fallback drops the padding mask, so FA2 is required rather than optional.
- [docs/stage6c-direction-measured.md](docs/stage6c-direction-measured.md) — **the
  direction, measured, and a negative result.** The drifting-to-stable contrast at
  matched noise levels is a 2.5–4.7% relative perturbation (cosine 0.999). Projected onto
  the two matrices that write to the residual stream, the best column alignment is 0.129
  against a 0.018 chance level: above chance, but far too weak to justify an edit whose
  effect would be 0.1–0.3% of the state norm.

- [docs/stage6d-attention-contrast.md](docs/stage6d-attention-contrast.md) — **the
  attention-map contrast.** Both streams, 10 blocks each, ten exactly matched timesteps,
  plus a 30-vs-40-step null baseline. Video attention differs between 20 and 30 steps
  **no more than between 30 and 40** (median ratio 1.02); audio shows a modest excess
  (1.31) confined to blocks 0, 3 and 12, on one seed.
- [docs/stage6e-review-corrections.md](docs/stage6e-review-corrections.md) — **nine
  defects found in review**, six of which changed what 6c/6d measured: mislabelled
  attention streams, the unconditional CFG pass overwriting the conditional one, capture
  drivers using `generate()` defaults instead of the config, a degenerate effect size,
  silent pairing drops, and Euler-solver timestep tables. All fixed and re-run.

Stage 7 reverses the order: TBC-style adapter first, WSP second, to diagnose and fix what
the adapter breaks.

- [docs/stage7-adapter-step0.md](docs/stage7-adapter-step0.md) — adapter training fits on
  one A100 (14 s/step at 5 s, 40 s at 10 s); the 10 s quality-loss curve: colour drift at
  12–20 steps, audio 16–24% quieter at 12.
- [docs/stage7-pilot5s.md](docs/stage7-pilot5s.md) — **the first positive result**: a
  313k-parameter adapter halves 12-step's quality gap to a 48-step teacher in both
  streams, 2.17× faster than 30 steps.
- [docs/stage7-v2-5s.md](docs/stage7-v2-5s.md) — scaled up: on the river prompt 12 steps
  + adapter is **closer to the teacher than 30 steps**; on an unseen prompt it produces an
  end-of-clip brightness flash.
- [docs/stage7-v2-10s.md](docs/stage7-v2-10s.md) — at 10 s: the 5 s adapter does not
  transfer (blur); trained at 10 s it is 2.24× faster with audio level at 30-step parity
  and softer video.
- [docs/stage7-wsp-adapter.md](docs/stage7-wsp-adapter.md) — **WSP as the second step**:
  the write map and attention read, then an ablation, localise the flash to a
  cross-modal interaction created in the first two steps; a targeted fix removes it at
  part of the gain.
- [docs/stage7-v3-5s.md](docs/stage7-v3-5s.md) — **the re-evaluation**: 9 clips, prompt
  adherence, training on the adapter's own outputs, and a noise floor. Re-drawing a clip
  moves its quality distance by ~0.2, which overturns the flash and "beats 30 steps"
  findings; the adapter's gain over plain 12 steps and its adherence recovery hold in
  direction. A statistical recheck (exact tests, a symmetric distance, a per-metric
  breakdown) narrows that further: it is not settled on unseen prompts.
- [docs/stage7-long30.md](docs/stage7-long30.md) — 30 s by running the 10 s model at 3×
  length: one steady scene with local motion; resolution is inherent to Ovi's VAE.

TBC published **two documents describing the same adapter at different stages**: a
research post on **Oasis** (600M Minecraft DiT, goal: rollout stability) and the shipped
**Mischief 1** product on **Wan 2.2** (goal: ~5× faster, 80% cheaper). The parameter
figures corroborate one adapter — 156k/600M = 0.026%, consistent with "under 0.1% of
base". So the brief's "TBC used Wan 2.2" is **right for the product**; an earlier version
of this README wrongly refuted it by reading only the research post.

What does not survive: the mechanism is **not a LoRA** (LoRA is a baseline TBC claims to
beat by ~5%), and Wan 2.2's "MoE" is a timestep-split dense two-expert ensemble with no
per-token router to exploit. What does survive, and matters most: TBC's own Figure 7 shows
the adapter's benefit is **concentrated at low DDIM step counts** — at 40 steps the base
model nearly matches it. Stability and speed are two readings of one effect. That is the
mechanistic bridge the brief was reaching for, and WSP's residual-stream measurement maps
directly onto TBC's unanswered placement question.

### Where that leaves the hypothesis

The instrument works end to end and, after the Stage 6e fixes, measures what it says: the
conditional pass, under Ovi's real sampler settings, at exactly matched noise levels, in
both streams. What it found:

| Question | Answer |
|---|---|
| Can Ovi be made faster without losing quality? | **yes, by configuration** — 30 steps is 1.42× (5s) / 1.53× (10s); on 9 clips it is within re-draw noise of a 48-step reference. Ovi's own README documents the knob |
| Does 20 vs 30 steps leave a measurable trace inside the model? | **yes** — 2–8.5% relative residual difference, growing toward low noise |
| Is that trace concentrated in a few weight columns (a WSP edit target)? | **no** — best alignment 0.112 video / 0.077 audio vs 0.018 chance; flat plateaus |
| Is it concentrated in a few attention heads? | **video: no** — indistinguishable from a 30-vs-40 null. **audio: weakly** — blocks 0, 3, 12 exceed the null 1.6–3×, one seed |
| Did the captured 20-step rollout actually drift? | **no** — the colour probe finds oscillation, not drift, at both 20 and 30 steps |

So WSP-first targeting was **not supported on Ovi's video stream** and **open but weak on
audio**: the speed goal was met by configuration without it.

Stage 7 tested the reverse order, adapter first and WSP second:

The final answers, after the v3 re-evaluation on 9 clips (3 river seeds + 6 unseen
prompts) with a measured noise floor:

| Question | Answer |
|---|---|
| Does a TBC-style adapter speed Ovi up? | **the speed comes from the step count**: 12 steps is 2.17× (5 s) / 2.24× (10 s) faster than 30, with or without the adapter. The adapter adds no measurable cost per step; what it can add is quality at 12 steps |
| Is 12 steps + adapter better than plain 12 steps? | **in direction, not settled**: closer to the 48-step teacher on 6 of 9 clips (exact test p = 0.074; 7 of 9, p = 0.016 on a symmetric distance), on 3–4 of the 6 unseen prompts, and most of the gain is one metric (sharpness). It recovers most of the CLIP prompt-adherence margin 12 steps loses (18.7 vs 16.4; teacher 19.5) |
| Is it as good as 30 steps? | **no**: 30 steps is closer to the teacher on all 6 unseen prompts, and the adapter's video is softer than the teacher's (on 7 of 9 clips at 5 s; more at 10 s) |
| Does it beat simply running a few more steps? | **not measured**: UniPC at 16–20 steps was never run on the same pod, so the adapter's speed-up at equal quality is unknown |
| Did training on the adapter's own outputs help? | **no**: better on the training prompt, worse on unseen prompts (it overfits 10 training prompts) |
| Can WSP find and fix what the adapter breaks? | **inconclusive**: the defect it localised (an end-of-clip flash) did not reproduce in 54 re-run clips on other hardware, so it was a property of particular samples |
| Audio? | 12 steps does not hurt audio adherence (CLAP); no adapter fixed the slight brightening at 12 steps |

The hypothesis holds in a narrower form than the brief: cutting steps gives a real
speed-up, and an adapter appears to keep part of the quality that costs, but it does not
match the slow setting, and the WSP-guided fix could not be confirmed. The project's last
lesson is methodological: a single generated clip moves by ~0.2 on the quality distance
when it is merely re-drawn, so differences smaller than that need several seeds per
prompt and many more training prompts ([stage7-v3-5s.md](docs/stage7-v3-5s.md)). The
re-draw difference was seen between an A100 and an H100; its cause (kernel differences
amplified by sampling, or different starting noise) was not tested.

**What would settle it:** 10 prompts × 3 seeds on one pod, comparing a 48-step reference,
UniPC at 12 / 16 / 20 / 30 steps and the adapter, about 5–6 H100-hours by this project's
timings. No adapter weights are published until that is run.

### Reproducing the Stage 7 statistics

```bash
python scripts/analyse_v3.py artifacts/v3_5s          # per-variant means
python scripts/analyse_v3_paired.py artifacts/v3_5s   # paired tests, symmetric distance, per-metric breakdown
```

## Part 3: reading a network from its weights (Stage 8)

Parts 1 and 2 never tested the paper's "reading" claim itself. Stage 8 does, by repeating
the paper's Experiment 2 ([docs/stage8-reading-weights.md](docs/stage8-reading-weights.md)).
A reader gets every weight of a small diffusion model as text, about 100,000 tokens, with
no tools, and must say what the model draws. The design and the scoring rule were
committed and pushed before any model was called.

| reader | paper's 3 networks | 3 fresh, same shapes | 2 fresh, new shape (cross) | shuffled-weights control |
|---|---|---|---|---|
| GPT-6 Astra (the paper's) | 3/3 | 3/3 | 1/2 by the keyword rule, 2/2 on a plain reading | names nothing, 0% confidence |
| Kimi | 3/3 | 1/3 | 0/2 | "digit-4-like figures", 76% |
| GLM | 1/3 | 1/3 | 0/2 | "no large coherent shape", 25% |
| DeepSeek | 0/3 | 0/3 | 0/2 | "a vertical stroke", 74% |
| Qwen | 0/3 | 0/3 | 0/2 | no shape named, 62% |

The paper's result repeats with its own reader and extends to networks and a shape it
had not seen. The project's original question, whether a cheaper model can do the same,
gets a no on this test. Limits: one reply per network; a 12 × 12 toy whose output layer
is laid out like the image; no way to check that the reader did not simulate the network.

```bash
python scripts/read_weights.py score    # re-score the saved replies
```

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

## License

Apache 2.0; see [LICENSE](LICENSE). This covers the code and documents in this
repository. Ovi, Qwen3.5-0.8B and the other models used are under their own licenses.
