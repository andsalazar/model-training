# Hand-off: WSP + TBC-style adapter on Ovi

*State of the project as of 2026-11-09, written for the next agent picking it up.
Read this first, then the README's "Where that leaves the hypothesis", then
[stage7-v3-5s.md](stage7-v3-5s.md). Revised the same day after a pre-publication review:
the Stage 7 statistics were rechecked and several claims below were narrowed.*

## 1. The goal (from the project owner)

A learning / proof-of-concept project, to be published (GitHub or Hugging Face) if it
works:

1. Use MIT's **Weight-Space Programming** (WSP: an LLM or measurement reads a network's
   weights and edits them surgically) to "read" **Ovi**, an open-weight joint
   **video + audio** diffusion transformer (11.66B, Apache 2.0).
2. Apply a **TBC / Mischief-1-style adapter**, a tiny neuron-inspired residual module
   (~156k params, "under 0.1% of base") that TBC reports made Wan 2.2 ~5× faster.
3. Test the hypothesis, true or false, that the two together **speed up Ovi's
   generation while keeping quality in both video and audio**, especially at the
   longest length Ovi can produce (10 s), without hurting short clips.

The owner values honest negative results; "testing the hypothesis is the learning".
Two further aims, stated on 2026-11-09:

- The published repository should also **show the hands-on GPU and model-training
  work**, so the README carries a record of what was run, on what hardware, at what cost.
- The underlying curiosity was the **methods themselves**: whether WSP's "reading" of a
  model's parameters can be replicated with another model as the reader, and what TBC's
  method actually is. The README answers both directly under "The two questions behind
  it"; keep that section in step with any new result.

**What "TBC-style" means here, and what it does not.** The adapter in this repository is
a standard bottleneck residual adapter, `x + up(gelu(down(layernorm(x))))`, placed before
the first fusion blocks and sized like TBC's (313k parameters against their ~156k). It is
trained on a frozen Ovi to make 12 sampling steps follow a 48-step run. TBC's post
describes something different in three ways: a module built on "localized spatial
activations and a learned projection" derived from recordings of cultured neurons,
trained alongside moderate fine-tuning of the base model, for rollout stability on Oasis.
TBC published no code or weights. So this project tests whether a TBC-*sized*,
TBC-*placed* adapter helps few-step sampling on Ovi. It does not reproduce TBC's method,
and there is no parameter-matched control (a LoRA or an adapter at other blocks), so
nothing here says the early-layer residual design matters in itself.

## 1a. In flight: the deciding experiment (started 2026-11-09 20:45 local)

**Read this first if `docs/stage7-v4-decide.md` still says "Not yet run".**

- Pod `wsp-stage7-v4` (H100 SXM, $3.99/hour, recorded in `runs/pods.json`) is running
  `scripts/pod_decide.sh`: 10 unseen prompts × 3 seeds × 7 variants. The real run started
  at 21:23 local and should take about 5.5 hours. Design and decision rule:
  [stage7-v4-decide.md](stage7-v4-decide.md), pushed before the pod was created.
- **A local watchdog** (`scripts/pod_watchdog.py`, log in `runs/watchdog.log`) copies the
  small result files to `artifacts/v4_decide/` every 5 minutes, and when the run finishes
  it copies the seed-100 clips and terminates the pod. It terminates the pod at
  **07:10 local on 2026-11-10** regardless (about $42 in total; the owner's limit is $45).
  It only runs while this machine is on.
- **A pod-side backstop** (`scripts/pod_backstop.sh`) is set to remove the pod at 15:25
  UTC (07:25 local). It is unverified: the pod's own key could not read the pod, so the
  removal may be refused.
- **When picking this up:** check `runs/watchdog.log` for `TERMINATED ... read back 404`.
  If it is missing, read the pod with the Runpod MCP `get-pod`, copy
  `/local/v4_decide/eval/eval.json`, `adherence.json` and `audio_mp4.json` if the pod
  still exists, and terminate it. Then run
  `python scripts/analyse_decide.py artifacts/v4_decide --out artifacts/v4_decide/decide.json`,
  fill in the Results section of the stage document against the rule written there, and
  delete this section.

## 2. Where things stand: the answers so far

| Question | Answer | Evidence |
|---|---|---|
| Can Ovi be sped up without losing quality? | **yes, by configuration**: 30 steps instead of 50 is 1.42× (5 s) / 1.53× (10 s). On 9 clips 30 steps is within re-draw noise of a 48-step teacher and matches its prompt adherence. Ovi's README documents the same knob | [stage6-speed-findings.md](stage6-speed-findings.md), [stage7-v3-5s.md](stage7-v3-5s.md) |
| Does WSP find a concentrated target to edit? | **no**: step-count differences are spread across weights (best column alignment 0.11 vs 0.018 chance) and attention (video: no more than a null contrast) | [stage6c](stage6c-direction-measured.md), [6d](stage6d-attention-contrast.md) |
| Does a TBC-style adapter speed Ovi up further? | **the speed comes from the step count, not the adapter**: 12 steps is 2.17× (5 s) / 2.24× (10 s) faster than 30 with or without it. The adapter costs nothing measurable per step and recovers part of the quality 12 steps loses. Its speed-up at *equal quality* is unknown: UniPC at 16–20 steps was never run on the same pod | [stage7-pilot5s.md](stage7-pilot5s.md), [stage7-v3-5s.md](stage7-v3-5s.md) |
| Is it better than plain 12 steps? | **in direction, not settled**: closer to a 48-step teacher on 6 of 9 clips (exact test p = 0.074; 7 of 9, p = 0.016 on a symmetric distance), but on only 3–4 of the 6 unseen prompts, and most of the gain is the sharpness term. CLIP margin 18.7 vs 16.4 (teacher 19.5), 8 of 9 clips; raw CLIP does not separate them | [stage7-v3-5s.md](stage7-v3-5s.md#statistical-recheck) |
| Is it as good as 30 steps? | **no**: 30 steps wins on all 6 unseen prompts; the adapter's video is softer than the teacher's at 10 s (sharpness 6.7 vs 14.1) and, less so, at 5 s (7 of 9 clips) | [stage7-v3-5s.md](stage7-v3-5s.md), [stage7-v2-10s.md](stage7-v2-10s.md) |
| Can WSP diagnose and fix the adapter's defects? | **inconclusive**: it localised an end-of-clip flash to a cross-modal interaction, but the flash did not reproduce on other hardware (0 of 54 clips) | [stage7-wsp-adapter.md](stage7-wsp-adapter.md), [stage7-v3-5s.md](stage7-v3-5s.md) |
| Audio? | 12 steps makes audio slightly quieter/brighter; adapters help loudness somewhat but never fix brightness; audio prompt adherence (CLAP) is unaffected by any variant | v2 and v3 docs |
| 30 s? | Ovi has no 30 s model; the 10 s model at 3× length gives one steady scene with local motion; resolution is ~600 px of 960 even at 10 s (inherent to the VAE) | [stage7-long30.md](stage7-long30.md) |
| Can a language model read what a network does from its raw weights (the paper's Experiment 2)? | **yes with the paper's reader, no with cheaper ones**: GPT-6 Astra named the shape for the paper's 3 networks, 3 fresh ones and a new shape, and declined on a shuffled control; Kimi 4/8 and fails the control; GLM 2/8; DeepSeek and Qwen 0/8, mostly no answer. One reply per network | [stage8-reading-weights.md](stage8-reading-weights.md) |

**The most important methodological finding:** the same prompt + seed + settings gives a
*different clip* on an A100 vs an H100, and re-drawing a clip moves the four-metric
quality distance by **~0.2**, which is as large as most differences reported from 3–4 clips.
Any further claim needs several seeds per prompt. Two A100 pods reproduced each other
exactly, so the difference tracks the GPU type, but **its cause is untested**: slightly
different bf16 kernels amplified by sampling, or different starting noise (it is drawn on
the GPU). Compare the starting-noise tensors across GPU types before explaining it.

## 3. Chronology (one line per stage)

- **Stages 0–5** (Qwen3.5-0.8B, language): reproduced MIT's WSP edit; derived edits from
  activation measurement (worked once, failed to reproduce across hardware, harmful when
  stabilised); a LoRA adapter trained on the model's own truncated reasoning fixed
  over-thinking selectively; it did not generalise to format compliance.
  [WRITEUP.md](WRITEUP.md) covers 0–5.
- **Stage 6** (Ovi): scoped TBC (two documents; adapter is a residual module, not LoRA;
  benefit concentrated at low step counts); built quality/entropy metrics; found the
  config speed-up; ported WSP capture to Ovi's DiT; measured residual directions and
  attention contrasts. **6e** is a review that found nine defects (mislabelled attention
  streams, CFG pass overwrite, wrong sampler defaults, degenerate effect size, silent
  pairing drops, wrong solver tables) and re-ran everything.
- **Stage 7** (adapter first, WSP second): training fits on one 80 GB GPU; pilot halves
  the 12-step gap; v2 scaled up; 10 s does not transfer zero-shot, trained at 10 s it is
  faster and softer; WSP read + ablation localised a flash; 30 s probe; v3 re-evaluation
  with 6 unseen prompts, prompt adherence, on-policy training and the noise floor.
- **Pre-publication review** (2026-11-09): Stage 7 statistics rechecked and claims
  narrowed; repository made public.
- **Stage 8** (reading weights, no GPU): repeated the paper's Experiment 2 with its
  released networks and prompts, five fresh networks, a shuffled control and five
  readers. Design and scoring were pushed before any model was called. Astra reads the
  shapes; the cheaper readers do not.

## 4. Repository map

```
src/wsp/          library (pip install -e ".[dev]")
  adapter.py        TBC-style ResidualAdapter / OviAdapters (per-stream switches,
                    load_partial), StepGate, Euler-on-UniPC-grid scheduler, recording
                    scheduler, patched_schedulers, target velocity, normalised MSE
  dit.py            DiT capture schedule/residual capture; ovi_sampler_kwargs (use it!)
  hooks.py          DitHooks (norm1 pre-hook), StepTracker with CFG pass numbering
  attention.py      attention summaries (torch/numpy, chunked), contrasts, ranking
  attention_hooks.py  flash_attention intercept labelled by (stream, block)
  quality.py        sharpness / flicker / motion coherence / colour stability
  entropy.py        delta-entropy collapse metric
  targeting.py      direction -> weight columns
  runpod.py         REST client with an ownership guard (runs/pods.json)
  reading.py        Stage 8: shapes, the small DDPM, the paper's prompt and weight-text
                    format, the shuffled control, the keyword scorer, the quote check
  (activations, agent, audit, runner, tools, verify, stage4/5_data: Stages 0-5)
scripts/          pod_* run on the GPU pod; analyse_* / create_pod_retry run locally
  pod_adapter_run.py   Stage 7 driver: teacher | train | onpolicy | eval
  pod_adherence.py     CLIP/CLAP prompt adherence with mismatched-prompt baseline
  pod_wsp_adapter.py   WSP read of an adapter (attention vs teacher, write maps)
  pod_flash_ablation.py  causal ablation of adapter parts
  pod_ovi_setup_v2.sh  proven Ovi install; then pod_make_config.sh (writes the config)
  pod_fetch_10s.sh     10 s checkpoint; pod_v3_smoke.sh / pod_v3_5s.sh: v3 pipeline
  create_pod_retry.py  get a GPU when stock is tight (REST v1 GPU priority list)
  analyse_v3.py        per-variant summary of a v3 eval
  analyse_v3_paired.py paired statistics for a v3 eval: bootstrap, exact sign-flip test,
                       symmetric distance, per-metric breakdown (use this for claims)
  pod_decide.sh        Stage 7 v4 pipeline (pod_decide_smoke.sh first); analyse_decide.py
                       analyses it locally at prompt level
  pod_watchdog.py      local: copy results, terminate the pod at completion or deadline
  pod_backstop.sh      on the pod: self-removal at a deadline (run from a file, never inline)
  read_weights.py      Stage 8, local: build | ask | score | quality. `ask` needs the
                       OpenRouter key and a --max-usd limit; it skips replies it has
tests/            806 tests, all passing (pytest -q, ~15 s, no GPU needed; 2 need
                  runs/paper-logs and are skipped without it)
docs/             one doc per stage; correction banners on superseded results
artifacts/        the evidence JSON is tracked (3 MB); adapters (*.pt), clips (*.mp4),
                  logs and the three largest raw captures are local only (.gitignore)
runs/pods.json    pod registry (ownership guard; gitignored)
```

### Artifacts worth knowing (JSON tracked; `.pt` and `.mp4` on the owner's machine only)

| path | what |
|---|---|
| `artifacts/adapter5s_v2/adapter_best.pt` | the v2 5 s adapter (the best general one; blocks 0–2, rank 8, 313k params) |
| `artifacts/adapter10s_v2/adapter_best.pt` | trained at 10 s from the 5 s adapter |
| `artifacts/v3_5s/onpolicy/`, `b03/` | on-policy adapters (overfit; not recommended) |
| `artifacts/v3_5s/eval.json`, `adherence.json`, `audio_mp4.json`, `summary.json`, `paired.json` | the v3 evaluation and its paired statistics |
| `artifacts/v3_5s/clips/` | storm + violin: teacher / 12-step / v2 / 30-step, for viewing |
| `artifacts/long30/` | 10 s and 30 s clips, resolution measurements |

## 5. How to run things

**Local** (Windows, PowerShell 5.1):
```powershell
.venv\Scripts\python.exe -m pytest -q
doppler run --project model-training --config dev -- <command>   # injects secrets
```
Secrets in Doppler (`model-training` / `dev`): `OPENROUTER_API_KEY`,
`RUNPOD_MODEL_TRAINING_API_KEY` (read by `create_pod_retry.py`; never print it).
SSH key `~/.ssh/id_ed25519` is registered on Runpod as `wsp-model-training`.

**A GPU pod** (needs ≥ 80 GB: 5 s training peaks at 52 GB, 10 s at 58–67 GB):
1. Get one: `doppler run ... -- .venv\Scripts\python.exe -u scripts\create_pod_retry.py --public-key "<key>"`
   (tries A100 → H100 → H200 → RTX PRO 6000; stops after one is created; records it in
   `runs/pods.json`). Read the SSH address back with the Runpod MCP `get-pod`.
2. `scp -r src scripts` to `/local/wsp/`, then strip CR: `find /local/wsp/scripts -name '*.sh' -exec sed -i 's/\r$//' {} +`
3. `bash /local/wsp/scripts/pod_ovi_setup_v2.sh` (~10 min on H100), then
   `bash /local/wsp/scripts/pod_make_config.sh`. Add `pod_fetch_10s.sh` for 10 s work.
4. Run a smoke test (`pod_v3_smoke.sh`) before any long job.
5. Copy results back, then terminate the pod (Runpod MCP `delete-pod`, confirm 404).

Timings (H100 SXM, 5 s clip): 48-step teacher 197 s; 30 steps 125 s; 12 steps 58 s;
on-policy collect 224 s per clip; training step 14.5 s. The A100 is ~1.9× slower.

## 6. Gotchas that cost time (read before acting)

- **Ovi's `generate()` defaults differ from its config** (guidance 5.0/4.0, slg 9, empty
  negative prompts vs 4.0/3.0, 11, real negatives). Always pass
  `**wsp.dit.ovi_sampler_kwargs(config)`.
- **Ovi runs every step twice (CFG)**; captures must pick a pass (`cfg_pass=0`, the
  conditional one), or the second silently overwrites the first.
- **Run Ovi from `/local/Ovi`**: its engine loads a relative config at import time.
- **Install to `/local` (container disk)**, never a network volume (slow, quota).
- **Prebuilt FlashAttention 2.7.4 wheel works on Ampere/Hopper** with torch 2.6.0.
  Blackwell (RTX PRO) needs torch 2.7.1+cu128 and a source FA2 build
  ([stage6-pod-setup.md](stage6-pod-setup.md)); results would not be comparable across.
- **Results are not comparable across GPU types**: always regenerate every compared
  variant on the same pod.
- **PowerShell**: no `&&`; inline quoting into `ssh`/`python -c` breaks. Write scripts to
  files, convert to LF, `scp`, run. This bit again on 2026-11-09: an inline self-removal
  timer lost its quotes, computed a negative sleep and fired at once; it did nothing only
  because the removal was unauthorised. Never chain a destructive command after `;`.
- **A new pipeline script: check its log 30 seconds after launch**, not after ten
  minutes. A mis-named set aborted the first v4 smoke test immediately and cost nine
  minutes of pod time before it was looked at. Piping a here-string into `python -` **corrupts
  non-ASCII** (dashes, ×, Δ become `?`). Use the edit tool or file writes instead. The
  file-create tool writes CRLF; bash rejects it on the pod.
- **GPU stock**: listings say "Low" but direct creates are often refused; use
  `create_pod_retry.py`. REST v1's `allowedCudaVersions` enum stops at 13.0 while hosts
  report 13.2, so omit it.
- **Runpod billing reads**: pass `lastN` alone or `startTime` and `endTime` together.
- **OpenRouter**: large concurrent requests can be refused with HTTP 402
  `in_flight_budget_exhausted` when credits are low relative to what is in flight; rerun
  the same `ask` command afterwards, it only fills in what is missing. Reasoning models
  can spend the whole output limit on reasoning and return an empty answer (Stage 8 saw
  this at 16,000 and at 64,000 tokens). Live prices differ from
  `wsp/editors/registry.py` (checked 2026-11-03); Stage 8 records the cost the API
  reports.
- **The paper's log release** (prompts, networks, replies for all eight experiments) is
  cloned sparsely under `runs/paper-logs` (gitignored):
  `git clone --filter=blob:none --no-checkout https://github.com/RikiyaT/lm_understand-logs runs/paper-logs`,
  then `git sparse-checkout set exp1_which_game exp2_diffusion harness` and `git checkout`.
- **Teacher-forced validation loss does not predict rollout quality** (it missed the 10 s
  blur and on-policy overfitting). Judge by full rollouts.

## 7. Open items and suggested next steps

In order of value for the owner's goal:

1. **Run the deciding experiment first.** 10 prompts × 3 seeds on one pod, each with the
   48-step teacher, UniPC at 12 / 16 / 20 / 30 steps, base Euler-12 and the v2 adapter.
   By the H100 timings (3.7 s per step plus ~13 s fixed) that is about 650 s per
   prompt-seed, **5–6 GPU-hours** in all. It answers what v3 left open: does the adapter
   beat simply running 16–20 steps? It needs `base_unipc16` / `base_unipc20` added to
   `VARIANTS` in `pod_adapter_run.py`, an evaluation set with seeds, and the new names in
   `analyse_v3_paired.py`'s `COMPARISONS`. Average the seeds of a prompt before testing
   (prompts, not clips, are the independent unit) and prefer the symmetric distance.
   Consider a distribution-level video metric (feature statistics over many clips)
   instead of distance to one teacher sample. Check the starting noise across GPU types
   while the pod is up (see section 2).
2. **Scale training data.** 10 prompts let on-policy fine-tuning overfit. Generate a
   few hundred diverse prompts (e.g. with an LLM via OpenRouter), generate teachers
   (~197 s each on H100) and retrain; that is the most likely way to beat 30 steps.
3. **Re-test the 10 s blur** under the new protocol, and only then add the planned
   fine-detail / spectral term to the loss.
4. **Audio brightness**: never fixed by any variant; WSP flagged audio block 3 once
   (one A100 read). Worth one clean multi-seed test.
5. **Make the reading result sturdier** (API only, no GPU). Stage 8 has one reply per
   network. Three more Astra replies per network would cost about $45 at the observed
   $1.67 per reply and would say how stable 7–8 of 8 is. The paper's Experiment 1 (which
   game a policy network plays; networks and prompts are in `runs/paper-logs`) is the
   next closest test and needs no training to repeat with its released networks.
6. **Publish.** The repository is public as a negative-results and methodology case
   study, and `origin/main` (github.com/andsalazar/model-training) is level with local
   work. **Hold the Hugging
   Face adapter release until item 1 is done.** If the adapter goes up, label it a
   research artifact, not a recommended model: 12 steps with it is 2.17× faster than 30
   steps, better than plain 12 steps in direction, and below 30-step quality on every
   unseen prompt.

## 8. Account state and spend

- **No pod or volume created by this project's recent sessions is running.** The last
  two pods (an A100, $57.57, and an H100, $29.09) are terminated.
- The account still holds stopped older pods and a network volume that this project's
  recent sessions did not create. They may bill for storage; leave them to the owner.
  Their IDs are in `runs/account-state.md`, which is local only (`runs/` is gitignored).
- Approximate total GPU spend for the project: ~$120 through Stage 6, plus ~$87 for
  Stage 7's last two pods, roughly **$207**. Earlier figures come from session notes;
  Stage 7's from billing reads.
- Stage 8 spent **$21.09** on OpenRouter (45 replies plus cut-off first attempts), from
  the costs the API reported. The owner approved a $25 limit.

## 9. Working style the owner expects

- Ask before spending: state the hourly price and an estimate before creating a pod,
  check in between stages, and terminate idle pods.
- Report negatives plainly; correct earlier docs when a result does not hold (the docs
  carry correction banners; keep that practice).
- Commit after each meaningful step, with a `Co-authored-by` trailer.
