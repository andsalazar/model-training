# Hand-off: WSP + TBC-style adapter on Ovi

*State of the project as of 2026-11-09, written for the next agent picking it up.
Read this first, then the README's "Where that leaves the hypothesis", then
[stage7-v3-5s.md](stage7-v3-5s.md).*

## 1. The goal (from the project owner)

A learning / proof-of-concept project, to be published (GitHub or Hugging Face) if it
works:

1. Use MIT's **Weight-Space Programming** (WSP: an LLM or measurement reads a network's
   weights and edits them surgically) to "read" **Ovi**, an open-weight joint
   **video + audio** diffusion transformer (11.66B, Apache 2.0).
2. Apply a **TBC / Mischief-1-style adapter**, a tiny neuron-inspired residual module
   (~156k params, "under 0.1% of base") that TBC used to make Wan 2.2 ~5× faster.
3. Test the hypothesis, true or false, that the two together **speed up Ovi's
   generation while keeping quality in both video and audio**, especially at the
   longest length Ovi can produce (10 s), without hurting short clips.

The owner values honest negative results; "testing the hypothesis is the learning".

## 2. Where things stand: the answers so far

| Question | Answer | Evidence |
|---|---|---|
| Can Ovi be sped up without losing quality? | **yes, by configuration**: 30 steps instead of 50 is 1.42× (5 s) / 1.53× (10 s), quality held | [stage6-speed-findings.md](stage6-speed-findings.md) |
| Does WSP find a concentrated target to edit? | **no**: step-count differences are spread across weights (best column alignment 0.11 vs 0.018 chance) and attention (video: no more than a null contrast) | [stage6c](stage6c-direction-measured.md), [6d](stage6d-attention-contrast.md) |
| Does a TBC-style adapter speed Ovi up further? | **yes**: 12 steps + adapter is 2.17× (5 s) / 2.24× (10 s) faster than 30 steps | [stage7-pilot5s.md](stage7-pilot5s.md) |
| Is it better than plain 12 steps? | **yes**: closer to a 48-step teacher on 6 of 9 clips; recovers most of the prompt adherence 12 steps loses (CLIP margin 18.7 vs 16.4; teacher 19.5) | [stage7-v3-5s.md](stage7-v3-5s.md) |
| Is it as good as 30 steps? | **no**: 30 steps wins on all 6 unseen prompts; at 10 s the adapter's video is softer (sharpness 6.7 vs 14.1) | [stage7-v3-5s.md](stage7-v3-5s.md), [stage7-v2-10s.md](stage7-v2-10s.md) |
| Can WSP diagnose and fix the adapter's defects? | **inconclusive**: it localised an end-of-clip flash to a cross-modal interaction, but the flash did not reproduce on other hardware (0 of 54 clips) | [stage7-wsp-adapter.md](stage7-wsp-adapter.md), [stage7-v3-5s.md](stage7-v3-5s.md) |
| Audio? | 12 steps makes audio slightly quieter/brighter; adapters help loudness somewhat but never fix brightness; audio prompt adherence (CLAP) is unaffected by any variant | v2 and v3 docs |
| 30 s? | Ovi has no 30 s model; the 10 s model at 3× length gives one steady scene with local motion; resolution is ~600 px of 960 even at 10 s (inherent to the VAE) | [stage7-long30.md](stage7-long30.md) |

**The most important methodological finding:** the same prompt + seed + settings gives a
*different clip* on an A100 vs an H100, and re-drawing a clip moves the four-metric
quality distance by **~0.2**, which is as large as most differences reported from 3–4 clips.
Any further claim needs several seeds per prompt.

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
tests/            730 tests, all passing (pytest -q, ~15 s, no GPU needed)
docs/             one doc per stage; correction banners on superseded results
artifacts/        LOCAL ONLY (gitignored): adapters, eval JSONs, clips
runs/pods.json    pod registry (ownership guard)
```

### Local artifacts worth knowing (gitignored, on the owner's machine)

| path | what |
|---|---|
| `artifacts/adapter5s_v2/adapter_best.pt` | the v2 5 s adapter (the best general one; blocks 0–2, rank 8, 313k params) |
| `artifacts/adapter10s_v2/adapter_best.pt` | trained at 10 s from the 5 s adapter |
| `artifacts/v3_5s/onpolicy/`, `b03/` | on-policy adapters (overfit; not recommended) |
| `artifacts/v3_5s/eval.json`, `adherence.json`, `audio_mp4.json`, `summary.json` | the v3 evaluation |
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
  files, convert to LF, `scp`, run. Piping a here-string into `python -` **corrupts
  non-ASCII** (dashes, ×, Δ become `?`). Use the edit tool or file writes instead. The
  file-create tool writes CRLF; bash rejects it on the pod.
- **GPU stock**: listings say "Low" but direct creates are often refused; use
  `create_pod_retry.py`. REST v1's `allowedCudaVersions` enum stops at 13.0 while hosts
  report 13.2, so omit it.
- **Runpod billing reads**: pass `lastN` alone or `startTime` and `endTime` together.
- **Teacher-forced validation loss does not predict rollout quality** (it missed the 10 s
  blur and on-policy overfitting). Judge by full rollouts.

## 7. Open items and suggested next steps

In order of value for the owner's goal:

1. **Fix the evaluation protocol first.** 3+ seeds per prompt, ~10 prompts, paired
   comparisons with bootstrap intervals (`scripts/analyse_v3.py` has the paired logic in
   spirit; extend it). Consider a distribution-level video metric (feature statistics
   over many clips) instead of distance to one teacher sample.
2. **Scale training data.** 10 prompts let on-policy fine-tuning overfit. Generate a
   few hundred diverse prompts (e.g. with an LLM via OpenRouter), generate teachers
   (~197 s each on H100) and retrain; that is the most likely way to beat 30 steps.
3. **Re-test the 10 s blur** under the new protocol, and only then add the planned
   fine-detail / spectral term to the loss.
4. **Audio brightness**: never fixed by any variant; WSP flagged audio block 3 once
   (one A100 read). Worth one clean multi-seed test.
5. **Publish.** The original aim: a write-up (README + docs/stage6*–7*) to GitHub, and
   optionally the v2 adapter + inference script to Hugging Face with an honest model card
   (2.17× faster than 30 steps, better than plain 12 steps, below 30-step quality).
   **93 local commits are not pushed to `origin/main`** (github.com/andsalazar/model-training).
   Ask the owner before pushing.

## 8. Account state and spend

- **No pod or volume created by this project's recent sessions is running.** Last pods
  `i7zo87s9uyr99n` (A100, $57.57) and `k8urmixjnf4yif` (H100, $29.09) are terminated.
- The account still holds **six stopped older pods** (`ia30p9xluq53bk`, `8wd35k5h4173dv`,
  `is8p317y8tclot`, `ec9nd87sddpzyf`, `o4uhh2qw28w9zk`, `qvg4nrmiqb8man`) and network
  volume **`a9739uqc2p`** (EUR-IS-1). They may bill for storage. They were not created by
  the recent sessions; leave them to the owner.
- Approximate total GPU spend for the project: ~$120 through Stage 6, plus ~$87 for
  Stage 7's last two pods, roughly **$207**. Earlier figures come from session notes;
  Stage 7's from billing reads.

## 9. Working style the owner expects

- Ask before spending: state the hourly price and an estimate before creating a pod,
  check in between stages, and terminate idle pods.
- Report negatives plainly; correct earlier docs when a result does not hold (the docs
  carry correction banners; keep that practice).
- Commit after each meaningful step, with a `Co-authored-by` trailer.
