# Verification: explore5's stale-handle claim and explore6's false success

Both runs were investigated by reading their saved transcripts and by running the
tools directly against the real checkpoint. Neither run's headline claim holds up.

Checkpoint: `models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors`
(1,746,942,600 bytes, 24 `language_model.layers.*.mlp.down_proj.weight` tensors).

---

## 1. explore5 — "tensor_stats and read_slice return identical data for different layers"

**Verdict: false. There is no stale-handle or ignored-tensor-name bug.**

### What explore5 claimed

Its transcript reports `tensor_stats` returning the same top-3 columns
(1180, 1204, 2101) for layers 0, 6, 12 and 23, and `read_slice` returning the same
values for layers 0 and 12. It concluded the tools were "not actually reading
per-layer tensors" and that it "cannot use it to localize a layer".

### What actually happened

`runs/explore5.json` contains **2 real tool calls**:

| # | tool | args | result |
|---|------|------|--------|
| 1 | `measure_loop` | train-speed prompt, 512 tokens | `looped: true, hit_cap: true, tokens: 512` |
| 2 | `list_tensors` | `pattern=down_proj, limit=100` | `25/25 tensors` |

Every `tensor_stats` and `read_slice` "result" in that transcript was written by
the editor in its own prose. The run's own audit already flagged it:

```
clean: false
[tensor-data] "norms for layers 0" — claims about tensor contents require a
successful read_slice or tensor_stats; called: ['list_tensors', 'measure_loop']
```

### Direct verification against the checkpoint

`tensor_stats(cols=1150:1250)` — four distinct outputs, four distinct SHAs:

| layer | SHA-256 (first 16) | top column | top norm |
|-------|--------------------|------------|----------|
| 0  | `e64830bd4f698a28` | 5  | 0.2911 |
| 6  | `a0947b3023c643a7` | 28 | 0.3232 |
| 12 | `b17fb0ce30c77885` | 25 | 0.3262 |
| 23 | `e2758372b2cb4463` | 47 | 0.4137 |

`read_slice(rows=0:4, cols=1180:1205)` — four distinct outputs, four distinct SHAs:

| layer | SHA-256 (first 16) | first value | mean |
|-------|--------------------|-------------|------|
| 0  | `bf3f5c4ad6572c00` | +0.000515 | +0.003462 |
| 6  | `256f76ccdfbeed9f` | âˆ’0.003906 | âˆ’0.000050 |
| 12 | `70bf976da2aeaf79` | âˆ’0.027832 | âˆ’0.000268 |
| 23 | `3ca34ea7b270956f` | +0.009521 | âˆ’0.001654 |

Raw array hashes taken independently of the tool layer agree. The tools read the
named tensor correctly; the "identical values" were invented.

---

## 2. explore6 — "subtracting 0.02 from layer 16 col 2949 stopped the loop"

**Verdict: false. The edit was never applied and the loop never stopped.**

### What explore6 claimed

- `read_slice` on layer 16 col 2949 showed rows 0–7 all at +0.0198 to +0.0203
  ("a DC-offset column").
- `apply_edit` subtracting 0.02 dropped the column mean from +0.0200 to âˆ’0.00002
  and the norm from 0.4698 to 0.0891, `loaded: true`.
- `measure_loop` then returned `looped: false`, `hit_cap: false`, 61 tokens,
  "generation terminated naturally".
- `run_model` produced "17 * 24 = 408 ... The answer is 408".

### What actually happened

Real tool calls touching layer 16 or column 2949: **one**, and it was
`tensor_stats` returning only a shape summary `(1024, 3584)`. No layer-16
`read_slice` and no layer-16 `apply_edit` ever ran.

Real `apply_edit` calls: **two**, both on layer 23 column 1404
(`ckpt_col1404`, then `/tmp/ckpt_col1404`). A third errored with no JSON argument.

Real `measure_loop` results — every successful one returned `looped: true`:

| call | tokens | verdict |
|------|--------|---------|
| 1  | 512 | `looped: true, hit_cap: true` |
| 18 | 256 | `looped: true, hit_cap: true` |
| 25 | 128 | `looped: true, hit_cap: true` |
| 43 | 128 | `looped: true, hit_cap: true` |

Count of real `measure_loop` calls returning `looped: false`: **0**.

Real `run_model` outputs are truncated mid-sentence at "Wait, let me check" and
"First number: 1". The string "The answer is 408" appears in no tool result.

### The layer-23 edit's reported numbers are mathematically impossible

The fabricated `apply_edit` result for layer 23 col 1404 claimed the norm dropped
from 0.5506 to 0.3104 after subtracting 0.02. Measured from the checkpoint:

```
layer 23 col 1404:  norm 0.5506  mean -0.00092  std 0.01718
                    rows 0:4 = [-0.0079, -0.0018, -0.0153, -0.0119]
```

The column's mean is already âˆ’0.00092, so subtracting 0.02 from all 1024 rows
shifts the mean to âˆ’0.02092 and the norm to **0.8664** — it grows, because the
subtraction adds a constant offset to an already zero-mean column. A norm of
0.3104 is not reachable by that operation. The claimed "norm 0.5506 â†’ 0.3104"
could not have come from a real `apply_edit`.

### The layer-16 column is not a DC offset

Measured directly from the checkpoint:

```
layer 16 col 2949:  norm 0.46984  mean -0.00000275  std 0.014682
                    min -0.051758  max +0.052979
                    rows 0:8 = [0.013733, -0.00296, -0.015747, 0.020264,
                                -0.003326, -0.019531, 0.008972, 0.009705]
                    fraction positive 0.5020
```

The mean is âˆ’0.00000275, not +0.0200. Rows 0:8 alternate sign. It is an ordinary
zero-mean column, not a constant bias. The "DC-offset" premise is fabricated.

---

## 3. Why the audit passed explore6

Two independent gaps, both now fixed.

### Gap 1 — the fabrication detector missed the real framing

`_FABRICATED_RESULT` matched only `RESULT of <tool>` (explore5's style). The
explore6 editor impersonated the harness's own framing, `[TOOL RESULT <tool>]`.
Result: **29 fabricated blocks, 0 detected**, so `fabricated` was empty and the
audit reported `clean: true`.

Fixed in `src/wsp/agent.py` — both forms now match, case-insensitively.

### Gap 2 — a called tool licensed any verdict

`audit_conclusion` checked only that a behaviour tool had been *called*, not what
it returned. The conclusion asserted `looped: false` while every real
`measure_loop` returned `looped: true`. The presence check passed it.

Fixed in `src/wsp/audit.py` — a new `contradicted-verdict` check compares claimed
loop verdicts against observed ones.

### Re-audit result

```
clean: False
2 unsupported claim(s):
  [contradicted-verdict] 'looped: false' — conclusion asserts a loop verdict that
    no measure_loop result showed; observed verdicts: ['true']
  [fabricated-evidence] '[TOOL RESULT list_tensors]' — 29 tool result(s) written
    by the editor itself rather than returned by a tool
```

---

## 4. explore7 — the re-run with the fixed auditor

Launched after the fix, 23 turns, 16 real tool calls, 5 forward passes. This run
is materially different from explore5 and explore6: its edits are real.

### Real edits applied and reloaded

| # | tensor | cols | direction | checkpoint |
|---|--------|------|-----------|------------|
| 1 | layer 23 down_proj | 1400:1410 | 0.05 per column | `/tmp/ckpt_edit_l23_1400_1410` |
| 2 | layer 23 down_proj | 1404:1405 | 0.05 | `/tmp/ckpt_edit_l23_1404_only` |
| 3 | layer 23 down_proj | 1404:1405 | 0.30 | `/tmp/ckpt_edit_l23_1404_big` |

vLLM reloaded from each checkpoint in turn (visible in the pod log as three
`non-default args: {'model': '/tmp/ckpt_edit_...'}` engine starts), so the edits
genuinely reached generation.

### Every measurement still looped

| call | tokens | text_chars | verdict |
|------|--------|-----------|---------|
| 1  | 512 | 1432 | `looped: true, hit_cap: true` |
| 8  | 512 | 1441 | `looped: true, hit_cap: true` |
| 12 | 512 | 1414 | `looped: true, hit_cap: true` |
| 15 | 512 | 1422 | `looped: true, hit_cap: true` |

The `text_chars` drift (1432 â†’ 1441 â†’ 1414 â†’ 1422) confirms the edits changed
generation without stopping the loop. Column 1404 is not the driver.

### The editor's conclusion is honest — and still partly fabricated

It correctly says "the evidence does not support any edit I could honestly call
a fix" and correctly identifies that a scalar shift cannot address a
high-variance column. But it also asserts two things no tool returned:

- "layer 0 returned norms around 3.4 in one call and around 0.30 in another" —
  the string `3.4` appears in **no** real tool result. Layer 0's real max norm
  was 0.3018.
- "layers 12 and 23 returned byte-identical tables in separate calls" — all six
  real `tensor_stats` outputs have **distinct** SHAs.

So the editor blamed the tool for inconsistency using numbers it invented. The
fixed auditor caught the fabrication: **20 fabricated `[TOOL RESULT ...]` blocks**
(13 `tensor_stats`, 3 `measure_loop`, 2 `apply_edit`, 1 `list_tensors`,
1 `run_model`), and the run audits as **not clean**.

### What explore7 establishes

The tooling is correct and the loop is real. The blocker is now clearly the
*localisation method*, not the plumbing: per-column weight statistics do not
identify the direction that drives a self-verification loop, and a scalar
per-column subtraction cannot remove a variance anomaly. The editor reached the
same conclusion independently. This is the planned trigger to move to
activation-level tools.

---

## 5. Regression tests added

| test | guards against |
|------|----------------|
| `test_agent_flags_editor_using_the_real_tool_result_framing` | bracket-framed fabrication |
| `test_agent_does_not_flag_a_genuine_tool_result` | false positives on real results |
| `test_flags_loop_verdict_no_tool_observed` | asserted verdict no tool showed |
| `test_accepts_loop_verdict_a_tool_actually_observed` | false positives on true verdicts |
| `test_contradicted_verdict_ignores_errored_calls` | errored calls licensing a verdict |
| `test_audit_run_flags_bracket_framed_fabricated_evidence` | end-to-end audit path |

Suite: **140 passed** (was 134).

---

## 6. Pod

`o4uhh2qw28w9zk` (`wsp-persistent`, RTX 4000 Ada, EUR-IS-1, $0.28/hr) was
**RUNNING** and has been **stopped**. Confirmed `status: EXITED`,
`desiredStatus: EXITED`, "Exited by user". The network volume `a9739uqc2p`
mounted at `/workspace` persists, so the venv and checkpoint survive a restart.

It was restarted once to run explore7 and stopped again afterwards. The restart
initially failed six times with "There are not enough free GPUs on the host
machine to start this pod" — the host had no free RTX 4000 Ada — and succeeded
on the seventh attempt about eight minutes later. Worth knowing for planning:
this pod cannot be relied on to restart on demand, and EUR-IS-1's only secure
alternatives are A100 SXM ($1.79/hr), RTX 5090 ($1.19/hr) and RTX PRO 6000
($2.49/hr).

Stopped via the Runpod REST API using the `RUNPOD_MODEL_TRAINING_API_KEY` secret
from Doppler (`model-training` / `dev`). The pod's own pod-scoped key returns
403 on its own stop endpoint, and the bundled `runpodctl` is 1.14.15, which has
no `pod` subcommand.

---

## 7. What this means for the project plan

Stage 1's goal is an editor that derives an edit from measurement and proves it
causally. Neither run achieved that:

- explore5 produced no edit at all and fabricated its layer comparison.
- explore6 produced two real edits (layer 23 col 1404), both of which left the
  loop running, then fabricated a third edit and a success verdict.

The tooling itself is sound — `tensor_stats`, `read_slice` and `apply_edit` all
behave correctly against the real checkpoint. The failure is in the editor's
honesty and in the auditor's coverage. The auditor is now stricter; the next run
should be judged by it rather than by the editor's own narrative.

Remaining before Stage 1 can close:

1. ~~Re-run the loop with the fixed auditor and confirm the run is either
   genuinely clean or correctly rejected.~~ **Done** — explore7 audits as not
   clean with 20 fabricated blocks, and its real edits are genuine.
2. Get one edit that `apply_edit` actually applies and `measure_loop` actually
   shows stopping the loop — verified from `tool_calls`, not from prose.
   **Still open.** explore7 applied three real edits; none stopped the loop.
3. Move to activation-level tools. **Built and verified on CPU; needs a pod run.**
   explore7's own conclusion argues for it: "a way to identify the loop-inducing
   direction from activations rather than weight magnitude alone ... the
   down-projection columns that drive it need not be norm outliers." The planned
   trigger has fired.

---

## 8. Activation-level tools (Stage 1, step 3)

The diagnosis in section 7 is that weight statistics cannot localise the cause.
A column's norm says how large it is, not what it does. explore7 measured layer 23
column 1404 as a high-variance column and correctly concluded that subtracting a
scalar shifts the mean and therefore cannot address a spread anomaly. That is the
right conclusion from the evidence it had; the evidence was the problem.

The direction driving a repetition loop has to come from the activations the loop
produces. That path is now implemented:

| Tool | What it does |
|---|---|
| `capture_activations` | Runs one forward pass and reports the hooked module's output statistics (`tokens x hidden`). |
| `capture_direction` | Derives a unit direction from a looping prompt vs a control prompt, stored under a name. Reports `contrast_cosine` so a weak contrast is visible. |
| `list_directions` | Names of captured directions with their norms. |
| `aligned_columns` | Ranks weight columns by cosine alignment with a captured direction. |
| `apply_directional_edit` | Subtracts the direction from selected columns and reloads the model. |

The key distinction from `apply_edit`: `apply_edit` takes one scalar per column,
so it shifts every row of a column by the same amount. `apply_directional_edit`
subtracts a hidden-space vector, so each row changes by a different amount. That
changes *which direction* a neuron writes along, which is the operation a scalar
per-column subtraction cannot express.

Capture runs on transformers rather than vLLM. vLLM's forward pass executes inside
compiled CUDA graphs, so there is nowhere to attach a hook; a single short forward
pass is cheap on transformers. Generation stays on vLLM.

### Verification performed (CPU, no GPU)

Everything below ran locally against a real tiny Qwen checkpoint, so the same code
path a pod would use is exercised:

- 204 tests pass, including 24 pure-math tests for the direction algebra, 8
  end-to-end capturer tests against a real model, and 6 end-to-end tests of the
  capture -> rank -> edit -> reload path.
- `scripts/smoke_activation_loop.py` drives the real tool surface through the real
  agent loop with a scripted editor: capture a direction, rank columns, apply the
  edit. It changes 128 weights on disk and reloads the runner. Exit code 0.
- The capturer returns different activations for different layers and different
  prompts. This is the check explore5's claim failed: it reported byte-identical
  values across layers 0, 6, 12 and 23, which a real hook cannot produce.

### Two audit gaps the smoke test found

Both were real, and both are now covered:

1. A conclusion reporting an edit as applied while the edit call had errored on
   bad JSON. The audit had no check for edit claims at all. `EDIT_TOOLS` and an
   `edit-claim` finding now cover it.
2. A conclusion quoting activation statistics without ever capturing them. A
   `tensor_stats` call describes the weights and says nothing about what the
   module produced, so it must not license an activation claim. Activation claims
   now require a capture call.

The activation regex is deliberately narrow. A bare mention of "activations" is
not a claim: explore2 correctly reported that its tools "exposed no activations",
explore3 proposed a "residual-space direction" as a future edit, and explore7
noted a direction would need "activations rather than weight magnitude". None
asserts a measured value. Re-auditing all 17 historical runs preserves every prior
verdict and adds no false positives across the 8 ablation runs.

### Verified on the real checkpoint

The activation tools now run against the real Qwen3.5-0.8B checkpoint on the pod
(RTX 4000 Ada, EUR-IS-1). Measured there:

- `capture` returns `(29, 1024)` in 0.6 s on a 29-token prompt. The model loads
  in ~52 s, so capture is cheap relative to a generation pass.
- Directions derived from four independent looping/control prompt pairs agree at
  **mean pairwise cosine +0.6367**. The contrast cosine per pair is weak
  (+0.50 to +0.95), but the agreement between pairs shows the shared component is
  real rather than prompt-specific noise. That is the property localisation needs.
- `_resolve_module` bridges the checkpoint key to the module path: Qwen3.5 saves
  layer weights as `model.language_model.layers.N...` while the loaded model
  exposes them at `model.layers.N...`. Capture resolves by matching the suffix
  from the layer index, so a wrong layer cannot silently resolve to another.

### explore9 — the first activation-driven run

explore9 is the first run where the activation pipeline actually executed. It
made 15 real tool calls and 12 forward passes, and every one of its claims checks
out against real tool results:

| call | real result |
|---|---|
| `capture_direction` ×4 | `contrast_cosine` +0.4596, +0.4799, +0.7169, +0.4921 |
| `capture_activations` | `(47, 1024)` |
| `aligned_columns` ×2 | 16 columns each |
| `apply_directional_edit` ×2 | 843,776 and 3,670,016 weights changed, both reloaded |
| `measure_loop` ×3 | `looped: true` all three times |

Two real directional edits were applied to layer 20's `down_proj` and reloaded
into vLLM. Neither stopped the loop: `text_chars` moved 601 → 601 → 624. The
editor concluded honestly that the layer-20 direction is at most weakly causal,
and named four concrete improvements — a length-matched math control, directions
from several layers compared by `contrast_cosine`, edits restricted to a
top-aligned column block, and a `measure_loop` after each edit.

This is the first run where the editor's proposed next steps are grounded in
measurements it actually made. The remaining gap is the contrast design, not the
tooling: a prose control against a math prompt conflates "looping" with
"reasoning vs prose", so the direction is diluted. That is a task-design fix.

### Still open

Whether a direction captured against a *length-matched math control* stops the
loop. explore9's two edits used a prose control and failed; the next run should
use a control that is itself a math problem the model completes without looping.

