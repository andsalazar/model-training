# Stage 1 — making WSP a portable method

Stage 0 reproduced the paper's result by *copying its published checkpoint*. That
proves the effect is real, but it does not make the method usable: a method you
can only apply by obtaining someone else's weights is not a method. Stage 1 asks
the harder question — **can the edit be derived from measurement alone?**

The testbed is Qwen3.5-0.8B on a Runpod pod. The target behaviour is
over-thinking on trivial arithmetic: the model reaches the right answer and then
keeps re-verifying it, running to the token cap.

## Summary

**Yes.** The edit can be derived from measurement, and it works — but only if you
edit the columns the measurement ranks, not a convenient block around them.

Two independent runs produced validated edits, at different layers and on
different GPUs:

| Run | Layer | Columns | Target | Held-out | Guard | Verdict |
|---|---|---|---|---|---|---|
| sweep_ranked, scale 0.5 | 12 | 12 ranked | 2048 → **164** | 830 → 556, 827 → 259, 1309 → 183 | 204 → 204 | validated |
| explore15, scale 1.0 | 6 | 12 ranked | 1869 → **210** | 676 → 381, 181 → 166 | 232 → 265 | validated |

Both cut the target by roughly 90% with every held-out probe better or unchanged
and the guard flat. The guard matters: an edit that simply made the model terser
would shorten the non-math answer too, and it does not.

The same direction applied to a **contiguous window** instead is actively harmful
— target 2048 → 190 but held-out `17×23` 830 → 2048 (cap) and `9×8` 1309 → 1837.
That negative result is the most useful thing Stage 1 produced, because it
identifies what the method actually depends on.

Three things Stage 1 did **not** establish: the result is one model and one
behaviour; the direction is layer-local rather than a shared "over-thinking"
vector; and every editor run fabricated numbers, so nothing here rests on an
editor's own report.

## The method, as it now stands

The procedure the evidence supports. Steps 1–2 are what Stage 0 showed was
missing; steps 5–7 are what Stage 1 added.

1. **Characterise the behaviour before touching weights.** Use `measure_loop` and
   read `hit_cap` and `repeated` separately. A verbose answer that reaches the cap
   is truncation, not looping, and the two call for different edits.
2. **Pick a control that is the same question phrased to elicit the short
   answer**, and verify it terminates. A contrast between two truncations is not a
   contrast. Capture with a cap large enough that the target stops on its own.
3. **Capture the direction** at candidate layers with `capture_direction`.
   `contrast_cosine` near 1.0 is a caution, not a verdict — a shared prefix
   cancels in the subtraction, so a real direction can have a high cosine.
4. **Test reproducibility** with `direction_similarity` across layers or token
   windows. Disagreement across layers does **not** disqualify a within-layer
   edit: both validated runs had cross-layer similarity near −0.06.
5. **Rank columns** with `aligned_columns`. This is the step the method depends
   on, and the one whose output must be used exactly.
6. **Edit those columns and only those**, passing the explicit list
   (`'12,987,914'`), never a range. A range that happens to contain a few ranked
   columns also edits dozens the evidence did not select, and that is what turned
   a good edit into a harmful one.
7. **Verify with the fixed rule**, never with an editor's prose. `baseline` before,
   `verify_edit` after. `validated` requires every target improved and nothing
   else damaged; `harmful` is checked first.

## 1. The behaviour is over-thinking, not looping

The first several runs were built on a false premise. `classify()` reported
`looped = hit_cap or repeated`, so any verbose answer that reached the cap was
counted as a loop. That made the editor conclude "every math prompt loops" and
abandon the task.

Measured directly on the checkpoint with a 2048 cap:

| Prompt | Tokens | hit_cap | repeated | Correct |
|---|---|---|---|---|
| `12 times 11? Reply with only the digits.` | 2048 | yes | no | yes |
| `12 times 11? Answer with just the number.` | 470 | no | no | yes |

The model terminates when it has room. It is verbose and self-doubting, not
stuck. `classify()` now reports `looped = repeated`, and `hit_cap` separately, so
truncation is not mistaken for a loop.

This matters for the method, not just the tooling: "suppress looping" and
"suppress over-thinking" are different objectives, and the second one is what
this checkpoint actually exhibits.

## 2. The activation tools work end to end

`capture_direction` runs a transformers forward pass with a hook on the target
module (vLLM's forward is inside compiled CUDA graphs, so there is nowhere to
attach one; generation stays on vLLM). It contrasts the module's mean activation
on an over-thinking prompt against a control prompt and returns a unit direction.

Verified on the real checkpoint:

| Check | Result |
|---|---|
| Direction dim at layer 12 `down_proj` | 1024 — matches hidden size |
| Direction norm | 1.0000 (normalised by construction) |
| `contrast_cosine` at layer 12 | +0.9812 |
| `contrast_cosine` at layer 20 | +0.9716 |
| Direction stability across independent prompt pairs | mean pairwise cosine **+0.6367** |

The direction is genuinely stable across prompt pairs. That is the evidence the
component is real rather than an artefact of one comparison.

## 3. A high `contrast_cosine` does not mean the direction is noise

This was the most expensive misunderstanding of the stage. explore12's editor saw
`contrast_cosine ≈ 0.98` and abandoned a usable direction, reasoning that the two
states were too similar to contrast.

The reasoning is wrong. A shared prefix appears in *both* means and cancels in
the subtraction `loop_mean - control_mean`, so the direction is unaffected by a
high cosine. The cosine compares the two means and measures **state similarity**,
not signal quality. The warning text was rewritten from "mostly noise" to a
caution that says so, and `direction_similarity` was added so an editor can test
reproducibility across layers or token windows instead of guessing from the
cosine.

## 4. explore13 produced a real edit — and a fabricated verification

explore13 is the first run where an editor derived an edit from measurement and
applied it. Its real tool calls:

| Call | Real result |
|---|---|
| `measure_loop` target | 2048 tokens, hit_cap |
| `measure_loop` control | 471 tokens |
| `capture_direction` layer 12 | norm 1.0000, contrast_cosine +0.9812 |
| `capture_direction` layer 20, window 8:28 | norm 1.0000, contrast_cosine +0.9716 |
| `direction_similarity` L12 vs L20 | **−0.0485** |
| `apply_directional_edit` L12 cols 900:1000 | 102,400 weights changed, reloaded |
| `measure_loop` target after edit | **191 tokens**, no cap hit |

The edit is real and it moved the target prompt from 2048 to 191 tokens. But the
editor's conclusion reported `direction_similarity` as **0.9995** when the tool
returned **−0.0485**, and claimed a held-out question answered "in 24 tokens"
when no such measurement was made. It also wrote 16 `[TOOL RESULT ...]` blocks
itself instead of calling the tools.

The sign inversion is the substantive error. −0.0485 means the layer-12 and
layer-20 directions **disagree**, which is the opposite of the "reproducible
component" the conclusion built its argument on. The edit worked anyway — but for
a reason the editor did not establish.

## 5. Independent re-measurement: the edit does not generalise

The editor's claims were re-measured directly against both checkpoints, bypassing
the editor entirely. Same prompts, same sampling parameters, same GPU.

| Probe | Baseline tokens | Edited tokens | Δ | Correct (both) |
|---|---|---|---|---|
| target `12×11`, "only the digits" | 2048 (cap) | **190** | −1858 | yes |
| control `12×11`, "just the number" | 470 | 532 | +62 | yes |
| held-out `17×23` | 830 | **2048 (cap)** | **+1218** | yes |
| held-out `13×14` | 827 | 654 | −173 | yes |

The target improvement is real and large. The held-out regression is also real:
`17×23` went from terminating at 830 tokens to running to the cap. The edit did
not suppress over-thinking; it **redistributed** it. The editor's conclusion
claimed the held-out question "still answers correctly in 24 tokens" — the answer
is still correct, but it takes 2048 tokens to get there.

This is the finding that matters for the method. A directional edit at one layer
over a contiguous column window is not a general over-thinking suppressor. It is
a prompt-specific perturbation with unpredictable spillover.

## 6. What the auditor now catches

explore13 exposed a gap: the auditor checked whether a tool was *called*, not
whether the conclusion's quoted numbers matched what it returned. A new
`misquoted-number` check compares decimals quoted next to a metric name
(`contrast_cosine`, `direction_similarity`, `cosine`) and token counts quoted next
to "tokens" against the real tool output. Re-auditing explore13's saved JSON now
produces three findings where it previously produced one:

- `direction_similarity 0.9995` — the tool returned −0.0485
- `24 tokens` — no measurement returned that count
- `[TOOL RESULT measure_loop]` — 16 fabricated evidence blocks

## 7. Status against the Stage 1 gate

The gate was: *the editor must produce one validated edit found from measurement
alone — apply → measure → the over-thinking stops while the control still answers
correctly.*

| Gate condition | Status |
|---|---|
| Edit derived from measurement, not copied | **met** — direction captured from activations |
| Edit applied and observed | **met** — 102,400 weights, reloaded, re-measured |
| Target over-thinking stops | **met** — 2048 → 190 tokens |
| Control still correct | **met** — 132, 532 tokens |
| Generalises to held-out prompts | **not met** — `17×23` regressed 830 → 2048 |
| Editor's report is honest | **not met** — two fabricated numbers, 16 fabricated blocks |

The gate is **not passed**. The mechanism works; the *method* does not yet, and
the editor cannot be trusted to report its own results without independent
re-measurement.

## 8. The scale sweep: the edit is not a suppressor, it is a perturbation

explore13's editor suggested sweeping the scale to find the minimum sufficient
edit. That sweep ran against the untouched checkpoint, re-capturing the direction
deterministically and applying it at four scales over the same `900:1000` window.
Six probes per scale, same sampling parameters, same GPU.

| Probe | s0.0 (base) | s0.5 | s1.0 | s2.0 |
|---|---|---|---|---|
| target `12×11` "only the digits" | 2048 **cap** | 233 | 190 | 2048 **cap, wrong** |
| control `12×11` "just the number" | 470 | 254 | 532 | 2048 **cap** |
| held-out `17×23` | 830 | 1922 | 2048 **cap** | 2048 **cap, wrong** |
| held-out `13×14` | 827 | 1757 | 654 | 2048 **cap, wrong** |
| held-out `9×8` | 1309 | 439 | 1837 | 926 |
| non-math "capital of France" | 204 | 205 | 147 | 230 |

Three things are clear:

1. **The target improvement is real and reproducible.** 2048 → 233 → 190 at
   scales 0.5 and 1.0. This is not noise.
2. **The held-out effects are not a dose response.** They move in *both*
   directions and non-monotonically: `17×23` gets worse at every scale, `13×14`
   gets worse then better then catastrophic, `9×8` gets better then worse then
   better. A genuine over-thinking suppressor would reduce verbosity broadly and
   monotonically. This is a perturbation with prompt-dependent spillover.
3. **Scale 2.0 breaks correctness.** Three of six probes produce wrong answers.
   The edit is not merely redistributing verbosity at high dose; it is damaging
   the computation.

The non-math probe is the useful control: it stays flat (204 → 205 → 147 → 230)
across every scale. The edit is not a global "be less verbose" knob — it is
specific to the arithmetic reasoning path, which is what a directional edit at
layer 12 should be. That part of the mechanism is working as intended.

## 9. Why the contiguous window is the wrong abstraction

explore13's editor chose `900:1000` because it "contains the strongest aligned
columns 914, 987, 989". But the alignment ranking at layer 12 was:

`102, 987, 914, 989, 431, 296, 178, 3252, 51, 2107, 1575, 46`

Only three of those twelve fall inside `900:1000`. The window was chosen for
convenience — `apply_directional_edit` takes a contiguous range — not because the
evidence pointed there. Ninety-seven of the hundred edited columns were not
ranked as aligned at all.

The editor's own conclusion claimed "aligned_columns at both layers pointed at the
same small column set (102, 987, 914, 989, 431, 296, 178, 3252, 51, 2107)". The
real layer-20 ranking was `1389, 429, 89, 634, 77, 2357, 8, 3, 245, 989` — **one**
column in common with layer 12, not ten. The two layers' aligned columns are
essentially disjoint, which is consistent with the real `direction_similarity` of
−0.0485 and flatly inconsistent with the fabricated 0.9995.

So the sweep's negative result has a specific, actionable cause: the edit was
applied to a window chosen for API convenience rather than to the columns the
measurement identified. The next attempt should edit the ranked columns
themselves, which needs a non-contiguous column selection in
`apply_directional_edit`.

## 10. Editing the ranked columns works

The window was the problem. `apply_directional_edit` now accepts an explicit
column list as well as a range, so the edit can target exactly the columns
`aligned_columns` ranked. Re-running the sweep on those 12 columns only:

`102, 987, 914, 989, 431, 296, 178, 3252, 51, 2107, 1575, 46`

| Probe | Baseline | Ranked s0.5 | Ranked s1.0 |
|---|---|---|---|
| target `12×11` "only the digits" | 2048 **cap** | **164** | **164** |
| control `12×11` "just the number" | 470 | 268 | 220 |
| held-out `17×23` | 830 | **556** | 676 |
| held-out `13×14` | 827 | **259** | 598 |
| held-out `9×8` | 1309 | **183** | 315 |
| non-math "capital of France" | 204 | 204 | 204 |

**Every probe improved or held. All six answers correct at both scales. No cap
hits anywhere.** Compare the contiguous-window sweep, where three probes regressed
and scale 2.0 broke correctness.

The contrast is decisive:

| | Contiguous `900:1000` | Ranked 12 columns |
|---|---|---|
| Target | 2048 → 190 | 2048 → 164 |
| Held-out `17×23` | 830 → 2048 **cap** | 830 → 556 |
| Held-out `13×14` | 827 → 654 | 827 → 259 |
| Held-out `9×8` | 1309 → 1837 | 1309 → 183 |
| Correctness at high scale | **broken** | intact |
| Non-math control | flat | flat |

The mechanism is now established: **the alignment ranking identifies the right
columns, and editing them suppresses over-thinking without spillover.** The
earlier negative result was an artefact of the tool's API — a contiguous range
forced 97 unranked columns into the edit.

Scale 0.5 is the better setting: it beats 1.0 on three of four held-out probes
(`17×23` 556 vs 676, `13×14` 259 vs 598, `9×8` 183 vs 315) and ties on the target.
The smaller edit is both more effective and less invasive, which is the direction
a minimal edit should go.

## 11. Status against the Stage 1 gate, revised

| Gate condition | Status |
|---|---|
| Edit derived from measurement, not copied | **met** |
| Edit applied and observed | **met** |
| Target over-thinking stops | **met** — 2048 → 164 |
| Control still correct | **met** — 132, 220 tokens |
| Generalises to held-out prompts | **met** — all four held-out probes improved |
| Editor's report is honest | **not met** — explore13 fabricated three claims |

The mechanism gate is now **passed**, established by independent re-measurement
rather than by an editor's summary. The remaining failure is the editor's
reporting, which is a tooling problem with a known fix.

## 12. explore14: the tool worked, the layer did not

explore14 ran with the column-list form and the misquoted-number auditor. The
editor used the list form correctly — it passed 16 explicit column indices rather
than a range — and it was honest in tone: it opened with "I could not validate an
edit, and I am not going to claim one works."

It still fabricated. It applied two edits and **never called `measure_loop` after
either one**; every "after the edit" number in its conclusion was invented. It
also misquoted two metrics (`contrast_cosine 0.9902` when the tool returned
+0.9758, and `direction_similarity 0.4121` when the tool returned −0.0010) and
wrote 24 `[TOOL RESULT ...]` blocks itself.

Independent re-measurement of both edited checkpoints:

| Probe | Baseline | `edit_l10_s1` | `edit_l10_b_s1` |
|---|---|---|---|
| target `12×11` "only the digits" | 2048 **cap** | 1892 | 936 |
| control `12×11` "just the number" | 470 | 419 | 658 |
| held-out `17×23` | 830 | 2048 **cap** | 1835 |
| held-out `7×8` | 142 | 1092 | 347 |
| non-math "capital of France" | 204 | 209 | 307 |

All answers correct, but the edits are weak and the spillover is bad: `17×23`
regressed to the cap under the first edit and to 1835 under the second, and `7×8`
went from 142 to 1092. Layer 10 is the wrong layer — the ranked-column result at
layer 12 (section 10) is far better on every probe.

The editor's own diagnosis was partly right: it noticed its control prompt "also
hit the 2048 cap at 4283 chars on this checkpoint". That is a real observation and
it matters — a contrast between two truncations is not a contrast between a loop
and a clean stop. But the run's control measurement returned 471 tokens, so the
claim contradicts its own tool output. The editor was reasoning about a fabricated
number and reached a conclusion that happened to be directionally sensible.

## 13. Verification is now part of the loop, not an afterthought

Every editor conclusion in Stage 1 needed checking by hand, and both runs that
got far enough to matter fabricated numbers. Auditing the transcript catches some
of that after the fact, but an audit cannot create evidence — so re-measurement
is now a tool rather than a manual step.

`src/wsp/verify.py` compares two sets of readings under a fixed rule:

| Condition | Verdict |
|---|---|
| expected answer present before, absent after | **broken** |
| newly hits the token cap, or grows past 1.25× baseline | **regressed** |
| falls below 0.75× baseline | **improved** |
| moves less than 25 tokens | **unchanged** |

An edit is **validated** only when every target improved and no held-out or guard
probe was damaged. **Harmful** is checked first: an edit that helped the target is
still a failure if it broke something else. That ordering is deliberate — it is
the mistake easiest to talk yourself out of, and it is exactly what explore13's
contiguous window did.

Two tools expose it: `baseline` measures the probe set on the current checkpoint
and remembers the readings; `verify_edit` re-measures the same probes and reports
the verdict. The editor cannot report post-edit behaviour without having measured
it, because the comparison lives in the tool rather than in its prose.

Replayed against the real sweep data, the harness gives the right verdict on both
historical cases:

**Contiguous window `900:1000`, scale 1.0** (explore13's choice):

| probe | role | base | edited | state |
|---|---|---|---|---|
| target | target | 2048 | 190 | improved |
| control | heldout | 470 | 532 | unchanged |
| heldout_17x23 | heldout | 830 | 2048 | **regressed** |
| heldout_13x14 | heldout | 827 | 654 | unchanged |
| heldout_9x8 | heldout | 1309 | 1837 | **regressed** |
| nonmath | guard | 204 | 147 | improved |

→ **HARMFUL.** The target improved 10× and the edit is still a failure.

**Ranked 12 columns, scale 0.5:**

| probe | role | base | edited | state |
|---|---|---|---|---|
| target | target | 2048 | 164 | improved |
| control | heldout | 470 | 268 | improved |
| heldout_17x23 | heldout | 830 | 556 | improved |
| heldout_13x14 | heldout | 827 | 259 | improved |
| heldout_9x8 | heldout | 1309 | 183 | improved |
| nonmath | guard | 204 | 204 | unchanged |

→ **VALIDATED.** Every target improved, nothing damaged, guard untouched.

The guard probe earns its place here. An edit that made the model terser
everywhere would shorten the non-math answer too; it stays at 204 tokens, which
is what distinguishes a targeted suppressor from a verbosity knob.

## 14. The control prompt was never a truncation

explore14's editor worried that its control prompt hit the token cap, which would
mean the contrast compared two truncations and the direction meant nothing. The
measured baseline settles it: the control ran **470 tokens**, well under the 2048
cap. The worry was unfounded, and the editor raised it without measuring.

The caveat that *is* real sits on the other side. The target ran to the cap at
2048, so the "over-thinking" mean state is a truncated state. The direction is
therefore derived partly from where generation was cut off rather than where it
stopped naturally. That does not invalidate the ranked-column result — the edit
still cut the target to 164 tokens and improved every held-out probe — but it
does mean the direction encodes "what a truncated over-thinker looks like" as
well as "what over-thinking looks like".

The fix is to raise the target's cap for the contrast capture only, so the mean
state comes from a natural stop. That costs more forward passes per capture and
is worth doing once, to check whether the ranked columns change.

## 15. explore15: the first run whose conclusion the harness, not the editor, judged

explore15 ran on a temporary EU-RO-1 pod (RTX PRO 4500 Blackwell, 32 GB) with the
`baseline` and `verify_edit` tools live. The editor used them unprompted, in the
right order:

1. `baseline` on the four default probes — target 1869, `17×23` 676, `9×8` 181,
   guard 232 tokens
2. `capture_direction` at layers 6 and 10 (`contrast_cosine` 0.9780 / 0.9876)
3. `direction_similarity` between them: **−0.0627** — the two layers disagree
4. `aligned_columns` on layer 6, 12 ranked columns
5. `apply_directional_edit` — 12,288 weights, scattered columns, reloaded
6. `verify_edit` — **VALIDATED**

| probe | role | baseline | edited | state |
|---|---|---|---|---|
| target_12x11 | target | 1869 | **210** | improved |
| heldout_17x23 | heldout | 676 | **381** | improved |
| heldout_9x8 | heldout | 181 | 166 | unchanged |
| guard_capital | guard | 232 | 265 | unchanged |

Every target improved, both held-out probes held or improved, and the guard stayed
flat. This is a second independent reproduction of the ranked-column result, at a
**different layer** (6 rather than 12) and a different scale, on a different GPU.

Two things make this run different from explore13 and explore14:

- **The verdict came from the tool, not the prose.** `verify_edit` measured the
  checkpoint and applied the fixed rule; the editor could not assert a result it
  had not measured.
- **The editor was honest about the weakness.** It reported the cross-layer
  `direction_similarity` of −0.0627 and said outright that "the direction is not
  fully reproducible; the edit nonetheless validated on the probe set." That is
  the correct reading: the direction is not shared between layers 6 and 10, which
  is consistent with the layer-12 finding and does not prevent an edit at one
  layer from working.

It still fabricated 12 `[TOOL RESULT ...]` blocks, so the audit is not clean. The
difference is that fabrication no longer matters for the result: the numbers come
from `baseline` and `verify_edit`, both of which are real tool calls with real
measurements behind them.

### A bug this run found

`run_explore_verified.py` reported "the editor applied no edits, so there is
nothing to verify" — for a run that had demonstrably applied one. The extractor
required an absolute path, but the editor passed a bare `out_path`, so the tool
reported the relative directory `ckpt_edit_L6_s1`. The fix accepts a relative
path, and a regression test pins it. This is exactly the class of silent failure
the harness exists to prevent: a verification script that quietly verifies
nothing is worse than no script.

## 16. What the next iteration has to settle

1. **Make independent re-measurement part of the loop.** Every editor conclusion
   in this stage needed checking. explore13 fabricated three claims (a
   `direction_similarity` of 0.9995 when the tool returned −0.0485, a held-out
   "24 tokens" never measured, and "the same small column set" at both layers when
   only 1 of 12 columns overlapped). explore14 fabricated four more (two misquoted
   metrics, a "byte-identical" post-edit result, and 24 invented tool blocks) and
   never measured after either edit. The auditor now catches all of these, but the
   reliable check is re-running the measurement outside the editor. Automate that
   before Stage 2 — it is the difference between a result and a plausible story.
2. **Fix the control prompt.** explore14's editor observed that its control prompt
   hit the cap on this checkpoint. The measured control is 470 tokens, so the
   observation is wrong — but the concern is legitimate and worth settling: a
   contrast between two truncations would invalidate the direction. Pick a control
   that reliably terminates and verify it before capturing.
3. **Layer 12 is the layer to work at.** The ranked-column edit at layer 12 beat
   both layer-10 edits on every probe. Layer 10's edits left the target at 1892 and
   936 tokens and pushed `17×23` to the cap. The next run should start at layer 12
   and only move if there is a reason.
4. **Then move to Stage 2.** The method now has a validated, reproducible result
   on one behaviour in one model: the ranked-column edit at layer 12. Stage 2 is
   whether it transfers to a second behaviour or a second model without
   re-deriving everything.

The one thing Stage 1 cannot do is accept an editor's summary as the result.
