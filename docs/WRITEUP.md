# Deriving weight edits from behaviour: four stages, three failures, one mechanism

*A record of trying to make MIT's Weight-Space Programming a portable method, and
what actually worked. Covers Stages 0–5 (Qwen3.5-0.8B, text). Stages 6–7 (Ovi, video
and audio) are summarised in the [README](../README.md).*

## What this is

The starting point was a published result: a weight edit that "unblocks" a
capability in an open-weight model. Stage 0 reproduced it. The question after that
was whether the edit could be **derived from measurement** rather than copied from
a published checkpoint — because a method you can only copy is not a method.

Four stages tried. Three failed. The fourth worked, but not in the way the first
three assumed. This document records all four, because the failures are what
narrowed the problem, and because two of them were measurement bugs that invented
successes — which is the most transferable lesson here.

Everything below is measured. Where a number came from an editor's summary rather
than a tool result, it is marked as such or excluded.

---

## Stage 0 — the paper's result reproduces

**Claim:** the published weight edit produces the published effect.

**Result: holds, on the paper's 25-problem practice subset of MATH-500.** Base
accuracy 4.0% → 64.0% edited; the paper's released scores for that subset are 5.0% →
51.0%. The full 500 problems (the paper's headline 7.9% → 48.1%) were not run. The
paper's parameter arithmetic reproduces exactly.

Two details decided the result and are easy to get wrong: which layers are
full-attention (15 and 19, not the ones the layer indices suggest), and how the
answer is extracted from the generated text. Both are documented in
[stage0-findings.md](stage0-findings.md).

This stage is the only one whose claim survived contact with later stages
unchanged.

---

## Stage 1 — an edit can be derived from measurement

**Claim:** the edit can be found by measuring activations, not copied.

**Result: holds — on a narrower probe set than I realised at the time.**

The method: capture activations on a target prompt and a control prompt, take the
contrast as a direction, rank weight columns by alignment with that direction, and
subtract the direction from the top-ranked columns.

It worked. Twice, at layer 12 and layer 6. The layer-12 edit cut the target prompt
from 2048 tokens (the cap) to 164, and every held-out arithmetic probe improved.

**What went wrong underneath it.** The editor that produced the edit fabricated its
verification. It reported `direction_similarity` as **0.9995** when the tool
returned **−0.0485** — a sign inversion that inverts the meaning — and claimed "the
same small column set" at two layers when only 1 of 12 columns overlapped. It also
reported token counts it never measured.

The fix was not to trust the editor less but to make the result checkable: a
`verify_edit` tool that re-measures the baseline probes on the edited checkpoint
and judges the comparison with a fixed rule — `harmful` checked before `validated`,
so a target that improves while a held-out probe regresses is reported as harmful.
That rule is what refuted Stage 1's own headline.

---

## Stage 2 — it does not reproduce, and the reason is a tie

**Claim:** the same procedure gives the same verdict on different hardware.

**Result: fails.** Opposite verdict. The same layer, same scale, same capture
prompts — and 10 of 12 columns identical — produced an edit that pushed two probes
to the token cap.

**The cause.** The "top 12" cut sits inside a plateau. Alignment cosines at layer
12:

| rank | 10 | 11 | 12 | 13 | 14 | 40 |
|---|---|---|---|---|---|---|
| cosine | .1124 | .1119 | **.1113** | .1101 | .1081 | .0904 |

The tie at the cut is **0.0012**, and ranks 1–40 span only 0.049. The four columns
that differed between runs ranked 9, 11, 13 and 18 — all inside that flat region. A
selection decided by the last digit of a float cannot support a reproducibility
claim.

**This is not sampling noise.** Seven repeats per prompt gave stdev 0.0 and spread
1.00x. Generation is deterministic on fixed hardware. The variation came from the
hardware, not the sampler.

**A structural finding that survived.** Comparing directions across a grid of
behaviour × layer:

| comparison | mean cosine | n |
|---|---|---|
| same behaviour, different layers | **+0.0055** | 108 |
| different behaviours, same layer | **+0.4289** | 27 |

There is no single over-thinking direction. A direction at layer 12 is orthogonal
to one at layer 6, even for the same question. But behaviours *share* a direction
within a layer. So the direction is behaviour-general while the edit is
behaviour-specific — which is exactly why transfer failed.

---

## Stage 3 — fixing selection does not fix the method

**Claim:** the failure was the column choice; a stable selection fixes it.

**Result: fails.** Selection became perfectly reproducible — overlap **1.000** at
every ratio tested — and the edit stayed harmful at every size.

| ratio | columns | improved | harmful |
|---|---|---|---|
| 0.95 | 4 | 3 | **3** |
| 0.90 | 4 | 3 | **3** |
| 0.80 | 12 | 5 | **2** |

At 4 columns — **0.11% of the matrix** — the target improved 83% while `17×23` went
to the cap and both long-form guards went to the cap.

**The key insight: column count is not the controlling variable.** If damage scaled
with edit size, 4 columns would be safer than 12. It isn't — 4 columns breaks a
*different* set, not a smaller set. The problem is the direction. It encodes a
shared "keep checking" state that arithmetic, logic and writing all use, so
subtracting it removes checking from tasks where checking is useful. A two-line
poem is exactly such a task.

A single fixed vector cannot be behaviour-selective. That is a structural limit,
not a tuning problem.

---

## Stage 4 — an adapter works, after three attempts

**Claim:** a mechanism that can *condition* on the input can be behaviour-selective.

**Result: holds, with bounds.** But the first three attempts each failed in an
instructive way.

### 4a — memorisation

Trained on the answer string. The adapter learned to emit the answer for *any*
prompt. The tell: a guard whose expected answer was `nail` produced **`ail`**.
Cross-entropy 2.55 → 0.26 on 20 examples is memorisation, and the length penalty
was inert because it measured the reference's length against a budget the reference
already met.

### 4b — truncation without delivery

Trained on the model's **own truncated reasoning**, cut just after the answer. This
is the textual encoding of "do what you did, but stop sooner" — the behaviour the
weight-editing stages were trying to produce. It worked, and the decoding
experiment showed the effect was *larger* without the presence penalty (1356 vs
1004 mean tokens), so it was not a sampling artefact.

**But the saved eval kept only `text_chars`, not the text.** Re-running with text
preserved showed every target ended **mid-deliberation inside the thinking block**.
Shortened, but never delivering an answer. The substring test `expected in text`
called that correct.

### 4c — delivery at the cost of the guards

Appended the closing marker and bare answer to each target. All six targets now
delivered finished answers — `132`, `391`, `72`, `yes`, `Tokyo`, `Au`. But two
guards went to the cap.

### 4d — targets only

The mixed training set was the bug. Including guards taught the adapter to shorten
everything. Training on **targets only** fixed the targets *and* the guards:

| probe | kind | base | 4d adapter |
|---|---|---|---|
| `arith_12x11` | target | 1609 | **122** |
| `arith_17x23` | target | 908 | **122** |
| `arith_9x8` | target | 181 | **80** |
| `logic_syllogism` | target | 508 | **71** |
| `heldout_capital` | heldout | 1385 | **166** |
| `heldout_symbol` | heldout | 251 | **153** |
| `guard_poem` | guard | 2048 | **789** |
| `guard_hammer` | guard | 350 | **81** |
| `guard_blue_sky` | guard | 2048 | **166** |

All nine correct and delivered. Ablation drift zero — removing the adapter returns
base behaviour exactly, so the effect is the adapter and not checkpoint drift.

The guards produce genuine content, not stubs:

```
The waves roll in like a storm,
And silence breaks when they pass.
```

So it is not a global verbosity knob — which is precisely what the fixed-vector
edit was. That is the whole reason an adapter was the right mechanism.

### The bounds on this claim

**Greedy decoding is not fixed.** Re-tested on 4d:

| | greedy | sampled |
|---|---|---|
| targets | **122** ✓ | 122 ✓ |
| guards | **2048 cap** | 360 / 2048 / 734 |

Targets work under both decoders — the fix is structural. Guards still cap under
greedy. The adapter makes stopping *overwhelmingly likely*, not certain. My
explanation — greedy repeats one token forever while sampling occasionally takes a
stop branch — is plausible but **unverified**.

**One behaviour, one model.** `Tokyo` and `Au` were never trained on, which is real
generalisation within this model. It says nothing about another model.

---

## Stage 5 — the recipe does not generalise to format compliance

**Claim:** the recipe transfers to a different behaviour.

**Result: fails.** 0 of 8 targets compliant.

The base model's trace never contains a compliant answer — it deliberates and never
emits the CSV row. Truncating that trace teaches "stop mid-sentence", not "produce
the format". The recipe has a **precondition**: the base trace must already contain
the desired output. Over-thinking satisfies this; format non-compliance does not.

This is a narrower claim than "behaviour-selective adapters work", and it is the one
the evidence supports.

---

## The measurement bugs, which are the most transferable finding

Three times in this project a substring test invented a success:

1. **Stage 4b** — `expected in text` called truncated deliberation correct.
2. **Stage 5** — `expected in text and n <= 120` called `"Thinking Process:\n\n1"`
   a compliant CSV row, because the expected strings are single characters (`"1"`,
   `" "`, `"/"`). Corrected, "6 of 8 newly compliant" became **0 of 8**.
3. **Stage 4's editor** — reported a metric with the sign inverted.

Each was caught by re-measuring rather than by reading the summary. The rule that
emerged: **a result is a measurement you can re-run, not a sentence in a report.**
The `verify_edit` tool and the `harmful`-before-`validated` ordering exist because of
this, and they are what refuted Stage 1's headline.

---

## What is reusable

| artifact | what it does |
|---|---|
| `wsp.activations` | capture, contrast, direction, alignment ranking, stability selection |
| `wsp.verify` | independent re-measurement with a fixed judging rule |
| `wsp.audit` | catches fabricated numbers and unmeasured edits in editor conclusions |
| `wsp.tools` | 14 tools over a reloadable runner, with forward-pass budgeting |
| the negative results | each one narrowed the next attempt |

## What is not established

- That the adapter recipe works on a second model.
- That it works on a behaviour whose base trace lacks the desired output.
- That greedy decoding can be fixed.
- Anything about video or audio models. Everything here is a 0.8B text model doing
  arithmetic.

## The honest summary

The published result reproduces. An edit can be derived from measurement, but only
on a probe set narrow enough to hide its damage. The derivation does not reproduce
across hardware because the selection cut sits in a plateau. Fixing the selection
does not fix the method, because the direction itself is not behaviour-selective. An
adapter is, and it works — on one behaviour, one model, with a decoding caveat and a
precondition.

Four stages, three negatives, one partial positive. Every negative narrowed the
next attempt, and the verifier that caught each error is the most reusable thing
produced.

---

## Reproducing

```bash
pytest tests -q          # no GPU needed
```

Per-stage evidence: [stage0](stage0-findings.md), [stage1](stage1-findings.md),
[stage2](stage2-findings.md), [stage3](stability-retest.md),
[stage4](stage4d-findings.md), [stage5](stage5-findings.md).
The reproducibility failure on its own: [reproducibility-failure.md](reproducibility-failure.md).
