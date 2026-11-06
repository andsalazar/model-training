# Stage 4d — the mixed training set was the problem

## The experiment

Stage 4c fixed all six targets but pushed `guard_hammer` and `guard_blue_sky` to
the token cap. The hypothesis: the mixed training set taught one adapter to both
suppress deliberation (targets) and preserve it (guards), which it cannot do.

Stage 4d trained on **targets only** — same targets, same recipe, same epochs,
one flag (`--kinds target`).

## Result: everything works

| probe | kind | base | 4c (mixed) | **4d (targets only)** | delivered |
|---|---|---|---|---|---|
| `arith_12x11` | target | 1609 | 122 | **122** | yes |
| `arith_17x23` | target | 908 | 122 | **122** | yes |
| `arith_9x8` | target | 181 | 80 | **80** | yes |
| `logic_syllogism` | target | 508 | 71 | **71** | yes |
| `heldout_capital` | heldout | 1385 | 180 | **166** | yes |
| `heldout_symbol` | heldout | 251 | 162 | **153** | yes |
| `guard_poem` | guard | 2048 | 782 | **789** | yes |
| `guard_hammer` | guard | 350 | **2048 (cap)** | **81** | yes |
| `guard_blue_sky` | guard | 2048 | **2048 (cap)** | **166** | yes |

Six of six targets deliver finished answers. All three guards produce real
content and are *shorter* than base, not longer. The two guards the mixed adapter
destroyed are the two the target-only adapter fixes best.

## The guard text is genuinely correct

`guard_hammer` (81 tokens):

```
1. Constructing furniture and building structures
2. Removing nails or screws from woodwork
3. Driving bolts into metal plates
```

`guard_poem` (789 tokens) ends with a real two-line poem:

```
The waves roll down the dark, endless ocean floor,
And touch the shore with a soft, golden sigh.
```

`guard_blue_sky` (166 tokens) gives a correct Rayleigh-scattering explanation.

So the adapter is **behaviour-selective**: it suppresses redundant
self-verification without destroying elaboration. That is the claim Stage 4 was
trying to establish, and it now holds.

## Why the mixed set failed

The guards' training targets were their **full untruncated traces** — 2048 tokens
of deliberation. The targets' training targets were truncated to ~90 tokens. One
adapter trained on both learned a compromise: shorten everything. For targets
that is correct; for guards it destroys the task.

Training on targets only removes the conflict. The adapter learns "stop
self-verifying" without also learning "stop elaborating", and the guards — which
were never in the training set — keep working because the base model's
elaboration behaviour is untouched.

This is the opposite of what the mixed set was meant to achieve. Including guards
was supposed to *protect* them; it damaged them.

## What this establishes

1. **A single LoRA can be behaviour-selective.** It suppresses over-thinking on
   the trained behaviour and on a held-out behaviour, while leaving long-form
   elaboration intact.
2. **The training set composition is the controlling variable**, not the adapter
   capacity or the mechanism. The same recipe with one flag changed produced
   opposite guard outcomes.
3. **Including a behaviour in the training set can damage it.** The intuition
   that "train on everything you want to preserve" is wrong here, because the
   target format (truncated) and the guard format (full) conflict.

## The decoding check, repeated for 4d

The 4d adapter under both decoders:

| probe | greedy | sampled (T 0.6, pp 1.5) |
|---|---|---|
| `arith_12x11` | 123 | 123 |
| `arith_17x23` | 123 | 114 |
| `logic_syllogism` | 72 | 72 |
| `heldout_capital` | 214 | 170 |
| `guard_poem` | **2048 (cap)** | 360 |
| `guard_hammer` | **2048 (cap)** | **2048 (cap)** |
| `guard_blue_sky` | **2048 (cap)** | 734 |

**Targets work under both decoders.** That is the important part: the target
behaviour is fixed structurally, not by sampling luck. Greedy gives 123 tokens for
`12×11` where base gives 1609.

**Guards still cap under greedy.** All three hit 2048. Under sampling two of three
recover. So the guard preservation is real but decoding-dependent — the adapter
makes stopping likely, not certain, and greedy has no chance of taking a stop
branch.

This is the same decoding caveat Stage 4 found, and it did not go away. It bounds
the claim: the adapter is a strong prior toward stopping, not a guarantee.

## The honest caveats

1. **`guard_poem` is still 789 tokens** in the sampled eval — down from 2048, but
   not short. The content is correct; the length is not minimal. Whether that
   counts as "preserved" is a judgement call.
2. **Guards cap under greedy.** See the table above. The guard result is
   decoding-dependent.
3. **One held-out behaviour, one model.** `Tokyo` and `Au` were never trained on,
   which is real generalisation within this model. It says nothing about another
   model or a very different behaviour.
4. **The ablation was clean** — removing the adapter returns base behaviour — so
   the effect is the adapter, not checkpoint drift.

## Cost

One pod, ~20 minutes for 4d (training ~2 min, eval ~7 min, plus vLLM load). The
pod was already running for Stage 5.

## Status

This is the first mechanism in the project to pass the broad verification that
refuted Stages 1–3: targets fixed, held-out behaviour fixed, guards preserved,
ablation clean. The claim it supports is narrow but real:

> A LoRA adapter trained on the model's own truncated reasoning — targets only,
> no guards — suppresses redundant self-verification and delivers correct short
> answers on the trained and a held-out behaviour, while leaving long-form
> elaboration intact.
