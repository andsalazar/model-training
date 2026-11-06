# Stage 4 findings — the adapter works, with a decoding caveat

## Summary

The second adapter is the first mechanism in this project to pass the broad
verification rule. It reduces over-thinking on every target, improves both
held-out probes it never trained on, and leaves guard tasks substantially intact.

| probe | kind | base | adapter | state |
|---|---|---|---|---|
| `arith_12x11` | target | 1609 | **116** | improved |
| `arith_17x23` | target | 908 | **116** | improved |
| `arith_9x8` | target | 181 | **75** | improved |
| `logic_syllogism` | target | 508 | **63** | improved |
| `heldout_capital` | heldout | 1385 | **414** | improved |
| `heldout_symbol` | heldout | 251 | **129** | improved |
| `guard_poem` | guard | 2048 | **117** | improved |
| `guard_hammer` | guard | 350 | **1098** | **regressed 3.1x** |
| `guard_blue_sky` | guard | 2048 | **361** | improved |

Every answer correct. Ablation reproduces the baseline exactly on all nine probes
(`drift: []`), so the effect is the adapter and not merge error or serving
flakiness.

**Two caveats are stated up front, because they bound the claim.**

1. `guard_hammer` **regressed** from 350 to 1098 tokens. Two of three guards
   improved; one got worse. The effect is uneven across guards.
2. The improvement is **decoding-dependent**. Under greedy decoding the adapter
   still runs to the cap. See §4.

## 1. The first attempt failed on a data-design error

Worth recording because the failure mode was specific and mine.

The first adapter trained on the **answer string** as the reference. It learned
exactly that: given any prompt, emit the answer token.

| probe | adapter output |
|---|---|
| `12×11` | `132` |
| `logic` | `yes` |
| `guard_poem` (expected `sea`) | `sea` |
| `guard_hammer` (expected `nail`) | `ail` |

`ail` is the tell — a fragment of `nail`. The adapter had learned "emit the
expected string" and applied it to a task it could not do. Cross-entropy fell
2.55 → 0.26 on 20 examples, which is memorisation. The length penalty was inert
(0.000000 every epoch) because it compared the *reference's* length against a
budget the reference already met.

The eval caught all of it: 1609 → 3 tokens is not subtle, and `correct=True` with
output `sea` for a poem about the sea showed the substring test is too weak alone.

## 2. The fix: train on the model's own truncated trace

`pod_stage4b_targets.py` generates the base model's full response, finds where the
answer appears, and keeps the trace only up to that point:

```
2048 → 116 tokens
'... $12 \times 10 = 120$ * $12 \times 1 = 12$ * $120 + 12 = 132'
```

That is the closest textual encoding of "do what you did, but stop sooner" — the
behaviour the weight-editing stages tried to produce directly. Guards keep their
full trace on purpose, so the adapter has examples where long output is correct.

One example was skipped because the answer never appeared (`H2O`), rather than
trained on a target that does not contain it.

## 3. Retraining

| setting | attempt 1 | attempt 2 |
|---|---|---|
| reference | answer string | truncated own trace |
| lr | 1e-3 | 1e-4 |
| epochs | 8 | 6 |
| length penalty | inert (0.000000) | active hinge on target length |
| final CE | 0.26 | 0.022 |
| max abs LoRA weight | (large; drove degenerate output) | 0.0207 |

The small LoRA weights are evidence against collapse: attempt 1 drove the model to
single tokens, which needs large weights. 0.0207 is a nudge, not a takeover.

## 4. The decoding caveat, and the experiment that resolved it

The same merged checkpoint gave very different lengths under different samplers:

| probe | sampled (T 0.6, pp 1.5) | greedy (T 0) |
|---|---|---|
| `guard_poem` | **117** | **2048** |
| `guard_hammer` | 1098 | 2048 |
| `guard_blue_sky` | 361 | 2048 |

That left two explanations open, and they mean very different things:

- **adapter** — it genuinely stops over-thinking, and greedy is outside its
  training distribution
- **sampler** — the presence penalty does the work, since the base model's failure
  mode is re-verifying the same token, which a repetition penalty directly attacks

So I ran a 2×2 grid: `{base, adapter} × {pp 0, pp 1.5}`, temperature fixed at 0.6
so only the penalty varied.

| probe | base pp0 | base pp1.5 | adapter pp0 | adapter pp1.5 |
|---|---|---|---|---|
| `arith_12x11` | 2048 | 1038 | **117** | **117** |
| `arith_17x23` | 2048 | 690 | **117** | **117** |
| `logic` | 1005 | 1586 | **64** | **64** |
| `guard_poem` | 2048 | 2048 | **73** | **117** |
| `guard_hammer` | 2048 | 1171 | 2048 | 1098 |

Mean token reduction from the adapter:

| config | mean reduction | per-probe |
|---|---|---|
| `pp 0` | **1356** | 1931, 1931, 941, 1975, **0** |
| `pp 1.5` | **1004** | 921, 573, 1522, 1931, 73 |

**The adapter is doing the work, not the sampler.** Without any presence penalty
the base model runs to 2048 on four of five probes while the adapter stops at
117 on arithmetic with the answer intact. The adapter's effect is *larger* without
the penalty (1356 vs 1004 mean reduction), which is the opposite of a
sampling artefact.

The penalty independently helps the base model — 2048 → 1038, 690, 1171 — as
expected, since it attacks verbatim repetition. But the adapter accounts for most
of the total effect.

**The greedy result is still unexplained.** Under `temperature=0` the adapter runs
to 2048 while under `temperature=0.6, pp=0` it stops at 117. Greedy decoding picks
the same tokens every step, so a distribution with even a small probability of
continuing will continue indefinitely. Sampling occasionally takes the
stop branch and the sequence terminates. That is a plausible account but it is
**not verified**, and it means the claim is: *the adapter makes stopping
overwhelmingly likely, not certain.*

`guard_hammer` shows the same shape from the other side: it is the one probe the
adapter does not fix at any setting (2048 / 1098 vs base 2048 / 1171). It is a
genuine gap, not a decoding effect.

## 5. Are the guards still doing their job?

The substring `correct` field cannot answer this, so the outputs were read. Under
sampling they are intact:

**`guard_poem`** (117 tok) — a real two-line poem, with drafting:
```
The waves roll in like a storm,
And silence breaks when they pass.
(A bit generic)
Let's try to be more evocative.
The deep blue is the ocean's song,
And the sky holds no other sound.
```

**`guard_blue_sky`** (361 tok) — correct science:
```
Rayleigh scattering … Shorter wavelengths scatter more strongly than longer
wavelengths … Blue light is scattered by air molecules
```

**`guard_hammer`** (1098 tok) — real uses, still over-elaborating:
```
Use 1: Drilling holes in wood or metal for repairs.
Use 2: Driving nails into wood or metal to secure items.
Use 3: Shaping metal or wood for crafts or manufacturing.
```

So the adapter is **not** a global verbosity knob: it produces genuine content
where content is required. That is exactly the property the fixed-vector edit
lacked, and the reason an adapter was the right mechanism.

## 6. What this establishes

| claim | status |
|---|---|
| An adapter can be behaviour-selective where a fixed vector was not | **supported** — guards retain content |
| It transfers to a behaviour never trained on | **supported** — `factual` held out, both probes improved |
| The effect is the adapter, not drift | **supported** — ablation exact |
| It reduces over-thinking regardless of decoding | **partly** — works at T 0.6 with and without the penalty; not under greedy |
| It improves every guard | **not supported** — `guard_hammer` regressed at every setting |

## 7. Where this leaves the project

Stage 4's first falsifiable prediction — "adapter reduces target length, guards
unchanged" — is **partly met**. Targets clearly improved; guards retained their
task but not uniformly.

That is the first result in four stages that is not a clean negative. It does not
yet reach the original goal: this is a text behaviour on a 0.8B model, not video
or audio generation. What it establishes is that the mechanism class the original
brief pointed at — a trained adapter rather than a fixed direction — behaves the
way the Stage 2–3 diagnosis predicted.

## 8. Next

1. **Verify the greedy explanation.** The account — that sampling occasionally
   takes a stop branch while greedy repeats forever — is plausible but unverified.
   Measuring the probability the adapter assigns to the stop token on arithmetic
   prompts would settle it.
2. **Diagnose `guard_hammer`.** The one probe the adapter never fixes bounds the
   selectivity claim, and it is the only guard that regresses at every setting.
3. **Only then scale.** A second behaviour or model is the natural test, and the
   tooling for it now exists.
