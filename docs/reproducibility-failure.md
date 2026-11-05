# A weight-space edit that worked, and then didn't

*A reproducibility failure in measurement-derived weight editing, and what it
teaches about the method.*

## What this is about

[Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(WSP) showed that a small, targeted edit to a language model's MLP weights can
unblock a capability — in their case, editing 64 columns across two layers of
Qwen to turn a 5% MATH-500 score into 51%.

We reproduced that result (Stage 0: 4.0% base, 64.0% edited, matching their
arithmetic exactly). Then we asked the question the paper leaves open: **can the
edit be derived from measurement, rather than copied from a published
checkpoint?**

We found an edit that worked. Two independent runs validated it. Then we ran the
same procedure again on different hardware and got the opposite answer. This is
the account of why, because the reason is more useful than the original result.

## The method we built

The target behaviour was over-thinking: on Qwen3.5-0.8B, asking for `12 × 11` in
a way that invites verification produces 2048 tokens of self-doubt around an
answer it already had.

The procedure:

1. `measure_loop` the behaviour, reading `hit_cap` and `repeated` separately.
2. Capture a direction by contrasting the module's mean activation on a verbose
   phrasing against a terse one, at the same question.
3. Rank the weight columns by alignment with that direction.
4. Subtract the direction from the top-ranked columns.
5. Verify with a fixed rule over a probe set.

It worked, twice:

| run | layer | target | held-out | guard | verdict |
|---|---|---|---|---|---|
| A | 12 | 2048 → **164** | 830→556, 827→259, 1309→183 | 204 → 204 | validated |
| B (layer 6) | 6 | 1869 → **210** | 676→381, 181→166 | 232 → 265 | validated |

Both cut the target by ~90%, with held-out probes improving and an unrelated
guard unchanged. That guard mattered: an edit that simply made the model terser
would shorten the non-math answer too, and it didn't.

So we had a method. We wrote it up. Then we tried to extend it.

## The failure

Extending to a second behaviour needs no new machinery, so we ran the same
operating point — layer 12, top 12 columns, scale 0.5, same capture prompts — and
measured five probes.

| probe | role | baseline | edited | state |
|---|---|---|---|---|
| `12×11` | the behaviour captured | 1869 | **642** | improved |
| `17×23` | held-out | 676 | **278** | improved |
| logic (syllogism) | other behaviour | 621 | 492 | unchanged |
| fact (nearest planet) | other behaviour | 482 | **2048** | **regressed — hit cap** |
| write uses of a hammer | unrelated | 1204 | **2048** | **regressed — hit cap** |

Ten of the twelve edited columns were identical to the earlier validated run. The
verdict inverted anyway, and two probes unrelated to the capture were pushed to
the token limit.

Re-measuring the *original* probe set on one identical checkpoint rejected the
original result too: two of its four arithmetic held-outs regressed
(`13×14` 259 → 1167, `9×8` 181 → 302).

First hypothesis: sampling noise. Generation used `temperature=0.6`, so maybe a
single sample per probe is high-variance.

**It isn't.** Seven repeats per prompt, five prompts, on fixed hardware:

```
target_12x11   min=1869 max=1869 stdev=0.0 spread=1.00x
13x14          min= 259 max= 259 stdev=0.0 spread=1.00x
9x8            min= 181 max= 181 stdev=0.0 spread=1.00x
17x23          min= 676 max= 676 stdev=0.0 spread=1.00x
guard_capital  min= 232 max= 232 stdev=0.0 spread=1.00x
```

Perfectly deterministic. Every difference above is a real difference between
checkpoints, not run-to-run jitter. So the method itself was the variable.

## The cause: the top-k cut is a plateau

The alignment ranking has no meaningful twelfth column. Cosines at layer 12:

| rank | 10 | 11 | 12 | 13 | 14 | 18 | 20 | 40 |
|---|---|---|---|---|---|---|---|---|
| cosine | .1124 | .1119 | **.1113** | .1101 | .1081 | .1034 | .1023 | .0904 |

The tie at the cut is **0.0012**. Ranks 1 through 40 span only **0.049**. The four
columns whose membership differed between the two runs ranked 9, 11, 13 and 18 —
all inside that flat region.

So "top 12" was always an arbitrary point inside a tie, and a selection that turns
on the last digit of a float decides whether the edit is validated or harmful.
Different hardware, slightly different numerics, different answer.

Two further problems fell out of the same analysis:

**The 12-column edit is a large perturbation, not a surgical one.** 12 of 3584
columns is 0.33% of the matrix, and the measured effect on behaviour the capture
never looked at ranges from −60% to +700% tokens. Nothing about those numbers
looks like a targeted adjustment.

**Our own earlier comparison was confounded.** We had concluded "the ranking
identifies the right columns" because editing a contiguous window `900:1000`
failed while a ranked set succeeded. But `900:1000` contains 97 columns the
ranking did *not* select, so that comparison mostly showed that editing 100
columns breaks things and editing 12 doesn't. It never isolated ranking quality
from edit size. We were comparing an edit to a much larger edit and calling the
difference a ranking result.

## Two findings that survived

**There is no single over-thinking direction in the model.** We captured
directions on a 3 behaviours × 9 layers grid — 27 directions, 378 pairwise
cosines:

| comparison | mean cosine | n |
|---|---|---|
| same behaviour, **different layers** | **+0.006** | 108 |
| different behaviours, **same layer** | **+0.429** | 27 |

A direction at layer 12 is orthogonal to one at layer 6 even for the same
question. So any claim of the form "we located the model's over-thinking
direction" is wrong. What we had was *a working direction at a layer*, twice — not
two measurements of one thing.

But behaviours *do* share a direction within a layer, and that similarity is
strongest at the network's ends (+0.709 at layer 0, +0.540 at 23) and weakest in
the middle (+0.266 at 12). That is a real structural result and it explains why
transfer failed: sharing a direction is not sharing causation. The edit moved
arithmetic the intended way and everything else unpredictably.

**The verifier is what caught this.** We built `wsp.verify` to compare a baseline
reading against an edited one under a fixed rule — `broken`, `regressed`,
`improved`, `unchanged` — and to check `harmful` *before* `validated`, so an edit
that helped its target cannot mask damage elsewhere. Without that ordering, the
Stage 1 result would still be standing as fact, and we would have published it.

## What we would do differently

1. **Do not select top-k inside a plateau.** Either pick a criterion that is
   stable under perturbation, or report the result as an interval across the
   plateau rather than a point. We are now building the stability-based selection
   we should have started with.
2. **Publish the spectrum, not the k.** A reader cannot tell whether a "top 12"
   was meaningful without seeing the scores around the cut. Ours would have shown
   the problem immediately.
3. **Make the probe set broad by default.** Held-out instances of the *same*
   behaviour are not enough. Our guard was a short factual question (204 tokens)
   which a global verbosity change would barely dent; a long-form task exposed it
   at once.
4. **State the operating point exactly** — layer, scale, column count, selection
   rule, and checkpoint hash — because the verdict depends on all of them.

## Status

The honest claim is narrower than the one we started with:

> A direction-derived edit at one layer can cut a specific over-thinking behaviour
> by 55–90% on a fixed checkpoint on fixed hardware, while perturbing other
> behaviours unpredictably, and it does not reproduce across GPUs.

That is not a method. The tooling, the verifier, and the negative results are all
reusable, and the structural finding about layer-locality is worth having. But the
central claim needs the selection problem solved before it means anything.

We think the failure is the more useful half of this project, which is why it is
written up here rather than buried in a changelog.

## Reproducing

Everything is scripted and checkpointed:

- `src/wsp/verify.py` — the fixed comparison rule
- `src/wsp/activations.py` — direction capture, column ranking
- `scripts/pod_claim_c.py` — the 3×9 layer/behaviour grid
- `scripts/pod_ranking.py` — the alignment spectrum around the cut
- `scripts/pod_variance.py` — the determinism measurement
- `docs/stage0-findings.md`, `docs/stage1-findings.md`,
  `docs/stage2-findings.md` — full evidence, including corrections applied to
  Stage 1 after Stage 2 contradicted it
