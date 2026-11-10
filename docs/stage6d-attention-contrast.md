# Stage 6d: the attention-map contrast

*Why the residual result points at attention, what the module measures, and what would
count as a finding.*

## The problem the residual projection left open

Stage 6c measured a drifting-versus-stable direction in Ovi's residual stream and scored
it against every column of the two matrices that write to that stream — `self_attn.o` and
`ffn.2`. The best alignment anywhere was **0.129** against a chance level of
$1/\sqrt{3072} \approx 0.018$. Above chance, but nowhere near the 0.5+ that would indicate
a weight column writing along the captured direction. Every site's score distribution was
a plateau, so the 0.9-of-best threshold selected zero columns and the `min_columns`
backfill supplied four.

The arithmetic that followed is why no edit was applied: at cosine ≈ 0.1, a `strength=0.02`
damping moves the state ~0.2% of its norm, against a 2.5–4.7% effect to chase. A null
result from that would be uninterpretable rather than cheap.

### The structural explanation

A residual-stream state is a **sum** of every component's contribution: the attention
output, the FFN output, the timestep embedding, the text conditioning. A direction derived
from that sum is dominated by whichever component contributes most, and there is no reason
for that to be the component that distinguishes drifting from stable.

What distinguishes a stable rollout from a drifting one may not be *what* is written to the
residual stream but *where each token reads from* — the attention pattern. That is not
representable as a column of `o` or `ffn.2`. `o` maps the attention output back to the
residual width, but the pattern itself lives in the `[heads, queries, keys]` tensor before
that projection. A damping edit on `o` cannot reach it.

So the negative result does not say "there is no stability direction in Ovi". It says "the
stability direction is not in the two matrices we scored". Those are different claims, and
only the second one is measured.

## What the module measures

Full attention maps are enormous: 24 heads × 27,900 queries × 27,900 keys is 18.7 billion
entries per layer per step. So `wsp.attention` reduces each tensor to three per-head
aggregates, chosen so they are comparable across rollouts that visit different timesteps.

| Field | What it is | What a change means |
|---|---|---|
| `row_entropy` | mean Shannon entropy of each query's attention distribution, in nats | low = each query reads from few keys; a focused or collapsed pattern |
| `column_mass` | mean attention each key receives, normalised to sum to 1 | *which* keys are read — row entropy cannot say this |
| `diagonal_mass` | mean attention each query places on its own key | high = each patch reads its own position rather than its neighbours |

The reasoning for aggregates rather than raw maps is the same one that made delta entropy
work where per-frame entropy did not in [stage6-metric.md](stage6-metric.md): the two
rollouts visit different timesteps, so their query/key sets differ in content even when the
shapes match. An aggregate over the key axis is comparable; a raw map is not.

### The contrast and the ranking

`attention_contrast(a, b)` reports per-head differences, with `a` the condition of interest
(drifting) and `b` the reference (stable). The sign convention is stated in the docstring
because an earlier stage in this project shipped an inverted `direction_similarity` sign and
reported the opposite conclusion.

`effect_size` is the entropy difference divided by the pooled within-head spread of the two
maps' per-query entropies. It is a crude guard, but without it a head whose entropy differs
by 0.001 while its own queries vary by 0.5 would rank alongside a head that genuinely
separates. `head_ranking` orders by it, with ties broken by head index so the ordering is
deterministic — Stage 2's reproducibility failure came from a top-k cut sitting inside a
plateau of near-equal scores.

## Three bugs the tests caught

Each was a real defect in the first implementation, not a test artefact.

1. **`column_mass_entropy` mixed heads into one distribution.** Flattening the
   `[heads, keys]` array before computing entropy reported $\log(\text{heads} \times
   \text{keys})$ for a uniform map — 2.08 nats for a 2×4 case whose per-head answer is
   $\log 4 = 1.386$. It now averages per head.
2. **`min_effect=0.0` admitted unchanged heads.** The threshold filtered on effect size,
   and an unchanged head has effect size 0.0, so `0.0 < 0.0` is false and it passed. A
   caller asking for "all heads" got a list of zero-delta rows that reads like a finding.
   There is now an explicit "did not move" guard.
3. **A degenerate head divided by zero.** A head whose queries all have identical entropy
   has zero pooled spread. The first version returned `0.0`, which hides a real difference —
   the "tool that cannot look must return zero" failure this project has hit four times.
   It now scales by the head's own entropy level, which is finite and interpretable.

The third fix is the one worth noting: returning `0.0` would have been the *safe-looking*
choice and would have silently discarded the strongest possible signal.

## What would count as a finding

Three outcomes were possible and they mean different things. After the null baseline,
video lands on the third row and audio on a weak version of the first:

| Outcome | Reading |
|---|---|
| A few heads separate with large effect size | **the lead.** A targeted edit has somewhere to aim, and the head indices are the placement decision TBC never published |
| Many heads separate weakly | **ambiguous.** Consistent with "the whole pattern shifts", which is what a step-count change would do without any head being causal |
| No head separates | **the residual result stands unexplained.** Either the contrast is not the mechanism, or the measurement is still aimed at the wrong tensor |

The middle outcome is the most likely and the least useful. Distinguishing "many heads
shifted" from "a few heads carry the difference" needs a stability check across seeds —
the same lesson as the colour verdict, where a single-seed reading was wrong and three
seeds settled it.

## How the hook reaches real attention

FlashAttention's fused kernel never materialises the probability matrix, so there is no
tensor to hook. The reachable object is the **function**: Ovi's `WanSelfAttention.forward`
calls the module-level `flash_attention`, imported into `ovi.modules.model`'s namespace.
In the fused path each fusion block calls `audio_block.self_attn` **then**
`vid_block.self_attn`; both reach that name. Cross-attention goes through
`ovi/modules/fusion.py`, which imports `flash_attention` from `ovi.modules.attention`, so
it is **not** intercepted.

`wsp.attention_hooks.AttentionHooks` installs that intercept and computes the summary from
`q` and `k` directly, in query chunks. The intercept calls the original function and returns
its result untouched — a hook that altered the output would make every measurement a
measurement of the hook.

Three structural facts the hook has to respect, all verified against the source:

- **`flash_attention` never receives the timestep.** The step is numbered by wrapping
  `model.forward`, exactly as `wsp.hooks.DitHooks` does for residual capture, so a residual
  capture and an attention capture taken in the same run agree on what "step 3" means.
- **A call does not know its stream or block.** The v1 hook counted calls per forward and
  called the count a layer, which interleaved the two streams (see the retraction below).
  Calls are now labelled by forward pre/post hooks on every `blocks[i].self_attn` of both
  stacks; a call outside any `self_attn` is counted as `unlabelled`.
- **CFG runs the model twice at each timestep**, conditional then unconditional (with
  block `slg_layer` skipped). Both share a capture key, so the hook records one pass —
  the conditional one by default — and counts the other as `other_pass_declined`.

## v1 result — retracted

The first two runs (index-aligned, then nearest-timestep with a 50-unit tolerance)
reported a peak of 0.680 and then 0.1286 at "layer 11 head 17". **Both are retracted.** A
review ([stage6e-review-corrections.md](stage6e-review-corrections.md)) found that:

- "layers 0, 2, 4, 8" were **audio** blocks 0, 1, 2, 4 and "layer 11" was **video** block
  5 — the label was a per-forward call counter over interleaved streams;
- every summary was of the **unconditional** CFG pass, which overwrote the conditional;
- the rollout used `generate()`'s sampler defaults, not the config's;
- `effect_size` was always the relative fallback `|Δ|/mean`, never a pooled-SD effect;
- only 4 of 10 requested timesteps were compared, the rest dropped silently;
- the quoted schedule (t=233 vs 756) was Euler's; Ovi's UniPC gives 356 vs 768;
- "exactly zero" meant "no head crossed the 0.05 reporting threshold".

The index-alignment confound itself was real and its removal stands.

## v2 result

Same prompt and seed, 20 vs 30 steps, the config's sampler settings (guidance 4.0/3.0,
`slg_layer` 11, the config's negative prompts), conditional pass, **both streams**, blocks
0, 3, …, 27 of each. Capture steps were chosen so both runs sit on the same ten UniPC
timesteps — the 20-step run's even indices and the 30-step run's every-third indices
coincide exactly:

```
999, 978, 952, 920, 882, 833, 768, 681, 555, 356
```

**200 of 200 pairs compared, every one at gap 0, none skipped.** 2,360 / 3,540 calls fired,
none unlabelled or failed; 1,160 / 1,740 unconditional-pass calls declined; 200 summaries
per rollout. Effects are pooled-SD (`|Δ entropy| / pooled per-query std`) unless a row had
no head above 0.05.

### Max effect per block and timestep

Video (24 heads per block):

| block | 999 | 978 | 952 | 920 | 882 | 833 | 768 | 681 | 555 | 356 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | .06 | .09 | .12 | .21 | .33 | .51 | .58 |
| 3 | 0 | .31 | .27 | .30 | .36 | .36 | .39 | .37 | .29 | .35 |
| 6 | 0 | .24 | .29 | .28 | .37 | .35 | .20 | .24 | .32 | .42 |
| 9 | 0 | .26 | .34 | .33 | .41 | .32 | .28 | .32 | .31 | .34 |
| 12 | 0 | .26 | .27 | .54 | .50 | .32 | .27 | .29 | .33 | .38 |
| 15 | 0 | .33 | .23 | .30 | .26 | .39 | .37 | .32 | .37 | .44 |
| 18 | 0 | .32 | .40 | .39 | .42 | .45 | .40 | .39 | .40 | .42 |
| 21 | 0 | .32 | .46 | .58 | .41 | .42 | .35 | .35 | .36 | .41 |
| 24 | 0 | .56 | .30 | .50 | .46 | .38 | .34 | .39 | .43 | .45 |
| 27 | 0 | .93 | .56 | .49 | .55 | .46 | .39 | .41 | .42 | .45 |

Audio (24 heads per block):

| block | 999 | 978 | 952 | 920 | 882 | 833 | 768 | 681 | 555 | 356 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | .08 | .19 | .40 | .65 | .86 |
| 3 | 0 | 0 | .07 | .09 | .10 | .52 | **1.17** | **1.28** | **1.05** | .81 |
| 6 | 0 | .09 | .16 | .12 | .08 | .14 | .48 | .42 | .41 | .49 |
| 9 | 0 | .16 | .11 | .12 | .13 | .34 | .57 | .54 | .60 | .64 |
| 12 | 0 | .14 | .15 | .21 | .17 | .79 | .77 | .91 | .95 | .95 |
| 15 | 0 | .20 | .29 | .30 | .25 | .60 | **1.22** | **1.20** | **1.07** | .76 |
| 18 | 0 | .27 | .28 | .38 | .26 | .54 | .59 | .44 | .47 | .45 |
| 21 | 0 | .47 | .31 | .44 | .28 | .51 | .52 | .50 | .90 | **1.11** |
| 24 | 0 | .20 | .18 | .22 | .25 | .34 | .47 | .59 | .95 | .77 |
| 27 | 0 | .13 | .15 | .27 | .34 | **1.02** | .94 | .58 | .92 | **1.04** |

t=999 is 0 everywhere, which is the correctness check: both runs start from the same noise
at the same timestep, so their first-step states are identical.

### Concentration: few heads or the whole pattern?

| | heads moving (≥0.05) per block | top-1 / top-2 effect (median) | sign of Δ entropy (top head) |
|---|---|---|---|
| video | 13–24 of 24 (blocks ≥3) | 1.03–1.31 | **+ in 82 of 88** rows |
| audio | 0–23 of 24, rising as t falls | 1.21–1.75 | mixed |

Read alone, video looks like a whole-pattern shift (nearly every head moves, no head
stands clear, the 20-step rollout's attention is more diffuse) and audio looks closer to
the lead (larger, later, more head-concentrated effects; audio block 3 head 7, block 12
head 21 and block 18 head 2 each top in 5 of 9 timesteps). The null baseline below shows
how much of that survives.

### Null baseline: 30 vs 40 steps

Both 30 and 40 steps were colour-stable in the sweeps, so a contrast between them
measures what any change of step count does to attention, with no drift involved. Ovi's
UniPC grid is a shifted linspace, so 40 steps at indices 0, 4, …, 36 sits on the same ten
timesteps. Same prompt, seed, settings and blocks; 200 of 200 pairs at gap 0.

Per (stream, block, timestep) cell, t=999 excluded (90 cells per stream):

| | 20 vs 30 median max effect | 30 vs 40 (null) | median ratio | 20v30 larger in |
|---|---|---|---|---|
| video | 0.358 | 0.331 | **1.02** | 46 / 90 |
| audio | 0.415 | 0.285 | **1.31** | 58 / 90 |

**Video: no signal above the null.** The 20-vs-30 contrast is indistinguishable from
30-vs-40, and at low noise the null is *larger* (t=555: 0.37 vs 0.66). The null's top
heads are also mostly more diffuse at the lower step count (69 of 90), so "fewer steps,
more diffuse video attention" is what changing the step count does, not a signature of
drift.

**Audio: a modest excess, carried by three blocks.** Mean over timesteps:

| audio block | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 vs 30 | .24 | **.57** | .26 | .36 | **.56** | .65 | .41 | .56 | .44 | .60 |
| 30 vs 40 | .10 | **.19** | .20 | .35 | **.34** | .67 | .48 | .56 | .44 | .50 |

Blocks 0, 3 and 12 separate 20 from 30 by 1.6–3× more than the null does; the other seven
blocks match it. That is the only part of the attention contrast that is not explained by
step count, and it rests on one seed.

### What this does not establish

- **That the audio excess is about drift.** The null removes step-count effects, but the
  speed sweeps found audio unchanged between 20 and 30 steps, so there is no measured
  audio defect for blocks 0/3/12 to explain.
- **That this prompt and seed drift at 20 steps under the config's settings.** The colour
  verdict (2 of 3 seeds drift) came from the sweeps; this rollout's colour was not checked.
- **Causality.** A head that separates may respond to drift rather than cause it. Only an
  edit settles that.
- **Stability across seeds** — one seed pair. The colour verdict needed three.
- **Prompt adherence** — still unmeasured anywhere in the project.

## Tests

52 tests in `tests/test_attention.py`, 23 in `tests/test_attention_hooks.py` and 28 in
`tests/test_ovi_attention.py`. Four are load-bearing for the findings above:

- `TestSummarizeAttention::test_zero_rows_are_rejected` — a pre-softmax hook would look
  like this, and silently normalising it would produce plausible-looking entropies from a
  non-distribution.
- `TestAttentionContrast::test_unchanged_heads_are_not_reported` — the guard that stops a
  zero-delta row being reported as a contrast. Verified load-bearing by reimplementing the
  loop without it: the real version reports `[0]`, the unguarded version reports `[0, 1]`.
- `TestAttentionMapSummary::test_column_mass_entropy_is_reported` — the head-mixing bug.
  `_focused` has each query reading its own key, so column mass is *uniform* across keys and
  entropy is maximum, not zero. The first version of this test asserted zero and was wrong
  about what the fixture meant.

Two more are load-bearing for the hook:

- `TestSummarizeAttentionChunked::test_matches_the_materialised_path` — the chunked
  computation is what actually runs on the GPU, because the full matrix cannot be
  materialised. If it disagreed with the tested materialised path, every GPU number would be
  unverified. Verified load-bearing by reimplementing the reductions with a deliberate bug
  (normalising column mass across the whole array instead of per head): the comparison
  catches it.
- `TestRecording::test_step_is_numbered_from_the_model_forward` — two calls at the same `t`
  are one step (the CFG double-pass); a new `t` starts a new step. Without this the capture
  would file 40 attention calls as 40 steps instead of 20.
- `TestContrast::test_a_gap_beyond_tolerance_is_dropped_not_contrasted` — the guard that
  removed the confound. Verified load-bearing by the pairing check: index pairing gives
  `[(0,0),(2,2),...,(18,18)]` while nearest-timestep pairing gives
  `[(0,0),(2,4),(4,6),(6,10),(8,12),(10,16),(12,18),(14,18),(16,18),(18,18)]` — genuinely
  different, and the index version compares t=233 against t=756.
