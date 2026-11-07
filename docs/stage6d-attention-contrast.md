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

The module is an instrument, not a result. Three outcomes are possible and they mean
different things:

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
Replacing that name catches every self- and cross-attention call in both the fused and
unfused paths, because both reach attention through `self_attn(...)`.

`wsp.attention_hooks.AttentionHooks` installs that intercept and computes the summary from
`q` and `k` directly, in query chunks. The intercept calls the original function and returns
its result untouched — a hook that altered the output would make every measurement a
measurement of the hook.

Two structural facts the hook has to respect, both verified against the source:

- **`flash_attention` never receives the timestep.** The step is numbered by wrapping
  `model.forward`, exactly as `wsp.hooks.DitHooks` does for residual capture, so a residual
  capture and an attention capture taken in the same run agree on what "step 3" means.
- **The layer counter must reset per forward.** Without the reset it increases across the
  whole run, so a 12-block model reports layer 12 on the second step and every layer beyond
  the first is mislabelled. Verified load-bearing: with the reset a 2-block, 2-step run
  records 4 summaries; without it, 2 — the second step's blocks fall outside the schedule
  and are silently dropped.

## What this does not establish

- **No attention contrast has been measured on the real model yet.** The module and the
  hook are tested against constructed tensors and a stand-in module. The GPU run is the
  next step.
- **Whether head-level differences are causal.** A head that separates drifting from stable
  may be responding to the drift rather than causing it. Only an edit settles that.
- **Prompt adherence is still unmeasured anywhere in the project.**

## Tests

37 tests in `tests/test_attention.py`, 16 in `tests/test_attention_hooks.py` and 15 in
`tests/test_ovi_attention.py`. Three are load-bearing for the findings above:

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
