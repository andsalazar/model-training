# Stage 4c — closing the thinking block works, and costs the guards

## The fix

Stage 4b's targets were cut immediately after the answer, so the adapter learned
"stop right after computing" and never learned to leave the reasoning block.
Re-judged strictly, **zero targets delivered a finished answer** — every output
ended mid-deliberation.

Stage 4c appends the closing marker and the bare answer to each target:

```
…$120 + 12 = 132
</think>

132
```

That is a complete response: reason, close, answer. It is what the prompt asked
for. One-line change to the target builder, same training recipe, same epochs.

## Result: targets fixed

| probe | kind | base | 4b | 4c | delivered |
|---|---|---|---|---|---|
| `arith_12x11` | target | 1609 | 116 | 122 | **yes** |
| `arith_17x23` | target | 908 | 116 | 122 | **yes** |
| `arith_9x8` | target | 181 | 75 | 80 | **yes** |
| `logic_syllogism` | target | 508 | 63 | 71 | **yes** |
| `heldout_capital` | heldout | 1385 | 131 | 180 | **yes** |
| `heldout_symbol` | heldout | 251 | 125 | 162 | **yes** |

All six now end with `</think>` followed by the bare answer. The held-out
behaviour still works — `Tokyo` and `Au` were never in training. Lengths are
slightly longer than 4b (122 vs 116) because the closing marker and answer are
part of the output, which is the point.

## Result: guards regressed

| guard | base | 4b | 4c |
|---|---|---|---|
| `guard_poem` | 2048 | 165 | 782 |
| `guard_hammer` | 350 | 845 | **2048 (cap)** |
| `guard_blue_sky` | 2048 | 197 | **2048 (cap)** |

`guard_poem` still produces a real two-line poem after closing the block, so it
survives. But `hammer` now loops — "Fixing furniture / Driving nails / Chipping
wood / …" repeating — and `blue_sky` runs to the cap in a self-check loop
("Wait, I need to check if I am allowed to use…").

This is the trade-off the recipe makes visible. Training targets to *close and
answer* teaches "stop deliberating and emit", which is right for the targets and
wrong for the guards, where deliberation is the task. The adapter cannot tell
them apart because both were in the same training set with the same instruction
shape.

## What this establishes

1. **Closing the thinking block is learnable.** It was a data-design gap, not a
   model limit. 4b's failure to deliver was caused by cutting targets at the
   answer.
2. **The recipe has a real selectivity limit.** One adapter trained on mixed
   targets cannot both suppress deliberation (targets) and preserve it (guards).
   The 4b adapter preserved guards better because it never learned to close.
3. **The guard set is the binding constraint.** Every attempt so far either fixes
   targets and damages guards, or preserves guards and leaves targets unfinished.

## The obvious next experiment

Train **two adapters** — one on targets only (suppress), one on guards only
(preserve) — and compare against the mixed adapter. If the target-only adapter
fixes targets without touching guards, the problem was the mixed training set,
not the mechanism. That is cheap: same pipeline, one flag to filter the dataset.

If the target-only adapter still damages guards, then the mechanism is genuinely
non-selective and the honest conclusion is that a single LoRA cannot do this.

## Cost

One pod, ~25 minutes (targets ~7 min, training ~2 min, eval ~7 min). The pod was
already running.

## Status of the Stage 4 claim

The defensible claim is now:

> A LoRA adapter trained on the model's own truncated reasoning suppresses
> redundant self-verification and delivers correct short answers on the trained
> behaviour and on a held-out behaviour — at the cost of damaging long-form
> elaboration tasks, which the same adapter cannot distinguish from targets.

That is narrower than "behaviour-selective adapters work", and it is what the
evidence supports.
