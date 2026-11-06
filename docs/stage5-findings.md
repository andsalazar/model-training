# Stage 5 findings — the recipe did not generalise to format compliance

## Summary

Stage 5 asked whether Stage 4's adapter recipe transfers to a different
behaviour. It does not. The adapter trained cleanly (CE 2.55 → 0.26, LoRA weight
norm 0.0198 — the same signature as the working Stage 4 adapter) and then
produced **zero compliant format answers**.

The first evaluation reported "6 targets newly compliant". That was a bug in the
compliance check, not a result. After fixing it, the count is **0**.

## The measurement bug that invented a success

The check was:

```python
"compliant": ex.expected.lower() in text.lower() and n <= COMPLIANT_CAP
```

The expected strings for format targets are single characters — `"1"`, `" "`,
`"/"`, `"-"`, `"5"` — because the instruction is "output exactly five
comma-separated numbers". So the adapter's output

```
Thinking Process:\n\n1
```

contains `"1"`, is 6 tokens, and passed. Six of eight targets were "compliant"
on that basis. Every one of them is truncated deliberation.

The fix requires three conditions: the answer is present, the output is short,
**and** there are no reasoning markers (`thinking`, `analyze`, `wait`, `**`, …).
The marker list is deliberately broad — a false negative costs one missed
success, a false positive invents one.

Re-judging the same saved data with the corrected rule: **0 newly compliant**.

## What the adapter actually did

It learned to stop, but not to answer. Every target output is a truncated
deliberation header:

| prompt | base | adapter | adapter output |
|---|---|---|---|
| five comma-separated numbers | 2048 | 6 | `Thinking Process:\n\n1` |
| three words separated by spaces | 1456 | 3 | `Thinking ` |
| days Monday to Wednesday | 467 | 27 | `Thinking Process:\n\n1. **Analyze the Request:**…` |
| two colours separated by a slash | 1280 | 65 | `**Analyze the Request:** * Task: Output exactly two colours…` |

It cut length by 80–99% and produced no answer. That is the same failure mode as
the first Stage 4 attempt, in a different guise: the adapter learned a *shortcut
correlated with the target* rather than the target.

## Why the recipe failed here but worked in Stage 4

The difference is what "truncated trace" means for each behaviour.

In Stage 4 the target was arithmetic over-thinking. The model's trace **contains
the answer** and then keeps checking it. Cutting the trace just after the answer
gives a target that is "the same reasoning, minus the redundant verification" —
the adapter only has to learn *when to stop*.

In Stage 5 the target is format compliance. The model's trace **never contains a
compliant answer at all** — it deliberates about the format and runs to the cap
without ever emitting `1, 2, 3, 4, 5`. Truncating that trace gives a target that
is "deliberation, cut off", so the adapter learns to stop mid-sentence. There is
no correct answer in the training signal to learn.

That is a real constraint on the recipe, and it is checkable before spending
compute: **does the base model's trace contain the target output?** If not, trace
truncation cannot teach the behaviour.

## The guards also regressed

| guard | base | adapter |
|---|---|---|
| `12×11` | 1038 | **2048** (cap) |
| `17×23` | 690 | 232 |
| `9×8` | 247 | 146 |
| capital of France | 205 | 172 |
| closest planet | 1212 | 152 |

Four of five got shorter, but `12×11` went to the cap — worse than base. So the
adapter is not behaviour-selective here either; it is a blunt length reducer that
happens to help most prompts and hurt one.

## What this establishes

1. **The Stage 4 result stands** — it was measured with the same corrected rule
   and its targets genuinely improved with correct answers.
2. **The recipe is narrower than hoped.** It works when the base trace contains
   the desired output and the failure is redundant continuation. It fails when
   the desired output is absent from the trace.
3. **The pre-check is cheap.** Before training, measure whether the base model
   ever produces the target output. Stage 5 would have been rejected in minutes
   rather than after a training run.

## Cost

One pod, ~50 minutes total (setup 3.5 min, targets ~4 min, training ~2 min,
evaluation ~7 min, plus model download). Cheap, and the corrected compliance
check is reusable.

## Next

The honest framing for the write-up: the adapter recipe is a *targeted*
intervention with a stated precondition, not a general method. That is a weaker
claim than "behaviour-selective adapters work", but it is the one the evidence
supports.
