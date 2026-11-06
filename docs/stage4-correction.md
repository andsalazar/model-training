# Stage 4 correction — the adapter truncates deliberation, it does not finish answers

## Why this document exists

Stage 4's saved evaluation kept only `text_chars`, not the generated text. The
`correct` flag was a substring test (`expected in text`). Re-running the adapter
with the text preserved shows that the substring test was passing on answers
that appear **inside unfinished deliberation**, not on completed responses.

The adapter's headline numbers still hold. The interpretation of them does not.

## What the text actually shows

Every target's output ends mid-sentence, inside the reasoning block:

| probe | tokens | how the output ends |
|---|---|---|
| `arith_12x11` | 116 | `…$120 + 12 = 132` |
| `arith_17x23` | 116 | `…$340 + 51 = 391` |
| `arith_9x8` | 75 | `…$9 \times 8 = 72` |
| `logic_syllogism` | 63 | `…Constraint: Reply with only "yes` |
| `heldout_capital` | 131 | `…Option 2: Tokyo` |
| `heldout_symbol` | 125 | `…Is it the standard chemical symbol? Yes.` |
| `guard_poem` | 165 | `…And the sky hangs low and dark.` |
| `guard_blue_sky` | 197 | `…such as red and orange` |
| `guard_hammer` | 845 | `…3. Cleaning metal or wood for polishing.` |

The arithmetic cases are the least bad: the answer is computed and stated as the
last thing in the trace. But the model never leaves the thinking block to emit
the bare answer the prompt asked for. `logic_syllogism` is worse — it stops
inside the *constraint restatement*, before answering at all.

Only `guard_hammer` reaches a genuine final answer, after a `</think>` marker.

## What this means for the claim

The correct claim is narrower than "the adapter produces correct short answers":

> The adapter suppresses redundant continuation. It cuts the model's
> self-verification loop and leaves the computed answer as the last thing in the
> trace — but it does not reliably close the thinking block and emit the final
> response.

That is still a real effect and still behaviour-selective: the base model spends
1609 tokens on `12×11` and the adapter spends 116, with the answer present. But
it is *not* a complete fix for over-thinking. A user would still not get a clean
answer.

## Why the substring test hid this

`expected in text` asks "does the answer appear anywhere". For arithmetic that is
almost always true once the model has computed it — even when the model then
keeps going for another 1400 tokens. So the flag measured "the model got there",
not "the model stopped there".

The fix used in Stage 5's compliance check is the right shape: require the answer
to be present, the output to be short, **and** no reasoning markers. Applied
here, most of these outputs would fail — correctly, because they are unfinished.

## What survives

1. **The length reduction is real and large** — 1609 → 116, 908 → 116, 2048 →
   165. Not a measurement artifact; the text confirms it.
2. **The answers are computed correctly** — `132`, `391`, `72`, `Tokyo`, `Au` all
   appear and are right.
3. **The guards still do their work** — the poem is a real poem, the hammer list
   is a real list. Not a global verbosity knob.
4. **The ablation is clean** — removing the adapter returns the base behaviour.

## What does not survive

1. **"Correct" as previously reported.** The substring test conflated "answer
   present in the trace" with "answer delivered". Re-judged strictly, most
   targets are unfinished.
2. **The claim that the adapter fixes over-thinking.** It shortens it. Closing
   the thinking block is a separate behaviour the adapter did not learn, because
   the training targets were cut *at* the answer, not after it.

## The fixable part

The training targets were truncated immediately after the answer, so the adapter
learned "stop right after computing". If the targets instead included the closing
`</think>` marker and the bare answer, the adapter would learn to close and emit.
That is a one-line change to the target builder and a retrain — cheap, and it
would test whether the remaining gap is a data-design artifact or a real limit.

## Cost

One pod, ~25 minutes for the rebuild (targets ~7 min, training ~2 min, eval
~7 min, plus vLLM load). The pod was already running for Stage 5.
