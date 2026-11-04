# Stage 0 — what it established

Reproduction of Experiment 8 from [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab — MIT CSAIL, November 2026), plus a first
attempt at the question the paper leaves open.

## 1. The headline result reproduces

Run on the paper's own 25-problem practice subset, using its released
per-problem scores as the comparison target.

| Run | Accuracy | Cut off | Mean tokens |
|---|---|---|---|
| paper, base | 5.0% | 95.0% | 31,690 |
| **ours, base** | **4.0%** | **96.0%** | **31,686** |
| paper, edited | 51.0% | 31.6% | 21,704 |
| **ours, edited** | **64.0%** | **24.0%** | **22,716** |

The baseline matches to within one accuracy point and four mean tokens. The
edited arm scores *higher* than the paper reports, which is the safe direction:
the edit's effect is at least as large as claimed, not an artifact of a lenient
baseline.

The mechanism is confirmed independently of accuracy. The edit's signature is
that token-limit hits collapse — 96% → 24% here, 95% → 32% in the paper. The
model stops looping. That is the claim, and it holds.

## 2. The paper's arithmetic reproduces exactly

| Check | Result |
|---|---|
| `down_proj` shape | `[1024, 3584]` BF16 |
| 64 columns × 1024 rows × 2 layers | **131,072** — paper says 131,072 |
| Footprint of 800M params | **0.0164%** — paper says 0.016% |
| Edit confinement | stays inside the target columns |

## 3. Layers 15 and 19 are both full-attention layers

Qwen3.5-0.8B interleaves attention types: three linear-attention layers, then
one full-attention layer, repeating. The full-attention layers are **3, 7, 11,
15, 19, 23**.

Both layers the paper edited are in that set. This is unlikely to be
coincidence: a repetition attractor needs to see the whole sequence, which is
what full attention provides and linear attention does not. It yields a
testable prediction the paper does not make — **edit a linear-attention layer's
`down_proj` and the fix should fail.** That experiment has not been run.

## 4. Three evaluation details decide the result

The gap between a naive rerun and the paper is almost entirely protocol. Each
of these was wrong in an earlier version of this harness, and each moved the
baseline by more than the edit itself.

**Prompt.** The user turn is the problem, a newline, then Qwen's math
instruction. Not a wrapper asking for reasoning.

**Extraction.** The last `\boxed{}` after the final `</think>`. An unclosed or
empty box yields *no answer*. There is deliberately no fallback to the last
number: a truncated response's last number is an intermediate step, and
crediting it inflates the score. This single difference accounted for a
baseline of 15% where the paper reports 5%.

**Grading.** `math_verify` equivalence, not string normalisation. `0.5` and
`\frac{1}{2}` are the same answer.

The token budget is the fourth: at 32,768 tokens the base model runs to the
limit on 95% of answers. A 4,096-token cap stops generation before the loop can
express itself, which measures a different phenomenon.

## 5. The blind-vs-weights ablation: all five editors declined

The paper reports that Astra found a bug "that Qwen's own model card warns
about", which raises a confound — an editor could produce the fix from recall
rather than from reading weights. The ablation tests this by giving each editor
the same request twice, once with a weight slice and once without.

| Editor | Weights arm | Blind arm |
|---|---|---|
| `astra` | declined | declined |
| `kimi` | declined | declined |
| `deepseek` | declined | declined |
| `glm` | declined | declined |
| `qwen` | declined | declined |

Every editor declined, and every one gave the same reason. Astra, the paper's
own editor, put it most directly:

> "The conspicuous value `0.18262` is not, by itself, evidence of a defective
> weight. A column's effect depends on its intermediate-channel activation and
> how downstream layers process its output; weight magnitude alone does not
> establish a causal link to repetition. **Naming columns or values now would be
> guesswork.**"

DeepSeek and Kimi independently named what would be needed: activation-level
evidence from looping versus terminating traces — per-column causal ablation,
path patching, or logit-lens analysis.

### What this does and does not show

It does **not** contradict the paper. The paper's editor had tools this
ablation withholds: it ran the model, measured activations, and iterated across
50 forward passes. A single static slice with no feedback is a different and
much harder task, and the editors are right that it is underdetermined.

It does establish that the capability is **not** "read a weight matrix and spot
the bug". Every editor, including the one that actually did it, says a static
slice is insufficient. Whatever Astra did required measurement.

That also weakens the recall confound. Recall would not help either: knowing
"Qwen3.5-0.8B loops" does not tell you which 64 columns to change. The paper's
result is more likely genuine measurement than contamination.

## 6. Engineering findings

**vLLM is 43× faster than transformers here.** 1,539 tok/s versus 36 tok/s on
the same model and hardware. The transformers path fell back to reference
PyTorch for `causal_conv1d` (no prebuilt wheel), pinning a CPU core at 99% and
starving the GPU to 19% utilisation. Without vLLM the full comparison would
have taken ~25 hours instead of ~20 minutes.

**`math_verify` silently returns `[]` on Windows.** Its timeout uses
`signal.alarm`, which does not exist there, so every parse fails and every
answer grades as wrong — with no error. The timeout is now requested only where
the mechanism exists.

**safetensors' numpy backend cannot read bfloat16.** Every released checkpoint
uses it. The store now uses the torch backend and reads tensor metadata from the
header, so summarising a checkpoint does not materialise its tensors.

## 7. What Stage 0 did not establish

- Whether the edit generalises beyond the 25-problem practice subset. The paper
  reports all 500; this reproduction used the subset it released scores for.
- Whether editing a linear-attention layer fails, which would confirm the
  full-attention explanation.
- Whether an editor can *drive* the measurement loop. That is the open question
  the ablation reframed, and the subject of Stage 1.

## Cost

Roughly $9 of rented GPU across five pods, all stopped. The API spend for the
ablation was under $1.
