# Stage 4 — a behaviour-selective adapter

## Why Stage 3 sent us here

Stage 3 established two things:

- Threshold selection is reproducible — overlap 1.000 under direction
  perturbation, at every ratio tested.
- The resulting edit is harmful anyway — at **4 columns** (0.11% of the matrix) it
  improved the target 83% while pushing `17×23` and both long-form guards to the
  token cap.

Column count was not the controlling variable: 4 columns broke a *different* set of
probes rather than a smaller set. Combined with Stage 2's structural finding — the
captured direction is behaviour-*general* at a layer (+0.429 across behaviours)
while the effect is behaviour-*specific* — the diagnosis is that a **single fixed
vector cannot be behaviour-selective**. Every column carrying it suppresses the
same shared "keep checking" state, whatever task it is running.

The fix has to be a mechanism whose capacity is spent *deciding when* to suppress,
not *where* to write. That is a trained adapter.

## What the adapter must do

Learn to suppress over-thinking on the target behaviour while leaving other
behaviours intact. Concretely:

| requirement | why |
|---|---|
| Intervene on activations, not weights | the relevant signal is a state, and the same state is used by other tasks — the adapter must be able to *condition* on context |
| Train on positives *and* negatives | the Stage 3 failure mode is indiscriminate suppression; the loss has to include tasks where brevity is wrong |
| Stay small enough to interpret | the point is still to learn something about the mechanism, not to beat a benchmark |
| Be verifiable with the same rule | `wsp.verify` already exists and is what caught every previous error |

## Design

A LoRA adapter on the MLP `down_proj` of the layers Stage 1/3 identified (6 and
12), trained to reduce generated length on the target behaviour.

Two departures from a standard fine-tune, both forced by the evidence:

**Negatives are half the dataset.** For every over-thinking prompt, a prompt where
elaboration is correct — poem, list, explanation. Stage 3's `guard_poem` result is
the specification: a two-line poem must not collapse to nothing.

**The loss is on length *and* correctness.** Minimising tokens alone is trivially
solvable and useless. A per-example penalty fires when the expected answer is
absent, so the adapter cannot buy brevity with wrong answers.

```
loss = L_length(target) + λ · L_correct(answer must appear)
```

## Test plan

The adapter is judged by exactly the rule that refuted Stage 1–3, on a broad probe
set, plus two checks the earlier stages could not make:

1. **Broad probe set** — arithmetic, logic, factual, and three long-form guards.
   `harmful` checked before `validated`, as always.
2. **Held-out behaviour** — a prompt type never seen in training, to see whether
   the adapter generalises or memorises.
3. **Ablate the adapter** — swap it off and confirm the baseline returns, so the
   effect is the adapter and not checkpoint drift.

## Falsifiable predictions

| prediction | if true | if false |
|---|---|---|
| Adapter reduces target length, guards unchanged | behaviour-selective suppression is learnable → the project has its method | fixed-vector failure was not about selectivity, and the whole approach is capped |
| Adapter reduces target length, guards also collapse | the shared-state diagnosis was right but an adapter cannot condition finely enough | needs a larger or differently-placed intervention |
| Adapter fails to reduce length at all | the target behaviour is not reachable from this intervention point | the localisation from Stage 1 was wrong after all |

Each outcome is informative. The first is the only one that makes a method.

## Hardware

Training changes the compute shape from the earlier stages. Qwen3.5-0.8B with a
LoRA is a modest job: the weights fit in ~1.7 GB, and a LoRA trains a few million
parameters. A 24 GB card is sufficient, and a 32 GB card gives headroom for the
sequence lengths the thinking traces need (2048 tokens).

Pod cost is the dominant constraint, as it has been throughout. RTX 4090 (24 GB,
$0.34–0.89/hr depending on cloud) and RTX PRO 4500 Blackwell (32 GB, $0.72/hr) are
both adequate. Capacity in EU regions has been intermittent, so the plan assumes
the pod may need a restart and keeps everything either on the network volume or
reproducible from script.

## What Stage 4 does not attempt

- **Not** the original video/audio goal. That remains the destination, but it is
  far downstream of a mechanism that works on a text behaviour.
- **Not** a claim about WSP generally. Stage 4 tests one adapter on one behaviour
  on one model. If it works, the interesting question becomes whether the *same
  adapter recipe* transfers — which is a Stage 5.
- **No** merge back into the base weights until the adapter is validated. Keeping
  it separate is what makes the ablation possible.

## Honest prior

The prior on the first prediction being true is not high. Three stages have each
looked promising and then failed on a broader test, and the shared-state finding
suggests suppression is genuinely entangled with other behaviours. But the
adapter has a capability the fixed vector lacked — it can condition on the input
— and that is precisely the missing piece the diagnosis named. If this also fails,
the honest conclusion is that a 0.8B model's "keep checking" state is not
separably suppressible, which is itself a defensible finding.
