# Stage 2 — does the method transfer?

Stage 1 answered "can the edit be derived from measurement alone?" with yes, and
found the narrow condition it depends on: use the ranked columns exactly.

Stage 2 asks the question that decides whether this is a method or a lucky
result. Stage 1's evidence is one model, one behaviour, two layers. Three
specific claims need testing, and they are separable — each can fail on its own.

## Claim A — transfer to a second behaviour

Does the same procedure suppress a *different* over-thinking behaviour in the same
model?

**Why it might fail.** The layer-12 and layer-6 edits both targeted trivial
arithmetic. If the "over-thinking direction" is really an arithmetic-specific
feature, a logic or factual prompt will not move. If it is a general
self-verification feature, it will.

**Test.** Run the survey to find a second behaviour that is genuinely present —
not one that sounds plausible. Stage 1 lost several runs to editing against a
behaviour the model did not actually show, so measurement comes first.

**Pass condition.** A validated edit on a behaviour that is not the Stage 1 target,
with the guard and held-out probes intact.

**Honest failure mode.** The second behaviour may be present but not
direction-editable — for instance a formatting failure with no clear activation
contrast. That is informative, not a dead end: it bounds what the method covers.

## Claim B — transfer to a second model

Does the procedure work on a model it was not developed on?

**Why it matters.** Everything so far is Qwen3.5-0.8B. A method that only works on
one architecture with one tokenizer is a case study.

**Practical constraint.** A second model needs to fit the pod's 32 GB and be small
enough that a directional edit is plausible. Same-family models are the cheapest
test but the weakest evidence; a different family is stronger.

**Pass condition.** The seven steps run without model-specific code changes, and
the edit validates — or fails for a stated reason that is about the model rather
than the tooling.

## Claim C — the direction is layer-local, not shared

Stage 1 found `direction_similarity` near −0.06 between layers 6, 10 and 12, yet
edits at both 6 and 12 validated. That is a real finding and it needs stating
precisely: is there a *shared* over-thinking direction, or does each layer carry
its own?

**Why it is worth knowing.** If the direction is layer-local, the method is "find
a working layer, not a canonical one", and any claim about "the model's
over-thinking direction" is wrong. That changes how the result should be
described and published.

**Test.** Capture the direction at every layer and measure pairwise similarity,
plus per-layer edit outcomes at a fixed scale and column count. If several
layers work with unrelated directions, the feature is distributed and redundant —
a more interesting and more defensible result than a single magic layer.

## What "transfer" would let us claim

If A and B both pass, the honest claim is: *a measurement-driven directional edit
at a ranked column set suppresses a specific over-thinking behaviour, reproducibly
across behaviours and models, without damaging held-out instances or unrelated
behaviour.*

If A passes and B fails, the claim is narrower: *the procedure transfers across
behaviours within a model family.*

If A fails, the claim retreats to Stage 1: *the procedure produces a validated
edit for the behaviour it was developed on.* Still a real result — Stage 0 only
reproduced someone else's edit; Stage 1 derived one — but not a method.

## Order of work

Claim C is cheapest and can run on the model already loaded, so it comes first.
Claim A needs the survey, which is running now. Claim B needs a second model
downloaded, so it comes last.

## Cost

The pod is $0.72/hr. Claim C is a few forward passes per layer; Claim A is one
editor run plus verification; Claim B adds a model download. The whole of Stage 2
should fit in a few hours of pod time, and everything is scripted so a lost pod
does not lose a result.
