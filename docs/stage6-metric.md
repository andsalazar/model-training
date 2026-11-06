# The collapse metric: per-frame entropy does not work, delta entropy does

*Building TBC's differential-entropy metric as pure math, and what testing it against
constructed collapses revealed.*

Companion to [stage6-scoping.md](stage6-scoping.md) (what TBC built) and
[stage6-literature.md](stage6-literature.md) (the literature behind it). This document
covers the metric — the go/no-go for the whole Ovi project.

## Why the metric comes first

TBC's adapter is judged by "differential entropy across generated frames" as a proxy for
information collapse. That metric is the target: without it there is no behaviour to
measure and therefore nothing for WSP to derive an edit from.

The scoping document called this "the item to test first, and it is cheap to test". The
reasoning holds: if the metric cannot separate a rollout that degrades from one that does
not, on the same prompt, then the project has no target. That is worth knowing for a few
dollars rather than after a long training run. Every failure in Stages 1–5 came from
measuring the wrong thing or measuring it badly.

So the metric was built as **pure array math** — no GPU, no video model — and tested
against constructed collapses before any generation was attempted. That ordering is the
point: it isolates the metric from the model.

## What was built

`src/wsp/entropy.py`, with 37 tests in `tests/test_entropy.py`. Three estimators, because
they fail differently:

| Estimator | Property | Role |
|---|---|---|
| `histogram` | Binned plug-in. Biased downward; sensitive to bin width | Sanity check only |
| `knn` | Kozachenko–Leonenko k-NN. Consistent, parameter-light | **Default** for latent patches |
| `gaussian` | Closed-form for a fitted Gaussian. Exact when Gaussian | Comparison only |

The `gaussian` estimator is blind to non-Gaussian structure — which is precisely the
structure a collapse produces — so it is reported for comparison, never as the verdict.
The `histogram` estimator's bias grows with bin count relative to sample count, so it is
also not the verdict. The k-NN estimator is the default because it makes the fewest
assumptions, with the caveat that distances concentrate in high dimensions.

Two metrics are computed, and the distinction turned out to matter more than expected:

- **`entropy_series`** — entropy of each frame independently. "How much information is in
  this frame?"
- **`frame_delta_entropy`** — entropy of the *change* between consecutive frames. "How much
  new information is arriving?"

The second is the one that matches "respond only to what's changing". A rollout can hold
per-frame entropy flat while its delta entropy falls to zero: every frame is rich, but they
are all the same frame. That is exactly the degenerate state TBC's metric is meant to catch.

## Two bugs found by testing

Both were caught by tests against known-answer cases, not by inspection. Both would have
produced plausible-looking numbers in production.

### 1. The k-NN estimator dropped the dimension factor

The Kozachenko–Leonenko estimator is

$$h = \psi(n) - \psi(k) + \ln c_d + d \cdot \overline{\ln \varepsilon_i}$$

where $c_d = \pi^{d/2} / \Gamma(d/2 + 1)$ is the unit-ball volume and $\varepsilon_i$ is the
distance from sample $i$ to its $k$-th nearest neighbour. The $d$ multiplier on the mean
log-distance is easy to omit, and omitting it makes the estimate collapse toward zero as
dimension grows.

The first implementation omitted it. For 32-dimensional unit-variance Gaussian data — whose
true differential entropy is about 61 nats — it returned **−8.0**. A 70-nat error, and one
that reads as "no information" for data that has plenty. Worse, it made the ratio across
frames meaningless: every series looked flat because the dimension term was the only thing
that varied.

The fix applies the factor explicitly and adds a `_digamma` implementation so the metric
needs nothing beyond numpy. Two tests pin it: agreement with the Gaussian estimator on
Gaussian data within 15%, and linear growth in dimension.

### 2. The ratio guard silently returned 1.0 for negative entropy

Differential entropy is a *relative* measure and can be negative for concentrated
distributions. The original guard read:

```python
ratio = late / early if early > 0.0 else 1.0
```

So any series whose early mean was negative got `ratio = 1.0` — reported as perfectly
healthy, regardless of how far it had fallen. That is a silent false negative on exactly the
case the metric exists to detect.

The fix reports `ratio = nan` with `ratio_defined = False`, and `metric_separates` falls back
to comparing slopes when either ratio is undefined. A nan that a caller must handle is
better than a 1.0 that looks like a measurement.

## The finding that changes the plan

Testing against constructed collapses produced a result that was not anticipated:

| Blend strength | Per-frame ratio | Per-frame separates | Delta ratio | Delta separates |
|---|---|---|---|---|
| 0.25 | 0.9604 | no | 0.8889 | no |
| 0.50 | 0.9453 | no | 0.7319 | **yes** |
| 0.75 | 0.9611 | no | 0.4816 | **yes** |
| 0.90 | 0.9842 | no | 0.2343 | **yes** |

**Per-frame entropy does not separate the collapse at any strength. Delta entropy separates
it at three of four.**

This is not a bug — it is the metric behaving correctly and revealing something about the
failure mode. The construction converges each frame toward frame 0 progressively. Per-frame
entropy falls and then *recovers* as the blend saturates: at strength 0.90 the ratio is
0.9842, essentially flat. The frames are individually as rich as they ever were. What has
gone is the *change* between them.

That is the "every frame is rich but they are all the same frame" case, and per-frame entropy
is structurally blind to it. Delta entropy catches it cleanly, monotonically, with a gap that
grows from 0.11 to 0.76 as the collapse deepens.

**Consequence for the project:** if we had implemented only TBC's stated metric — differential
entropy across frames — we would have measured the wrong thing and concluded the metric does
not work. The delta metric is the one to lead with. This is the same class of error as the
three substring tests that invented successes in Stages 4–5: a metric that looks like it is
measuring the behaviour but is not.

## A fixture bug worth recording

The first version of the collapse construction blended each frame toward its *immediate
predecessor*. That reaches a stationary variance rather than converging, so per-frame entropy
fell once and then flattened — no trend. The metric correctly refused to call it a collapse,
and the script exited 1.

The failure was in the fixture, not the metric. But it is worth recording because the
temptation was to loosen the metric's threshold to make the test pass. The metric was right:
a series without a trend is not a collapse. The fixture was changed to converge progressively
toward frame 0, which produces a real trend.

This is the third time in this project that a test failure pointed at the test rather than the
code, and the first time the correct response was to fix the test.

## The estimator that works on real pixels is not the one that worked on synthetic data

The synthetic tests used 32-dimensional Gaussian arrays, where the k-NN estimator is
comfortable. Real Ovi frames are a different regime, and the first real run exposed it.

**k-NN returns negative entropy on raw pixel patches.** Every configuration tried:

| Patch | Patches | Dims | k-NN per-frame mean | Usable |
|---|---|---|---|---|
| 8 | 400 | 192 | −2539.57 | no |
| 8 | 1000 | 192 | −2738.64 | no |
| 16 | 400 | 768 | −8113.80 | no |
| 16 | 1000 | 768 | −8032.35 | no |
| 32 | 400 | 3072 | −11957.95 | no |
| 32 | 1000 | 3072 | −15503.76 | no |
| 16 + PCA 8 | 1000 | 8 | −65.16 | no |
| 16 + PCA 64 | 1000 | 64 | −589.69 | no |

Negative differential entropy is mathematically legal — it is a relative measure — but
here it is a symptom, not a measurement. The cause is that pixel patches contain many
near-duplicate values: flat sky, flat water, flat walls. When a patch has an exact or
near-exact neighbour, the k-th neighbour distance goes to ~0, `log(ε)` goes to −∞, and
the estimate is dragged arbitrarily negative. PCA does not help because the duplicates
survive projection.

**The Gaussian estimator works.** Same video, same patches:

| Configuration | Per-frame mean | Per-frame ratio | Delta ratio | Usable | Separates delta |
|---|---|---|---|---|---|
| Gaussian, raw 768d | 956.38 | 0.860 | 0.884 | **yes** | **yes** |
| Gaussian, PCA 16 | 108.58 | 0.996 | 0.983 | **yes** | **yes** |
| Gaussian, PCA 32 | 200.70 | 0.994 | 0.984 | **yes** | **yes** |

This inverts the synthetic-data conclusion about *which estimator to use*, while
preserving the conclusion about *which metric to use*. On synthetic Gaussian arrays the
k-NN estimator was the principled default and the Gaussian estimator was "blind to
non-Gaussian structure". On real pixel patches the Gaussian estimator is the only one
that produces a usable number, and it still separates a constructed collapse.

The reason is that the Gaussian estimator's weakness — blindness to non-Gaussian
structure — does not matter here. The collapse signal is a change in overall spread
across frames, which a covariance captures. The k-NN estimator's strength — sensitivity
to local density structure — is exactly what the near-duplicate patches corrupt.

**This is the fourth time in this project that a metric which looked right on synthetic
data behaved differently on real data.** The pattern is consistent enough to state as a
rule: validate the estimator on the real data distribution before trusting any number it
produces.

## The Gaussian estimator also fails at full dimension — and PCA is the fix

The Gaussian estimator worked on the first video at 768 dimensions (+956 per-frame mean)
but returned **−1353** and **−2303** on two others. That inconsistency is the tell: the
number was not measuring the video.

At patch size 16 a frame is 768-dimensional. Estimating a 768×768 covariance from 1000
samples is a marginal regime — the matrix is near-singular, and its log-determinant goes
sharply negative. The estimator was reporting the conditioning of the covariance, not the
information in the frame. Whether it came out positive or negative depended on how
degenerate that particular video's patches happened to be.

**PCA to 32 components fixes it.** The basis is fit on all frames together so every frame
is projected into the same space; fitting per frame would make the entropies
incomparable. With 32 dimensions and 1000 samples the covariance is well-conditioned and
the estimate is stable.

## The real-video result: three videos, one consistent answer

Three real Ovi rollouts, 121 frames each at 960×960, 30 denoising steps, Gaussian
estimator, 1000 patches, PCA to 32 components:

| Prompt | Per-frame ratio | Per-frame R² | Delta ratio | Delta R² | Collapsing |
|---|---|---|---|---|---|
| River through a forest, panning | 0.9936 | 0.319 | 0.9838 | 0.117 | no |
| Still mountain lake at dawn | 1.0017 | 0.068 | 0.9774 | 0.019 | no |
| Ceiling fan in an empty room | 0.9971 | 0.007 | 0.9894 | 0.066 | no |

**No real rollout collapses.** Every ratio is within 2% of 1.0 and every R² is far below
the 0.5 threshold. A 5-second Ovi generation is stable — which is what the model is
designed to be, and is consistent with TBC's own framing that collapse is a *long-horizon*
failure.

Constructed collapses on top of those same real frames:

| Prompt | Strength | Per-frame separates | Delta separates | Delta gap |
|---|---|---|---|---|
| River | 0.50 | no | no | +0.0754 |
| River | 0.75 | no | no | +0.1443 |
| River | 0.90 | no | **yes** | +0.2157 |
| Lake | 0.50 | no | no | +0.2600 |
| Lake | 0.75 | no | **yes** | +0.4993 |
| Lake | 0.90 | no | **yes** | +0.7283 |
| Fan | 0.50 | no | no | +0.1247 |
| Fan | 0.75 | no | **yes** | +0.2401 |
| Fan | 0.90 | no | **yes** | +0.3602 |

**Per-frame entropy separates in 0 of 9 cases. Delta entropy separates in 6 of 9**, and
the gap grows monotonically with collapse strength in every video. The synthetic finding
holds exactly on real data, now across three prompts and nine constructions.

The three cases where delta did not separate are all the weakest collapse (strength 0.50)
plus the river at 0.75. That is the threshold doing its job: a gap of 0.075–0.144 is
inside estimator noise, and calling it separation would be the same error as the three
substring tests that invented successes in Stages 4–5.

## What the metric does and does not establish

**Established:**

- The estimators are correct against known-answer cases (Gaussian closed form, dimension
  scaling, monotonicity in spread).
- The collapse trend is detected when it is real and not flagged when it is absent.
- Delta entropy separates a constructed collapse where per-frame entropy does not.
- The separation threshold is a real guard: a mild collapse (gap 0.106) is correctly rejected
  at 0.15 and correctly accepted at 0.05.

**Not established:**

- **No real rollout collapsed.** Three prompts, all stable. So the metric has never been
  shown to catch a *naturally occurring* collapse — only constructed ones. That is a weaker
  claim than "the metric works", and it is the claim being made.
- **Whether a longer rollout collapses.** Ovi generates 5 or 10 seconds. Collapse is a
  long-horizon failure by TBC's own framing, so 10 seconds may still be too short. If no
  natural collapse exists at any length Ovi supports, the project has no target.
- **Whether the metric correlates with perceptual quality.** TBC's own framing admits
  differential entropy is a proxy. A rollout could lose entropy while looking fine, or keep
  entropy while looking broken. Nothing here tests that.
- **The absolute scale.** Comparisons between frames under the same estimator and the same
  PCA basis are valid. Absolute values are not comparable across patch counts, dimensions or
  bases.
- **The SDPA substitution.** Generation used a PyTorch SDPA shim in place of FlashAttention
  (see [stage6-pod-setup.md](stage6-pod-setup.md)). That is adequate for a measurement
  experiment and not adequate for a final quality claim about Ovi's output.

## Where this leaves the project

The metric question is answered, and the answer is mixed in a way that matters.

**What works:** delta entropy, on PCA-reduced patches, with the Gaussian estimator. It
separates a constructed collapse in 6 of 9 cases across three real videos, never fires on a
healthy rollout, and its gap grows monotonically with collapse strength. Per-frame entropy —
TBC's stated metric — separates in 0 of 9.

**What does not:** there is no natural collapse to measure. Every real 5-second rollout is
stable. So the metric is validated against *constructed* degradation, not against the failure
mode TBC actually addresses.

That has a direct consequence for the plan. The scoping document's first step was "does the
metric separate a rollout that degrades from one that doesn't". The honest answer is: it
separates a *synthetically* degraded one, and no real one degrades. Before any adapter work,
the next question is whether a 10-second rollout — or a longer one via the 10s checkpoint —
produces natural collapse. If it does not, then Ovi at these lengths has no stability problem
for an adapter to fix, and the project needs either a longer horizon or a different target
behaviour.

That is a real finding, not a setback. It is the same shape as Stage 5: the recipe had a
precondition (the base trace must contain the desired output) that was not met, and knowing
that saved a training run that could not have worked.
