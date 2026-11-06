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

## What the metric does and does not establish

**Established:**

- The estimators are correct against known-answer cases (Gaussian closed form, dimension
  scaling, monotonicity in spread).
- The collapse trend is detected when it is real and not flagged when it is absent.
- Delta entropy separates a constructed collapse where per-frame entropy does not.
- The separation threshold is a real guard: a mild collapse (gap 0.106) is correctly rejected
  at 0.15 and correctly accepted at 0.05.

**Not established:**

- **Nothing has been measured on real video.** Every result above is on synthetic arrays. The
  construction tests the *metric*, not the model. Whether Ovi's real rollouts collapse in a
  way this metric catches is unknown.
- **Whether a 5-second rollout collapses at all.** Ovi generates 5 or 10 seconds. Collapse may
  need longer horizons to appear. If real rollouts never collapse, the metric has nothing to
  separate and the project needs a different target.
- **Whether the metric correlates with perceptual quality.** TBC's own framing admits
  differential entropy is a proxy. A rollout could lose entropy while looking fine, or keep
  entropy while looking broken. Nothing here tests that.
- **The absolute scale.** The k-NN estimate flattens in high dimensions. Comparisons between
  frames under the same estimator are valid; absolute values are not comparable across
  different patch counts or dimensions.

## Next step

The metric is ready to run on real frames. The script `scripts/pod_ovi_metric.py` has two
modes:

- `--mode synthetic` — tests the metric on saved frame tensors. Runs anywhere, no GPU. This is
  what was validated above.
- `--mode real` — generates with Ovi and saves frames. Needs a GPU with the model installed.

The real path needs a pod. The persistent pod (`o4uhh2qw28w9zk`, EUR-IS-1, RTX 4000 Ada 53GB)
is currently **EXITED** and its host has no free GPUs — the recurring blocker. Options are to
retry the start, or provision a fresh pod in a data centre with stock.

Once frames exist, the decisive question is: **does delta entropy separate a real degrading
rollout from a real healthy one?** If yes, WSP has a target. If no, the project needs a
different behaviour metric before any adapter work is worth doing.
