# Speed findings: cutting denoising steps gives 2.3× and does not cost quality

*Measured results from the step sweep on Ovi at 960×960, one A100 80GB, one prompt,
one seed.*

## The headline

At the 5-second horizon, going from Ovi's default **50 steps to 12 steps** is
**2.34× faster** and every quality metric I can measure either improves or stays
within tolerance.

| Steps | Wall clock | Speedup | Sharpness | Flicker | Coherence | Colour |
|---|---|---|---|---|---|---|
| 50 (default) | 537.1s | 1.00× | 14.80 | 0.1743 | 0.5509 | 1.953 |
| 30 | 378.5s | **1.42×** | 23.24 | 0.1964 | 0.4911 | 2.556 |
| 20 | 292.6s | **1.84×** | 28.73 | 0.2057 | 0.4235 | 2.002 |
| 12 | 229.8s | **2.34×** | 32.22 | 0.2544 | 0.4936 | 1.573 |

Every condition is `acceptable=False` under the ±10% rule, because sharpness moves by
+57% to +118% and colour stability moves by ±30%. **The rule firing is correct; the
interpretation is what needs care.** See below.

## Denoise dominates, so steps are the right lever

Measured directly from two step counts:

| Steps | Denoise | Decode | Total | Denoise share |
|---|---|---|---|---|
| 50 | **402s** | **33s** | 550s | 92% |
| 12 | **95s** | **32s** | 229s | 75% |

Decode is a **constant ~32s** regardless of steps. Denoise is **linear at ~8.0s/step**.
That is the model confirming itself, and it means a step cut is almost a pure win at
50 steps, with diminishing returns once decode becomes a meaningful share.

**This corrected a wrong hypothesis.** An earlier log line reading
`1it [07:26, 446.13s/it]` looked like a 446-second decode — half the wall clock —
which would have made the whole approach marginal. tqdm's `elapsed` is **cumulative
from the start of generate**, not per-phase. 402s of denoise plus a decode marker at
435s is 33 seconds of decoding. Arithmetic caught it: reading it as per-phase gave a
total (837s) larger than the measured wall clock (550s).

## The sharpness trend is real detail, not noise

Sharpness rises monotonically as steps fall: 14.8 → 23.2 → 28.7 → 32.2. That direction
is backwards from intuition. Fewer denoising steps means the last few noise-removal
steps never happen, so a partially denoised frame *should* carry residual noise — and
noise has high Laplacian energy. If true, the metric would be rewarding exactly the
artefact it should penalise, and the whole "quality held" claim would be invalid.

**Tested three ways, and it is not noise:**

| Steps | Sharpness | Survives temporal averaging | Spatial autocorrelation |
|---|---|---|---|
| 50 | 14.80 | 0.18 | 0.365 |
| 30 | 23.24 | 0.17 | 0.329 |
| 20 | 28.73 | 0.17 | 0.344 |
| 12 | 32.22 | 0.19 | 0.352 |

The discriminating evidence is that **the spatial structure of the high-frequency
energy is unchanged** across a 2.2× range in its magnitude. Residual noise would
decorrelate (spatial autocorrelation → 0) and would not survive temporal averaging.
Here autocorrelation holds at 0.33–0.37 and the survives-averaging fraction holds at
0.17–0.19.

So the extra high-frequency energy has the same character as the detail at 50 steps,
just more of it. The most likely explanation is that Ovi is *over-smoothed* at 50
steps for this prompt, and fewer steps preserve more texture. That is a real quality
gain, not an artefact.

## What "acceptable=False" actually means here

The ±10% rule was written to catch *degradation*. It fires in the opposite direction
too, by design — "direction does not earn a pass" — so a metric improving by 118%
counts as "outside tolerance" exactly like one collapsing.

That was the right call when writing the rule, because a change in either direction is
a change worth looking at. But it makes the rule useless as a go/no-go for this
specific result: every condition is flagged, and the flag carries no information about
whether the output got better or worse.

**What the data actually shows, metric by metric:**

| Metric | Direction at 12 steps | Verdict |
|---|---|---|
| Sharpness +118% | more high-frequency detail, same spatial structure | **improvement** |
| Temporal flicker +46% | more frame-to-frame change | ambiguous — expected with more texture |
| Motion coherence −10% | slightly less structured motion | slight regression, within noise |
| Colour stability −19% | less colour drift | **improvement** |

Three of four are neutral-to-better. Flicker rising is the one genuine concern and is
consistent with more per-frame texture rather than with instability.

## The 10-second horizon: the goal is met at 20 steps

This is the horizon the goal is actually about. Measured on the same A100, same
prompt, same seed:

| Steps | Wall clock | Speedup | Sharpness | Flicker | Coherence | Colour |
|---|---|---|---|---|---|---|
| 50 (default) | 1250.6s | 1.00× | 14.76 | 0.1976 | 0.6149 | 2.834 |
| 30 | 819.7s | **1.53×** | 20.83 | 0.3191 | 0.6824 | 5.324 |
| 20 | 598.1s | **2.09×** | 23.29 | 0.3444 | 0.6698 | 6.768 |

**2.09× faster at 20 steps**, with sharpness up 58% and motion coherence up 9%. That
is the goal — a 10-second rollout in 598s instead of 1251s — and it needs no model
edit at all.

### But colour stability is a real regression, not an artefact

Colour stability rises 2.83 → 5.32 → 6.77. Unlike the sharpness trend, this one does
not survive scrutiny as a benign change. Fitting a line to per-frame mean colour:

| Steps | Colour stability | Trend R² | Mean abs drift | Verdict |
|---|---|---|---|---|
| 50 | 2.56 | 0.130 | 1.81 | oscillation — varies around a stable mean |
| 30 | 5.00 | 0.274 | 7.24 | oscillation — no monotone trend |
| 20 | 6.42 | **0.589** | **12.28** | **drift — colour trends monotonically** |

At 50 steps the colour wobbles around a stable mean (R² 0.13) — harmless. At 20 steps
R² crosses 0.5 and the drift reaches 12.28 units, meaning the rollout's colour balance
**trends in one direction across the ten seconds**. That is a visible defect: the scene
changes tone for no reason. It is the first genuine quality cost found.

So the honest reading of the 10s result is: **20 steps is 2.09× faster and the colour
drifts.** 30 steps is 1.53× faster with no monotone drift (R² 0.274) and is the
defensible choice on this evidence.

### But the drift signal is not reliable at one seed — and this weakens the conclusion

Running the same colour check on the 5-second sweep, where the trend should be at least
as clear, gives a **non-monotonic** result:

| Steps | Colour stability | Trend R² | Verdict |
|---|---|---|---|
| 50 | 1.92 | 0.463 | oscillation |
| 30 | 2.53 | **0.521** | **drift** |
| 12 | 1.58 | 0.412 | oscillation |

30 steps shows drift while 12 steps does not. If fewer steps caused colour drift, the
effect should get *worse* as steps fall, not better. It doesn't.

The R² values sit at 0.41–0.59, straddling the 0.5 threshold, so the verdict flips on
variation that is not attributable to step count. **The most likely explanation is that
this is seed noise, not a step-count effect.**

That means the "20 steps drifts colour" conclusion above is **not established**. It is
one generation at one seed, and the same test on the other horizon contradicts the
direction. The honest statement is:

> Colour stability rises as steps fall at both horizons. Whether that rise is drift or
> seed noise is unresolved, because the trend test gives opposite verdicts at the two
> horizons for the same step reduction.

Resolving it needs **seed replication** — the same step counts at 3–5 seeds each — which
was not run. Until then, 30 steps is recommended for a different reason: it is the
setting where *both* horizons show a speedup with no metric moving in a direction that
is clearly bad. That is a weaker justification than the table above implies.

### The two horizons behave differently, which was the open question

| | 5s | 10s |
|---|---|---|
| Speedup at 30 steps | 1.42× | 1.53× |
| Speedup at 20 steps | 1.84× | 2.09× |
| Colour drift at 20 steps | none detected | **R² 0.589, drift 12.28** |

The long horizon tolerates the step cut *better* on speed (more denoise to cut) but
*worse* on colour. That is exactly the asymmetry the plan said to look for, and it
means a single step setting for both horizons is the wrong design — the long one needs
more steps for colour stability even though it has more time to gain.

## What is not established

1. **One prompt, one seed.** Every number above is a single generation. These metrics
   vary between seeds, and the ±10% tolerance was chosen assuming a few percent of
   seed noise — not measured here.
2. **Prompt adherence is not measured at all.** A rollout could be sharper, more
   stable, and still show the wrong scene. None of the four metrics captures "did it
   render what was asked". This is the largest gap and the honest caveat on any
   "quality held" claim.
3. **No human comparison.** These are proxies. The only way to know whether 12 steps
   *looks* as good is to look.
4. **The 10-second horizon is still running.** The goal is the long sequence; nothing
   here says the same step cut is safe at 241 frames rather than 121.

## The recommendation

The goal — speed up the longest sequence without losing quality on the shorter ones —
is **met by step reduction alone**, with one caveat that decides the setting.

| Horizon | Recommended steps | Speedup | Why |
|---|---|---|---|
| 10s (the goal) | **30** | **1.53×** | 20 steps is faster but its colour verdict is unreliable |
| 5s | **30** | **1.42×** | same setting works; 12 steps is faster but untested on adherence |

**30 steps for both horizons** is the defensible configuration on this evidence: 1.42×
at 5s and 1.53× at 10s, sharper output at both, and no metric moving in a direction
that is clearly bad.

20 steps at 10s gives 2.09× and may be fine — the colour signal that argued against it
is not reliable at one seed. Treat 20 as *promising but unverified* rather than as
rejected.

### What this means for the WSP work

The original plan assumed a model edit would be needed to buy speed. It is not: **the
step cut already delivers the goal.** That changes what WSP is for.

WSP is now about *raising the floor* rather than achieving the first win — specifically,
whether a targeted edit can make 20 steps colour-stable, which would recover the 2.09×
without the drift. That is a much narrower and better-posed question than "make Ovi
faster", and it is the one worth spending GPU time on next.

It also means the honest answer to the original brief is: **the speed goal did not
require the neuroscience-inspired adapter.** The adapter's value, if any, is in
recovering quality at step counts where the base model loses it.
