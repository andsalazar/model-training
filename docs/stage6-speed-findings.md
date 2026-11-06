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

## The recommendation so far

For the 5-second horizon, **30 steps is the conservative choice** — 1.42× faster with
sharper output and only minor metric movement. **12 steps is the aggressive choice** —
2.34× faster, and the evidence says the extra high-frequency content is real detail
rather than noise, but prompt adherence is untested.

Do not lower the default to 12 on this evidence alone. Do run the 10-second sweep,
because if the long horizon tolerates a similar cut then the goal is met without any
model edit at all — and that makes the WSP work a question of raising the floor
further, not of achieving the first win.
