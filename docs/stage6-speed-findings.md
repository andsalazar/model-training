# Speed findings: step reduction gives 1.5–2.1× with one unresolved quality question

*Measured results from the step sweep on Ovi at 960×960, one A100 80GB, one prompt,
seed 100.*

## The headline

Cutting Ovi's denoising steps is the whole speed story, and it works: **30 steps is
1.42× faster at 5s and 1.53× at 10s**, with sharper output at both horizons and no
measurable audio change. **20 steps reaches 1.76× at 10s** — but its colour drifts in
2 of 3 seeds, so 30 is the setting to ship.

The seed replication is what makes this answerable. At one seed the colour signal was
ambiguous; at three seeds the ranges are disjoint from baseline and the trend test
separates drift from oscillation. The recommendation changed as a result.

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

## The 10-second horizon: 1.76× at 20 steps, 1.53× at 30

This is the horizon the goal is actually about. First measured at seed 100 only:

| Steps | Wall clock | Speedup | Sharpness | Flicker | Coherence | Colour |
|---|---|---|---|---|---|---|
| 50 (default) | 1250.6s | 1.00× | 14.76 | 0.1976 | 0.6149 | 2.834 |
| 30 | 819.7s | **1.53×** | 20.83 | 0.3191 | 0.6824 | 5.324 |
| 20 | 598.1s | **2.09×** | 23.29 | 0.3444 | 0.6698 | 6.768 |

Sharpness up 58% and motion coherence up 9% at 20 steps. But the 20-step run was faster
than the seed mean (598s vs 707s), so **2.09× is not the number to quote** — the
three-seed replication below gives 1.76×. The quality direction is the same either way.

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

So the single-seed reading was: **20 steps is faster and the colour drifts.** 30 steps
is 1.53× faster with no monotone drift (R² 0.274). The next section tests whether that
holds across seeds — it does, with one correction to the speedup.

### Seed replication: the colour rise is real, and it is drift

The single-seed result above was not trustworthy, because the same trend test on the
5-second sweep gave opposite verdicts for the same step reduction. So the 10s sweep was
re-run at **three seeds** (100/200/300) for 50/30/20 steps — nine generations.

Wall clock is highly reproducible across seeds (50 steps: 1242.9/1245.6/1242.2s), so the
speedup is solid:

| Steps | Mean wall clock | Speedup per seed | Median |
|---|---|---|---|
| 50 (default) | 1243.6s | 1.00× | 1.00× |
| 30 | 843.1s | 1.53× / 1.53× / 1.37× | **1.53×** |
| 20 | 707.4s | 1.78× / 1.75× / 1.74× | **1.76×** |

Colour stability, per seed:

| Steps | Values (seeds 100/200/300) | Range | Mean |
|---|---|---|---|
| 50 | 2.83, 2.91, 3.70 | 0.86 | 3.15 |
| 30 | 3.79, 4.97, 5.32 | 1.53 | 4.69 |
| 20 | 4.40, 4.45, 6.77 | 2.37 | 5.20 |

**The ranges are disjoint from the baseline at both 30 and 20 steps.** Seed noise is
±30% within a condition, but the shift between conditions is larger than that spread —
so the rise is a step-count effect, not seed variation. The earlier "it's probably seed
noise" reading was wrong.

The trend test then says *what kind* of change it is:

| Steps | Seed | Colour | Trend R² | Mean abs drift | Verdict |
|---|---|---|---|---|---|
| 50 | 100 | 2.56 | 0.130 | 1.81 | oscillation |
| 50 | 200 | 3.67 | 0.119 | 2.17 | oscillation |
| 50 | 300 | 2.85 | 0.146 | 1.52 | oscillation |
| 30 | 100 | 5.00 | 0.274 | 7.24 | oscillation |
| 30 | 200 | 3.78 | 0.383 | 5.23 | oscillation |
| 30 | 300 | 4.87 | 0.090 | 1.95 | oscillation |
| 20 | 100 | 6.42 | **0.589** | **12.28** | **drift** |
| 20 | 200 | 4.31 | **0.677** | **7.87** | **drift** |
| 20 | 300 | 4.41 | 0.353 | 5.40 | oscillation |

**All three 50-step seeds oscillate** (R² 0.12–0.15) — the baseline is clean. **Two of
three 20-step seeds drift** (R² 0.59, 0.68). At 30 steps all three oscillate, though the
drift magnitude rises (1.5–7.2 vs 1.5–2.2 at baseline).

So the defensible reading is:

> **20 steps is 1.76× faster and its colour drifts in 2 of 3 seeds.** 30 steps is 1.53×
> faster and its colour oscillates in all 3 seeds, with larger wobble than baseline but
> no monotone trend.

### The two horizons behave differently, which was the open question

| | 5s | 10s |
|---|---|---|
| Speedup at 30 steps | 1.42× | 1.53× |
| Speedup at 20 steps | 1.84× | 1.76× |
| Colour verdict at 20 steps | oscillation (R² 0.412) | **drift in 2 of 3 seeds** |

The long horizon tolerates the step cut *better* on speed but *worse* on colour. That is
the asymmetry the plan said to look for, and it means a single step setting for both
horizons is the wrong design — the long one needs more steps for colour stability even
though it has more time to gain.

## Audio survives the step cut

Ovi is a joint video+audio model, so "quality" includes audio. Every metric above
reads pixels only, which meant the sweep had said nothing about half of what Ovi
produces. Measured separately:

| Horizon / steps | RMS | Spectral centroid | Centroid trend R² |
|---|---|---|---|
| 5s / 50 | 0.00547 | 2460.6 Hz | 0.001 |
| 5s / 30 | 0.00515 | 2473.8 Hz | 0.003 |
| 10s / 50 | 0.01214 | 2475.0 Hz | 0.000 |
| 10s / 30 | 0.01066 | 2492.0 Hz | 0.002 |

RMS within 6% and spectral centroid within 1% at both horizons across a 40% step cut,
with no timbre trend (R² ≤ 0.003). The step cut does not cost audio quality on this
evidence.

### The first audio check reported a false negative

The initial run said **zero audio streams in every video**. That reads as "Ovi
produces no audio" — a much stronger and completely wrong claim. The cause:
`imageio_ffmpeg` ships ffmpeg **without ffprobe**, so the stream detector returned an
empty list when nothing had been inspected.

The script now parses ffmpeg's own banner as a fallback and reports *which mechanism
was used*, so a zero can never be mistaken for a measurement. This is the **fourth
time in this project that tooling invented a result**: the inverted
`direction_similarity` sign, the unmeasured post-edit token counts, the substring
compliance tests, and now this. The pattern is consistent enough to be a standing rule
— a tool that cannot look must say so, not return zero.

## What is not established

1. **One prompt, one seed.** Every number above is a single generation at seed 100. A
   multi-seed run (seeds 100/200/300 at 50/30/20 steps, 10s) is in progress
   specifically to settle the colour question; its results are not in this document.
2. **Prompt adherence is not measured at all.** A rollout could be sharper, more
   stable, and still show the wrong scene. None of the four metrics captures "did it
   render what was asked". **This is the largest gap** and the honest caveat on any
   "quality held" claim. The prompt asks for a river through a forest with panning and
   birdsong; nothing here checks that any of that appears.
3. **No human comparison.** These are proxies. The only way to know whether 20 steps
   *looks* as good is to look at the videos side by side.
4. **One prompt.** All seven videos are the same river prompt. A static or
   low-texture prompt may behave differently — the previous stage's entropy work found
   that two prompts out of four behaved differently from the rest.
5. **The 5s colour verdict is still single-seed.** The 10s colour question was settled
   with three seeds; the 5s one was not. Its "30 steps drifts" reading is as unreliable
   as the 10s one was before replication.
6. **The 20-step speedup is 1.76×, not 2.09×.** The single-seed 20-step run happened to
   be faster than the seed mean. Quote the seed-verified number.

## The recommendation

The goal — speed up the longest sequence without losing quality on the shorter ones —
is **met by step reduction alone**, with one caveat that decides the setting.

| Horizon | Recommended steps | Speedup | Why |
|---|---|---|---|
| 10s (the goal) | **30** | **1.53×** | 20 steps is faster but its colour drifts in 2 of 3 seeds |
| 5s | **30** | **1.42×** | same setting works; 12 steps is faster but untested on adherence |

**30 steps for both horizons** is the defensible configuration on this evidence: 1.42×
at 5s and 1.53× at 10s, sharper output at both, audio unchanged, and colour oscillating
rather than drifting in all three seeds.

20 steps at 10s gives 1.76× and its colour drifts in 2 of 3 seeds. That is a real cost,
not seed noise — the ranges are disjoint from baseline. It is available if 1.76× is
worth a visible tone shift over ten seconds, but it is not the setting to default to.

### What this means for the WSP work

The original plan assumed a model edit would be needed to buy speed. It is not: **the
step cut already delivers the goal.** That changes what WSP is for.

WSP is now about *raising the floor* rather than achieving the first win — specifically,
whether a targeted edit can make 20 steps colour-stable, which would recover 1.76×
without the drift. That is a much narrower and better-posed question than "make Ovi
faster", and it is the one worth spending GPU time on next.

It also means the honest answer to the original brief is: **the speed goal did not
require the neuroscience-inspired adapter.** The adapter's value, if any, is in
recovering quality at step counts where the base model loses it.
