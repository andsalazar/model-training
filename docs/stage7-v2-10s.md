# Stage 7 v2 at 10 s: does the adapter carry over, and does 10 s training fix it?

*Driver: `scripts/pod_adapter_run.py --set v2_10s --model-name 960x960_10s`, pipeline
`scripts/pod_v2_10s.sh`. Teacher 48-step UniPC (986 s per 10 s clip), student 12 Euler
steps on the UniPC grid. Training trajectories: river ×2, fan, coast, fox, city,
waterfall, drummer; validation: market, rainforest; evaluation: river seeds
100/200/300.*

## 1. Zero-shot: the 5 s adapter on the 10 s model

The 5 s and 10 s checkpoints share an architecture, so the 5 s v2 adapter loads onto the
10 s model unchanged. Same-pod wall clock: 12 steps 283 s, 30 steps 630 s (**2.22×**).

| mean of 3 seeds | teacher 48 | UniPC 30 | UniPC 12 | Euler 12 | 5 s adapter, Euler 12 |
|---|---|---|---|---|---|
| sharpness | 14.06 | 18.56 | 22.38 | 26.96 | **2.94** |
| flicker | 0.224 | 0.304 | 0.268 | 0.235 | **0.087** |
| motion coherence | 0.662 | 0.688 | 0.567 | 0.477 | 0.805 |
| colour stability | 3.36 | 4.69 | 4.89 | 4.37 | 2.80 |
| **mean \|rel diff\| from teacher** | 0 | **0.317** | 0.365 | 0.452 | **0.505** |

**It does not transfer.** On the 10 s model the 5 s adapter **blurs** the video: sharpness
falls to a fifth of the teacher's and frame-to-frame change to under half. That is also
why motion coherence reads *above* the teacher's: a blurred, slowly changing frame is
trivially coherent. On the combined distance it is worse than every baseline.

Audio is mixed rather than broken. Its final audio latent is closer to the teacher than
Euler-12's on 2 of 3 seeds (0.100 / 0.172 / 0.130 vs 0.202 / 0.150 / 0.227), and
closer than UniPC-12's (0.141 / 0.110 / 0.096) on only one.

The end-of-clip brightness flash seen at 5 s does **not** appear here: the largest
frame-to-frame jump in any adapter clip is 2.06, within the range of the teacher's own
clips (0.68–3.60).

Two further observations:

- **The 12-step gap is smaller at 10 s.** UniPC-12 is 0.365 from the teacher here
  against 0.750 at 5 s, so the 10 s model loses less to step cutting on this prompt, and
  there is less for an adapter to win.
- **Teacher-forced loss did not see the blur.** At the start of 10 s training the
  5 s adapter's validation loss on 10 s trajectories was low (video 0.041, below the 5 s
  run's own final 0.045), yet its full rollouts blur. One-step error is small, and it
  compounds over 12 steps into a visible defect. Selecting the best epoch on this loss
  is therefore weak evidence; the rollout evaluation is the one to trust.

## 2. Trained at 10 s (from the 5 s adapter)

2 epochs over 96 samples, cosine 5e-4 → 5e-5, sequential CFG gradient (98 s per update,
58 GB), T5 offloaded. 5.6 h. Validation total 0.0577 → 0.0531 (epoch 0) → 0.0532
(epoch 1); epoch 0 selected. Same-pod wall clock unchanged: 281 s for 12 steps, **2.24×
faster than 30 steps** (630 s).

| river × 3 seeds | teacher 48 | UniPC 30 | UniPC 12 | 5 s adapter (zero-shot) | **10 s-trained adapter** |
|---|---|---|---|---|---|
| sharpness | 14.06 | 18.56 | 22.38 | 2.94 | 6.74 |
| flicker | 0.224 | 0.304 | 0.268 | 0.087 | 0.114 |
| motion coherence | 0.662 | 0.688 | 0.567 | 0.805 | 0.700 |
| colour stability | 3.36 | 4.69 | 4.89 | 2.80 | 2.70 |
| **video metric distance from teacher** | 0 | **0.317** | 0.365 | 0.505 | 0.340 |
| final video latent, rel. error (mean) | 0 | 0.734 | 0.828 | 0.760 | **0.666** |
| final audio latent, rel. error (mean) | 0 | **0.058** | 0.115 | 0.134 | 0.101 |
| audio loudness, mean \|rel diff\| (MP4) | 0 | 8.8% | 18.5% | — | **8.0%** |
| audio brightness, mean \|rel diff\| (MP4) | 0 | **0.8%** | 2.9% | — | 2.7% |

### Reading

- **Training at 10 s halves the zero-shot blur but does not remove it.** Sharpness is
  6.7 against the teacher's 14.1, and flicker stays low. On the four-metric distance the
  trained adapter (0.340) beats UniPC-12 (0.365) but not UniPC-30 (0.317).
- **Its final video latent is the closest to the teacher of any 12- or 30-step variant**
  (0.666 vs 0.734 for UniPC-30, better on 2 of 3 seeds). Together with the blur, this is
  the signature of an MSE-trained sampler: it lands near the *average* of the plausible
  outcomes, which is close in L2 and visibly soft. At 5 s the same objective did not blur
  (v2 adapter sharpness 13.0 vs teacher 11.6); at 10 s it does.
- **Audio loudness reaches 30-step parity** (8.0% vs 8.8%, UniPC-12 18.5%). Brightness is
  still not corrected (2.7% vs 0.8%), the same as at 5 s.
- **Validation loss was flat after epoch 0** and barely separated the epochs, consistent
  with the zero-shot finding that one-step loss does not see compounding rollout blur.

So at the target horizon: **2.24× faster than 30 steps, audio level as good as 30 steps,
video close in latent space but softer than 30 steps.** The blur is the defect to target
next. It is a known failure mode of pure-MSE step distillation, which suggests the fix is
in the objective (a sharpness- or distribution-matching term) as much as in placement.
WSP's job is to show whether the blur is written by specific blocks or spread across
them.
