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
58 GB), T5 offloaded.

*Pending.*
