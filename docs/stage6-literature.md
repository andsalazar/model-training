# The literature behind TBC's adapter — and what survives contact with the primary source

*What neuroscience, ML and applied-math work plausibly informed TBC's "Neural Dynamics
Adapter", which parts of the working hypothesis the primary source actually supports, and
what that leaves us to do.*

Companion to [stage6-scoping.md](stage6-scoping.md), which establishes what TBC built and
what is open. This document covers the *intellectual lineage* question.

## Method and its limits

Two research threads were run: one over the neuroscience literature, one over the ML and
applied-math literature. Every arXiv citation was fetched and checked against its live
`arxiv.org/abs/...` page rather than taken from a search summary. Two citation errors were
caught this way and are flagged below rather than silently corrected. TBC's post was read
end to end, twice, in rendered and raw HTML.

**The single most important limit: TBC's post has no bibliography, footnotes or named
citations of any kind.** Confirmed by full read. So *every* link between a named paper and
TBC's engineering choices is inference. Each topic below is graded:

- **(a) fact** — verified against a primary source.
- **(b) reasonable inference** — well-motivated but not stated by TBC.
- **(c) speculation** — plausible framing, no evidence.

This grading matters because the project's own history is a record of what happens when an
unverified claim is treated as measured. Three times in Stages 4–5 a substring test invented
a success. The same discipline applies to literature: a citation that *sounds* like it
supports a hypothesis is not evidence that it does.

## Ground truth from the primary source

Load-bearing facts, all directly quoted or paraphrased from
<https://www.tbc.co/post/the-neural-dynamics-adapter>:

| Claim | Detail |
|---|---|
| Base model | **Oasis**, "the open-source 600M-parameter Oasis Minecraft world generation model" |
| Architecture | DiT backbone + ViT spatial autoencoder + action conditioning + diffusion forcing |
| Failure being fixed | long-horizon rollout degradation — "information collapse" over extended autoregressive rollout |
| Adapter | "shallow, lightweight module that modifies **residual information in early DiT layers**", ~**156k parameters** |
| Neuron role | *source* of the transformation (4096-electrode MEAs), not a runtime component |
| Deployment | entirely in silico; fixed transformation at inference |
| Metric | **differential entropy across generated frames**, as a proxy for information collapse |
| Results | ~19% over base Oasis, ~15% over parameter-matched fine-tuning, ~5% over LoRA |
| Stated rationale | biological networks "naturally produce stable representations over extended temporal dynamics" |
| Published | Tue Nov 03 2026 (from page metadata) |

Two things this table settles immediately, both of which contradict the original brief:

1. **TBC used Oasis, not Wan 2.2.** Wan 2.2 is never mentioned in the post. The adapter, the
   fine-tuning baseline and the LoRA baseline were all applied to Oasis.
2. **TBC's goal was stability, not speed.** The metric is a quality/stability axis. Nothing in
   the post discusses latency, FLOPs or step-count reduction as an objective.

The brief's premise — "TBC did their LoRA on Wan 2.2 to make Mischief 1, penalising weights
that fired needlessly when pixels were similar frame to frame, exploiting Wan 2.2's MoE to
expedite processing" — is wrong on the model, wrong on the mechanism (it is not a LoRA), and
wrong on the goal. That correction is already recorded in
[stage6-scoping.md](stage6-scoping.md); what follows is what the literature says *instead*.

## The neuroscience lineage

### 1. Repetition suppression / neural adaptation — **(b)**

Neurons across visual cortex reduce firing when a stimulus is repeated relative to when it is
novel. This is the textbook phenomenon closest to the brief's "neurons go quiet when seeing the
same thing".

- Grill-Spector, Henson & Martin (2006), "Repetition and the brain: neural models of
  stimulus-specific effects", *Trends in Cognitive Sciences* 10(1):14–23. **Verified.** Frames
  three competing models: **fatigue** (generic response-proportional reduction), **sharpening**
  (the responding population narrows to fewer, more selective units), and **facilitation**
  (processing gets faster — repetition priming — with response *duration* falling even when peak
  amplitude does not).
- Desimone (1996), *PNAS* 93:13494–13499. **Verified.** Inferior-temporal repetition/familiarity
  effects.
- Krekelberg, Boynton & van Wezel (2006), "Adaptation: from single cells to BOLD signals",
  *Trends in Neurosciences* 29(5):250–256. **Verified.** Bridges single-unit adaptation to fMRI
  repetition suppression.
- Summerfield, Trittschuh, Monti, Mesulam & Egner (2008), "Neural repetition suppression reflects
  fulfilled perceptual expectations", *Nature Neuroscience* 11(9):1004–1006. **Verified.** Adds a
  fourth account: suppression tracks *fulfilled prediction*, not mere repetition — and is reduced
  when a repetition is statistically improbable.

⚠️ **Citation error flagged:** "Miller & Desimone (1994), *Annual Review of Neuroscience*" does
not appear to exist as specified. It likely conflates Miller & Desimone (1994), *Science*
263:520–522 with Desimone & Duncan (1995), *Annual Review of Neuroscience* 18:193–222. Both are
real; the combination is not.

**Why this is only (b):** the mechanistic debate is unresolved — no single model is "the"
explanation. And TBC's substrate is a *dissociated culture on an electrode array*, not intact IT
cortex receiving repeated naturalistic images. The general principle transfers; the specific
paradigm does not.

### 2. Predictive coding — **(b), with a terminology trap**

Hierarchical sensory cortex may encode the *difference* between input and a top-down prediction
rather than the raw input. Well-predicted input produces a small residual error signal — which
makes repetition suppression a special case of reduced prediction error.

- Rao & Ballard (1999), "Predictive coding in the visual cortex", *Nature Neuroscience* 2:79–87.
  **Verified.** The canonical hierarchical model: predictions down, residual errors up.
- Friston (2010), "The free-energy principle: a unified brain theory?", *Nature Reviews
  Neuroscience* 11(2):127–138. **Verified.**
- Clark (2013), "Whatever next? Predictive brains, situated agents, and the future of cognitive
  science", *Behavioral and Brain Sciences* 36(3):181–204. **Verified.**
- Auksztulewicz & Friston (2016), "Repetition suppression and its contextual determinants in
  predictive coding", *Cortex* 80:125–140. **Verified.** The precise match for the RS↔PC linkage.
  (Spratling 2017, *Brain and Cognition* 112:92–97, is real but is a general algorithms review,
  not RS-specific — a weaker fit than the framing implied.)

⚠️ **The trap:** TBC's adapter "modifies **residual** information in early DiT layers", and
"residual" is exactly the word predictive coding uses for the forward error signal. This is
almost certainly **coincidence**. "Residual stream" is independently-coined transformer
mechanistic-interpretability jargon (Elhage et al. 2021) meaning the additive skip-connection
pathway — it implies no prediction-error computation. TBC's post shows no top-down/bottom-up
error units, no precision weighting, no explicit generative model. The vocabulary resonance is
seductive and should be resisted.

### 3. Efficient / sparse coding — **(c) for the adapter, (b) for the company**

Sensory systems are shaped to represent natural signals with minimal redundancy and few active
units — statistically efficient and metabolically cheap.

- Attneave (1954), "Some informational aspects of visual perception", *Psychological Review*
  61(3):183–193. **Verified.** Earliest clear information-theoretic redundancy argument.
- Barlow (1961), "Possible principles underlying the transformation of sensory messages", in
  *Sensory Communication* (MIT Press), pp. 217–234. **Verified** — note this is a **book chapter**,
  not a journal article.
- Olshausen & Field (1996), "Emergence of simple-cell receptive field properties by learning a
  sparse code for natural images", *Nature* 381:607–609. **Verified.**
- Simoncelli & Olshausen (2001), "Natural image statistics and neural representation", *Annual
  Review of Neuroscience* 24:1193–1216. **Verified.**

**Grading:** TBC's company-level messaging does foreground brain energy efficiency versus
silicon. But *this post* frames the adapter's benefit entirely as rollout stability, and gives no
architectural detail — no sparsity penalty, no dictionary learning — suggesting sparse coding was
implemented in the 156k-parameter module. Reading the adapter as an Olshausen–Field implementation
would be overreach.

### 4. Temporal redundancy and motion discounting — **(c), weakest fit**

The visual system discounts self-generated sensory consequences using an efference copy; early
visual neurons are transient/change-detecting rather than sustained level detectors.

- von Holst & Mittelstaedt (1950), "Das Reafferenzprinzip", *Naturwissenschaften* 37(20):464–476.
  **Verified** (German original). Efference copy: predicted self-caused feedback is cancelled, a
  mismatch signals external change.
- Enroth-Cugell & Robson (1966), "The contrast sensitivity of retinal ganglion cells of the cat",
  *Journal of Physiology* 187(3):517–552. **Verified.** X-cell (linear) vs Y-cell (nonlinear)
  spatial summation.
- Cleland, Dubin & Levick (1971), "Sustained and transient neurones in the cat's retina and
  lateral geniculate nucleus", *Journal of Physiology* 217:473–496. **Verified.** A closer literal
  match to "transient rather than sustained" than the X/Y classification, which is about spatial
  summation, not time course.

⚠️ **Two flags.** "Grosso & Gerbino" could not be verified under any formulation tried — reported
as unconfirmed rather than invented. And Wallach (1959), "The perception of motion", *Scientific
American* 201(1):56–60, is real but is primarily about *induced motion* (the Wallach effect), a
related-but-distinct phenomenon; it should not be over-read as supporting eye-movement motion
discounting.

**Why this is the weakest topic:** the reafference principle requires a *self-generated* motor
efference copy cancelling a specific predicted consequence. A dissociated culture on an MEA
receiving externally-designed electrical stimulation has no self-movement to discount, and TBC
describes no closed sensorimotor loop. The analogy can be constructed but there is no evidence it
was built.

### 5. MEA neuroscience — **(a), the only topic verifiable at the fact level**

High-density CMOS MEAs record spikes, LFPs and burst/network dynamics from cultured tissue at
sub-cellular resolution. This is a mature, commercialised hardware lineage.

- Hierlemann, Frey, Hafizovic & Heer (2011), "Growing cells atop microelectronic chips",
  *Proceedings of the IEEE* 99(2):252–284. **Verified.**
- Frey et al., "Switch-matrix-based high-density microelectrode array in CMOS technology", *IEEE
  Journal of Solid-State Circuits* 45(2):467–482. **Verified as real, but dated 2010**, not 2009 as
  the framing suggested. A 2009 ISSCC precursor may exist; the journal paper verified is 2010.
- Obien, Deligkaris, Bullmann, Bakkum & Frey (2015), "Revealing neuronal function through
  microelectrode array recordings", *Frontiers in Neuroscience* 8:423. **Verified.** (DOI carries a
  2014 online-publication stamp; the citable issue is 2015 — a common ambiguity, not an error.)
- Ballini et al. (2014), "A 1024-Channel CMOS Microelectrode Array With 26,400 Electrodes...",
  *IEEE Journal of Solid-State Circuits* 49(11):2705–2719. **Verified as real, with a title
  correction**: this is an *electrophysiological* array, not a "neurochemical microarray" as the
  framing implied. No Ballini 2014 neurochemical-sensing paper was found.

**The exact numerical match:** **3Brain's BioCAM** is a CMOS HD-MEA with **exactly 4,096 electrodes
in a 64×64 grid** — matching TBC's stated count precisely. The ETH Zürich lineage (Hierlemann /
Frey / Bakkum) was commercialised as **MaxWell Biosystems** (MaxOne: 26,400 electrodes, 1,020
simultaneous channels). Given "4,096 electrodes", "CMOS-based" and "localized spatial activations"
are all idiomatic to this specific tradition, a commercial HD-MEA in this lineage is a well-grounded
inference — though TBC names no vendor.

**The genuine gap:** TBC explicitly hedges that the adapter may be built from "computational models
derived from those responses" rather than raw recordings, and names adapters "derived directly from
measured biological activity" as *future* work. So **which signal class** (sorted spikes, LFP,
multi-unit activity, burst statistics) actually became the 156k parameters is unknown. That is an
acknowledged gap, not something to infer past.

### 6. Criticality / avalanche dynamics — **(b), strongest conceptual fit**

Spontaneous cortical activity organises into "avalanches" whose sizes and durations follow power
laws — consistent with operating near a critical point theorised to optimise information
transmission and representational stability over time.

- Beggs & Plenz (2003), "Neuronal avalanches in neocortical circuits", *Journal of Neuroscience*
  23(35):11167–11177. **Verified**, including method: **organotypic and acute slices** of rat
  somatosensory cortex on an **8×8 (60-channel)** array.

**Why this is the strongest conceptual fit:** TBC's stated rationale — biological networks
"naturally produce stable representations over extended temporal dynamics" — is close to a
paraphrase of what criticality theory claims about critical networks (stable, scale-invariant
dynamics, avoiding decay to silence or explosion to saturation over long timescales). And Beggs &
Plenz's paradigm — living cortical tissue on an MEA, examined for emergent network dynamics — is
structurally the same kind of system.

**The temper:** Beggs & Plenz used *laminar* organotypic/slice cultures on a 60-channel array; TBC
almost certainly used *dissociated* cultures on a much higher-density commercial array. Some reviews
report criticality signatures are less consistently observed in dissociated cultures. That sub-debate
was not resolved in this research and is flagged as open. TBC also never uses the words "criticality",
"avalanche" or "self-organized" anywhere in the post.

### 7. Short-term synaptic depression — **(b), most literal mechanism**

Repeated rapid presynaptic firing progressively decreases release probability, acting as a fast,
local, automatic high-pass filter on repeated input — adaptive gain control at the synapse.

- Tsodyks & Markram (1997), "The neural code between neocortical pyramidal neurons depends on
  neurotransmitter release probability", *PNAS* 94(2):719–723. **Verified.**
- Abbott, Varela, Sen & Nelson (1997), "Synaptic depression and cortical gain control", *Science*
  275(5297):220–224. **Verified.**

**The distinctive point:** this is the one mechanism on the list that requires **no engineering
choice by TBC at all**. If the cultures contain real living neurons and real synapses — which they
do by the premise of the product — short-term depression is present by biophysical necessity,
whether or not anyone targeted, measured or thought about it. It is simultaneously the most literal
match to "quiet when repeated" and the least "designed-for" connection here.

### 8. Energy constraints — **(b) for the company, (c) for the adapter**

Neural signalling is metabolically expensive, and this is argued to be a major constraint shaping
codes toward sparsity.

- Attwell & Laughlin (2001), "An energy budget for signaling in the grey matter of the brain",
  *Journal of Cerebral Blood Flow & Metabolism* 21(10):1133–1145. **Verified.**
- Laughlin, de Ruyter van Steveninck & Anderson (1998), "The metabolic cost of neural information",
  *Nature Neuroscience* 1(1):36–41. **Verified.**
- Niven & Laughlin (2008), "Energy limitation as a selective pressure on the evolution of sensory
  systems", *Journal of Experimental Biology* 211(11):1792–1804. **Verified.**

**The distinction to preserve:** TBC's company messaging foregrounds brain energy efficiency. But
*this post* does not frame the adapter's benefit in energy or compute terms anywhere. Its metric is
rollout stability — a representational-quality claim. Attributing this specific result to the energy
literature would overstate it.

## The ML and applied-math lineage

### 1. Diffusion forcing and Oasis — **(a), TBC states it directly**

This is the single most solid literature-to-TBC link in the whole review, because TBC says it
themselves.

- Chen, Monso, Du, Simchowitz, Tedrake & Sitzmann, "Diffusion Forcing: Next-token Prediction Meets
  Full-Sequence Diffusion", NeurIPS 2024, **arXiv:2407.01392**. **Verified.** Trains a model "to
  denoise a set of tokens with independent per-token noise levels", enabling "rolling-out sequences
  of continuous tokens, such as video, with lengths past the training horizon, **where baselines
  diverge**."
- **Mechanism:** in standard autoregressive video rollout each frame conditions on previously
  *generated* (imperfect) frames, so errors compound geometrically — exposure bias. Diffusion
  forcing lets every token sit at its own independently-sampled noise level, so a frame can be
  re-denoised while later frames condition on it. The model learns to treat past context as
  "somewhat noisy" rather than ground truth.
- **Oasis:** no accessible peer-reviewed or arXiv paper exists. The primary fetchable source is the
  joint project blog (`oasis-model.github.io`, Decart/Etched, Dec 2024), which states verbatim: "The
  model was trained using Diffusion Forcing, which denoises with independent per-token noise levels",
  and describes "a spatial autoencoder [ViT-based] and a latent diffusion backbone [DiT-based]" —
  confirming TBC's "DiT layers" framing is architecturally accurate.

⚠️ **Unresolved discrepancy:** the documented open release is **Oasis 500M** (`Etched/oasis-500m`;
the blog says "weights of a 500M parameter model... and a live demo of a larger checkpoint"). TBC
says **600M**. Not resolved — could be TBC imprecision, a different checkpoint, or a counting
convention. Flagged, not asserted either way.

⚠️ **Weak corroboration:** the Oasis blog author list (Campbell, Chen, Quevedo, McIntyre) comes only
from search aggregation; the OpenReview page blocked access twice and a guessed Decart URL 404'd.
Moderate confidence only.

### 2. Temporal redundancy in video diffusion — **(b)**

What actually reduces compute from redundancy in video DiTs, concretely:

- Ma et al., "Latte: Latent Diffusion Transformer for Video Generation", TMLR 2025,
  **arXiv:2401.03048**. **Verified.** Decomposes spatial/temporal attention into four efficient
  variants to handle token blow-up while modelling spatio-temporal redundancy.
- Zheng et al., "Open-Sora: Democratizing Efficient Video Production for All", **arXiv:2412.20404**.
  **Verified.** **STDiT** decouples spatial and temporal attention for compute efficiency, plus a
  highly compressive 3D autoencoder.
- Yang et al., "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer", ICLR 2025,
  **arXiv:2408.06072**. **Verified.** 3D causal VAE (joint spatial+temporal compression). Note the
  "expert transformer" is an expert *adaptive LayerNorm* for text-video fusion — **not** a routed MoE.
- Wan 2.1 report, "Wan: Open and Advanced Large-Scale Video Generative Models", **arXiv:2503.20314**.
  **Verified.** Dense 1.3B/14B only, no MoE.

**The three real levers:** (i) 3D/spatio-temporal VAEs that compress the redundant temporal axis
before the backbone sees it; (ii) decoupled/factorised spatial-temporal attention so full O(n²) is
not paid across the whole space-time volume; (iii) caching across denoising steps (below).

**Grading:** these are the standard "exploit redundancy for speed" techniques. If anyone wanted to
dampen parameters carrying similar pixels frame-to-frame, the natural literature analogues are
temporal VAE compression and attention factorisation — **not LoRA**. TBC's post discusses none of
this. It is evidence about what techniques exist in this space, not about TBC.

### 3. Caching and latent reuse — **(b), and the closest analogue to "quiet when nothing changes"**

- Ma, Fang & Wang, "DeepCache: Accelerating Diffusion Models for Free", **arXiv:2312.00858**.
  **Verified.** Caches and reuses high-level U-Net features across *adjacent denoising steps*,
  cheaply recomputing only low-level features. Reported **2.3×** speedup on SD v1.5 (CLIP −0.05) and
  **4.1×** on LDM-4-G (FID +0.22).

⚠️ **The critical distinction:** the redundancy being exploited is across diffusion **timesteps**,
not across video **frames**. Consecutive denoising iterations of the *same* frame look similar to
each other; frame *N* and frame *N+1* of a video do not. The brief's "pixels similar from frame to
frame" is a different axis from what DeepCache exploits. This matters for the design.

**Real-world confirmation on Wan specifically:** the Wan2.2 GitHub README (fetched directly) states
"**Cache-dit** offers Fully Cache Acceleration support for **Wan2.2 MoE** with DBCache, TaylorSeer
and Cache CFG" — a third-party community project, not TBC's or the Wan team's.

### 4. LoRA and parameter-efficient editing: suppress vs add — **(b), and the mechanism is real**

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", **arXiv:2106.09685**. **Verified.**
  The original framing is purely *additive/efficiency*: freeze base weights, inject trainable ΔW = BA.
  Nothing about suppression.
- Biderman et al., "LoRA Learns Less and Forgets Less", TMLR Aug 2024, **arXiv:2405.09673**.
  **Verified.** LoRA "substantially underperforms full finetuning" at learning new behaviour, but
  "better maintains the base model's performance on tasks outside the target domain" and "helps
  maintain more diverse generations". Full fine-tuning's effective perturbation rank is 10–100×
  higher than typical LoRA ranks. **This cuts both ways**: the same capacity constraint that limits
  LoRA's ability to *add* behaviour is why it is gentler at *removing* it.
- Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024 Oral, **arXiv:2402.09353**.
  **Verified.** Decomposes weights into magnitude + direction, applies LoRA only to direction.
- **Gandikota, Materzynska, Zhou, Torralba & Bau, "Concept Sliders: LoRA Adaptors for Precise Control
  in Diffusion Models", ECCV 2024, arXiv:2311.12092. Verified.** Finds "a low-rank parameter direction
  corresponding to one concept while minimizing interference with other attributes"; sliders dial
  *down* (suppress) or *up* (amplify) continuously, explicitly including repairing/suppressing
  artifacts. **This is the cleanest evidence that LoRA-type low-rank directions can target
  *suppression* of an existing visual phenomenon, not just addition.**
- Gandikota, Materzyńska, Fiotto-Kaufman & Bau, "Erasing Concepts from Diffusion Models", ICCV 2023,
  **arXiv:2303.07345**. **Verified.** Fine-tuning method that permanently erases a visual concept.
- Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023, **arXiv:2212.04089**. **Verified.**
  *Negating* a task vector "decreases performance on the target task, with little change in model
  behavior on control tasks" — a clean example that small directional weight-space edits can
  selectively *subtract* a behaviour.

**What this settles:** the brief's *mechanistic* idea — dampen a specific existing behaviour via a
small parameter edit — has genuine, citable precedent. It is simply not what TBC's post describes
doing, and not what Wan 2.2's architecture would support.

### 5. MoE in diffusion transformers — **(a), and it kills the MoE part of the hypothesis**

- Fedus, Zoph & Shazeer, "Switch Transformers", JMLR, **arXiv:2101.03961**. **Verified.** Canonical
  sparse MoE: "selects different parameters **for each incoming example**" — per-token routing.
- Fei et al., "Scaling Diffusion Transformers to 16 Billion Parameters" (DiT-MoE),
  **arXiv:2407.11633**. **Verified.** Confirmed per-token behaviour: "Expert selection shows
  preference with spatial position and denoising time step", shifting "from specific spatial position
  to dispersion and balance" with depth — true sparse per-token/per-patch routing.
- Sun et al., "EC-DiT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing",
  **arXiv:2410.02098**. **Verified.** Also genuine per-token sparse expert-choice routing.

⚠️ **Citation correction:** the ID 2405.05465 belongs to a different paper (a serving-systems paper,
"Vidur"), **not** DiT-MoE. The correct ID is **2407.11633**.

⚠️ **Citation correction:** the Wan2.2 README links **arXiv:2503.20314** as "the Paper". That ID is
the **Wan 2.1** report — dense 1.3B/14B, no MoE. **There is no dedicated arXiv or peer-reviewed
technical report for Wan 2.2's MoE design.** It is documented only in the GitHub README/project blog.

**Wan 2.2's actual architecture, verified from the primary README:** "Wan2.2 introduces a
Mixture-of-Experts (MoE) architecture into video diffusion models. By **separating the denoising
process across timesteps** with specialized powerful expert models, this enlarges the overall model
capacity while maintaining the same computational cost." Model names confirm it: **T2V-A14B /
I2V-A14B** ("A14B" = ~14B *activated* parameters per step), with the high-noise and low-noise halves
of the trajectory routed to two separate **complete dense expert networks** via an SNR/timestep
threshold. Only one expert is active per denoising step, chosen by *which timestep it is*, not by
token content.

**So Wan 2.2's "MoE" is a timestep-split dense two-expert ensemble**, fundamentally different from
Switch/DiT-MoE/EC-DiT's sparse per-token routing. There is no per-token "redundant pixel" routing
decision to exploit. The hypothesis's mechanism does not map onto Wan 2.2's real architecture even
in principle — independent of the fact that TBC never used Wan 2.2.

### 6. Differential entropy as an information-collapse metric — **(c), TBC's own choice**

- Differential entropy, h(X) = −∫p(x)log p(x)dx, is the standard continuous extension of Shannon
  entropy. Textbook material.
- Tishby, Pereira & Bialek, "The information bottleneck method", **arXiv:physics/0004057**.
  **Verified.** Conventionally cited as the 1999 Allerton paper though the arXiv submission is dated
  April 2000. Establishes entropy/mutual-information as the vocabulary for representation
  compression — an *optimisation objective*, not a diffusion collapse diagnostic.
- Goodfellow et al., "Generative Adversarial Networks", **arXiv:1406.2661**. **Verified.** The
  framework within which "mode collapse" became a named failure mode. The term does not appear
  verbatim in the 2014 abstract; this is a canonical ancestor, not the coining paper.

**After two targeted searches, no canonical, widely-cited paper establishing "differential entropy
across generated frames" as a standard diffusion/video collapse metric was found.** Adjacent work
uses LPIPS/distance-based diversity metrics instead. Differential entropy *is* a long-standing
standard tool in computational neuroscience for measuring the richness of continuous-valued neural
population activity, with "lower entropy suggests more stereotyped or redundant encoding" being
strikingly close to TBC's "information collapse" language — but that could not be pinned to one
founding paper in this session.

**Most defensible conclusion:** TBC's use of differential entropy as an information-collapse proxy
looks like **their own methodological choice**, assembled from general information theory and
neuroscience convention rather than citing an established diffusion-specific benchmark. Absence of a
found paper is evidence of absence only to the extent of the search effort — not proof.

### 7. Residual-stream editing and interpretability — **(b), methodological similarity only**

- Elhage et al. (Anthropic), "A Mathematical Framework for Transformer Circuits", Transformer
  Circuits Thread, 2021. **Verified** (not arXiv). Contains an explicit section, "Virtual Weights and
  the Residual Stream as a Communication Channel", framing the residual stream as shared additive
  bandwidth every layer reads from and writes to — the conceptual basis for "a layer can write a
  small targeted correction without changing how other layers read/write".
- Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency",
  **arXiv:2310.01405**. **Verified** (21 authors incl. Zou, Phan, Hendrycks). Population-level
  residual-stream directions that causally control high-level behaviours when added back at inference.
- Li et al., "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model", NeurIPS
  2023 spotlight, **arXiv:2306.03341**. **Verified.** Shifts activations along a few attention-head
  directions at inference; "minimally invasive and computationally inexpensive"; LLaMA/Alpaca
  truthfulness 32.5%→65.1%.
- Todd et al., "Function Vectors in Large Language Models", ICLR 2024, **arXiv:2310.15213**.
  **Verified.** A compact causal vector from a few attention heads can trigger an entire learned
  input-output function when added into the residual stream elsewhere.

**Grading:** this is the standard toolkit for "find where a behaviour lives in the residual stream,
then intervene there" — exactly the surgical-targeting framing. TBC's architecture (small parameter
count, residual stream, specifically *early* layers) is methodologically consistent with this
tradition. TBC cites none of it, so the link is inferred from similarity, not stated lineage.

### 8. Weight-Space Programming — **(a), and the timing argument**

Verified directly against the primary source (`rikiyat.github.io/blog/weight-space-programming/`,
fetched in full):

- Title: "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights."
- Authors: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab — MIT CSAIL-affiliated.
- **A research blog post, not a peer-reviewed paper, and not on arXiv.** No arXiv ID exists.
- Published **Nov 2, 2026**.
- Core claims: a frontier model reads and edits raw weight tensors as data — identifying a game from
  its weights, fixing a "car jump" behaviour, and editing **0.016% of a Qwen-class model's weights**
  to raise MATH-500 from 7.9%→48.1%.

**The timing argument:** TBC's post is dated **Nov 3, 2026 — one day after** WSP's Nov 2 publication.
TBC's work (training a 156k-param adapter, running full-fine-tuning and LoRA baselines, computing
differential-entropy statistics across rollouts) plausibly took weeks. A one-day gap makes WSP being
foundational to TBC's design very unlikely. This is a fact-based argument, not proof — TBC's team
could have known the direction earlier via a preprint, talk or shared lab context. TBC's post does
not mention WSP or anything resembling weight-reading/writing.

Related weight-space editing literature, all verified: Ilharco et al. **arXiv:2212.04089** (Task
Arithmetic); Meng, Bau, Andonian & Belinkov, "Locating and Editing Factual Associations in GPT"
(ROME), NeurIPS 2022, **arXiv:2202.05262**; Meng, Sharma, Andonian, Belinkov & Bau, "Mass-Editing
Memory in a Transformer" (MEMIT), ICLR 2023, **arXiv:2210.07229**.

### 9. Video generation speed — **(b), and TBC's goal was not speed**

- Sauer, Lorenz, Blattmann & Rombach, "Adversarial Diffusion Distillation", **arXiv:2311.17042**.
  **Verified.** Distils to 1–4 steps via adversarial loss + score distillation.
- Luo et al., "Latent Consistency Models", **arXiv:2310.04378**. **Verified.** 2–4-step sampling; a
  768×768 LCM took 32 A100-hours to train.
- **Applied to Wan by third parties** (from the Wan2.2 README, fetched directly): **LightX2V**
  (Wan-based step-distillation models), **FastVideo** (distilled Wan with sparse attention),
  **Cache-dit** (caching for Wan2.2). Real, active techniques — but not evidence of what TBC did.

**What dominates video-DiT inference cost:** (1) denoising step count — the primary lever for
ADD/LCM-style distillation, cutting 20–50 steps to 1–4; (2) sequence length, scaling with
frame-count × spatial-tokens-per-frame, hence VAE temporal compression and attention factorisation;
(3) resolution, quadratic in spatial tokens. TBC's own "Limitations" section mentions DDIM step
counts of 10–20 as typical for their Oasis experiments — standard practice, not a distillation method
of their own.

## Verdict on the working hypothesis

> *TBC applied LoRA to Wan 2.2 to dampen parameters carrying similar video pixels frame-to-frame, to
> speed up generation, possibly exploiting Wan 2.2's MoE.*

**Not supported — contradicted by the primary source on every load-bearing point.**

| Claim | Verdict | Basis |
|---|---|---|
| TBC used Wan 2.2 | **False** | Post names **Oasis** throughout; Wan 2.2 never mentioned |
| TBC used LoRA | **False** | Adapter is a 156k-param residual module; LoRA is only a *baseline* |
| Goal was speed | **False** | Goal is rollout stability; metric is differential entropy |
| Dampened frame-to-frame redundancy | **Not stated** | No mechanism described; "localized spatial activations" is the only hint |
| Exploited Wan 2.2's MoE | **False, twice over** | TBC never used Wan 2.2; and Wan 2.2's MoE is timestep-split, not per-token |

**What *is* real in the hypothesis:** the underlying mechanism — a small low-rank or small-magnitude
weight-space direction used to *suppress* a specific existing behaviour — has solid precedent
(Concept Sliders, Task Arithmetic negation). The brief's instinct about the *kind* of edit is sound.
It is the attribution to TBC and to Wan 2.2 that is wrong.

**Most likely explanation for the confusion:** conflation with a real but unrelated ecosystem of
Wan-2.2-specific caching/distillation/LoRA tools (LightX2V, FastVideo, Cache-dit, DiffSynth-Studio's
Wan 2.2 LoRA training support) that genuinely exists but has no documented connection to TBC.

## What this changes for the project

The literature review does not rescue the original premise, but it does sharpen the plan in four
ways.

### 1. The behaviour to target is stability, not speed

The brief wanted speed. TBC's actual result is stability — preventing information collapse over long
rollouts. These are different objectives with different metrics. If we target speed we are not
reproducing TBC's method; we are doing something adjacent. **This should be an explicit decision, not
a drift.** The honest framing: TBC's *architecture* (small residual module, early layers) is
transferable; TBC's *behaviour* (rollout stability) is what the architecture was built for.

### 2. The "frame-to-frame redundancy" axis is the wrong one for the caching literature

DeepCache and its descendants exploit redundancy across **denoising timesteps**, not across video
frames. If the goal is "quiet when nothing changes", the timestep axis is where the established
speedups live (2–4×). The frame axis is where TBC's stability problem lives. Conflating them would
repeat the project's recurring error — measuring the wrong thing.

### 3. Wan 2.2's MoE offers nothing to exploit per-token

It is a timestep-split dense ensemble. There is no router making per-token decisions. Any plan that
depends on "using the MoE's routing to expedite processing" is built on a misreading of the
architecture. Note also that **Ovi is built on Wan2.2-TI2V-5B**, so this applies to our actual target
model too.

### 4. WSP's role is narrower and clearer than the brief assumed

WSP cannot read Mischief 1 (API only, no weights). It *can* read Ovi. And its job is not "apply the
TBC method" — it is to answer TBC's unanswered **placement question**: which early DiT layers and
which residual channels. That is a measurement task, which is what Stages 1–4 built. The instrument
transfers even though the behaviour does not.

## Open questions this review could not settle

1. **Which signal class TBC actually used** — spikes, LFP, multi-unit activity or burst statistics.
   TBC's own hedge ("computational models derived from those responses") prevents any confident claim.
2. **Oasis 500M vs TBC's 600M** — unresolved; possibly a different or larger checkpoint.
3. **Whether criticality signatures hold in dissociated cultures** — the sub-debate was not resolved.
   Beggs & Plenz used laminar slices on 60 channels; TBC likely used dissociated cultures on a
   high-density array.
4. **Whether differential entropy separates a degrading rollout from a healthy one on Ovi** — nothing
   has been measured on video yet. This remains the cheapest decisive experiment, per
   [stage6-scoping.md](stage6-scoping.md).
5. **Whether the audio branch can be edited the same way** — different backbone, different residual
   structure. Unmeasured.
6. **Cross-modal fusion interaction** — Ovi's video and audio branches are not independent; an edit to
   one may propagate. Unmeasured, and the most novel part of the brief.

## Citation corrections found during verification

| Given / assumed | Problem | Correct |
|---|---|---|
| Miller & Desimone (1994), *Annual Review of Neuroscience* | No such paper found | Miller & Desimone (1994) *Science* 263:520–522 **and** Desimone & Duncan (1995) *Annu. Rev. Neurosci.* 18:193–222 |
| Grosso & Gerbino | Unverifiable under any formulation | Reported unconfirmed; no paper asserted |
| Ballini et al. (2014) "neurochemical microarray" | It is an *electrophysiological* array | Title corrected to the electrical recording/stimulation paper |
| Frey et al. switch-matrix MEA, 2009 | Journal paper is 2010 | *IEEE JSSC* 45(2):467–482, 2010 |
| DiT-MoE = arXiv:2405.05465 | That ID is "Vidur", a serving-systems paper | **arXiv:2407.11633** |
| "Wan 2.2 paper" = arXiv:2503.20314 | That ID is the **Wan 2.1** report | No dedicated Wan 2.2 paper exists; README/blog only |
| Spratling (2017) as RS-specific | It is a general algorithms review | Auksztulewicz & Friston (2016) *Cortex* 80:125–140 |
| Wallach (1959) as motion discounting | It is primarily about *induced motion* | Real, but a related-not-identical phenomenon |

## Plausibility ranking

Highest to lowest, as *candidate intellectual background* — not as proven lineage, since none can be
proven given zero citations in the source post:

1. **MEA neuroscience** — confirmed by TBC's own hardware description; the only topic verifiable at
   the fact level directly against the post.
2. **Criticality / avalanche dynamics** — strongest conceptual *and* methodological match to TBC's
   own "stability over extended temporal dynamics" language.
3. **Short-term synaptic depression** — most literal mechanistic match to "quiet when repeated", and
   structurally present by default in any living-neuron substrate.
4. **Repetition suppression and predictive coding** — strong conceptual fit and mainstream
   neuroscience, but no architectural evidence of explicit implementation. The "residual stream"
   terminology overlap is suggestive and almost certainly coincidental.
5. **Efficient/sparse coding and energy constraints** — plausible as company-level philosophy, but not
   reflected in this post's own stated performance claims, which are about stability, not efficiency.
6. **Temporal redundancy / motion discounting** — weakest fit; requires a self-generated motor signal
   with no clear analog in an externally-stimulated culture, and one of its two requested citations
   could not be verified at all.
