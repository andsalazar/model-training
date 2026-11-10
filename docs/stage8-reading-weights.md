# Stage 8: can a language model read what a network does from its raw weights?

*The earlier stages never tested the paper's "reading" claim. Stage 0 confirmed the effect
of the paper's edit, and in Stage 1 an editor operated measurement tools without ever
seeing a weight value. This stage tests reading directly, by repeating the paper's
Experiment 2 with its own networks, with fresh ones, and with other readers.*

**Status: design fixed, no reader called yet.** Everything in "Design" and "Scoring" below
was written and committed before any reply existed. Results are added underneath.

Code: `src/wsp/reading.py`, `scripts/read_weights.py`. Evidence:
`artifacts/stage8_reading/`. No GPU; the networks train on a CPU in about 2.5 minutes
each.

## What the paper did

Experiment 2 of *Sparks of Weight-Space Programming*: three small diffusion models (an
MLP `160 → 64 → 64 → 144`, 23,824 weights) were trained to generate 12 × 12 images of
filled circles, squares and triangles. GPT-6 Astra was given all the weights as text
(about 100,000 tokens), with no tools and an instruction not to compute any forward pass,
and asked an open question: what images does this model generate? The paper reports the
right shape for all three, one reply each.

The authors released the networks, the exact prompts and the replies
([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs),
`exp2_diffusion`, CC BY 4.0). `wsp.reading` rebuilds each of the three prompts from the
released weights **byte for byte**, and its scorer gives the paper's own three replies
the same verdicts the paper's scorer did.

## Design

Nine networks, each sent to each reader once, in a fresh conversation, with the paper's
prompt and its one-line system instruction.

| group | networks | what it tests |
|---|---|---|
| **paper** | the paper's circle, square and triangle | does the reported result repeat, with the paper's reader and with others? |
| **fresh, same shapes** | circle, square, triangle, trained here with the same recipe and a new seed | does it hold on networks that have never been published? |
| **fresh, new shape** | two networks trained on a plus-shaped **cross** (seeds 1 and 2) | is the reader reading, or picking from the three shapes anyone would guess for a toy diffusion model? |
| **control** | the fresh circle network with the entries of each tensor shuffled | does the reader report a shape when the structure is gone? |

**The fresh networks.** Same architecture, noise schedule, objective, optimiser, step
count (40,000) and batch size (512) as the paper states. Not stated by the paper, and
chosen here: learning rate 1e-3, a fixed collection of 20,000 images per shape, and the
size ranges of the shapes (`wsp.reading._inside`). Each fresh network was checked by
generating 1,000 images and classifying them:

| id | group | shape | generated images classified as that shape | final training loss |
|---|---|---|---|---|
| `d88aba69` | paper | circle | 96.3% | (paper: about 0.20) |
| `a6a9dfb7` | paper | square | 96.6% | |
| `246a5b58` | paper | triangle | 96.0% | |
| `1987dad6` | fresh | circle | 95.9% | 0.201 |
| `cdd746d8` | fresh | square | 95.2% | 0.226 |
| `e7e2cc43` | fresh | triangle | 99.4% | 0.156 |
| `edc5465a` | fresh | cross | 99.2% | 0.159 |
| `ee40a412` | fresh | cross | 100.0% | 0.160 |
| `a6b02be9` | control | none | not applicable | |

The classifier is a small network trained here on the four shapes (99.2% on held-out real
images). The paper's own check, with its own classifier, gave its networks 98.4–100%, so
this one is somewhat stricter on the paper's images; the fresh networks score in the same
range as the paper's. The misclassified samples seen are small rounded squares and
circles, which are hard to tell apart at 12 × 12.

**The control.** Shuffling inside each tensor keeps every weight value, each tensor's
distribution, the shapes and the text format, and destroys which weight connects which
units. The shuffled network generates one fixed speckled grey image whatever noise it
starts from (variation across generated images 0.003, against 0.35–0.52 for the real
networks). It has no right answer. A reader that reads structure should name no shape, or
give a clearly lower confidence than it gives real networks.

**Readers.** `astra` (GPT-6 Astra, the paper's reader), and four cheaper ones from this
project's editor list: `kimi`, `deepseek`, `glm`, `qwen`. All through OpenRouter's chat
completions API with reasoning effort `high`, no temperature set, a 16,000-token output
limit, and no tools offered. `muse` is left out: its 131,072-token context is too close to
the size of the prompt.

**Protocol change before any scored reply (2026-11-09).** A single plumbing check
(DeepSeek, one network) spent the whole 16,000-token limit on reasoning, much of it
copying weight rows, and returned an empty answer. An empty answer says nothing about the
shape, so the rule was extended before the run: a reply that hits the limit before it
reaches its `Images:` section is asked for once more with a 64,000-token limit. The
cut-off reply is kept (`answers/<reader>/<id>.truncated.json`) and its cost is counted. A
reply that is still cut off at 64,000 is scored `missing`, which is incorrect.

**Differences from the paper's run.** The paper called Astra through the Codex CLI with
every tool switched off; this goes through OpenRouter's API, where no tools are offered.
The reasoning effort (`high`), the system line and the prompt are the same. The paper's
networks and prompts have been public since 2026-10-29, so a reader could in principle
have seen them; the fresh networks cannot have been seen.

## Scoring

Fixed before any reply was collected, in `wsp.reading.score_answer`:

1. Take the reply's `Images:` section only (the text up to the next section heading).
2. Look for the keyword classes below, as whole words.
3. Exactly one class present: that is the guess. More than one: `ambiguous`. None:
   `none`. No `Images:` section: `missing`.
4. A reply is **correct** only if the guess equals the network's shape. `ambiguous`,
   `none` and `missing` are incorrect.

| class | keywords |
|---|---|
| circle | circle(s), circular, disc(s), disk(s), round, rounded, oval(s), ellipse(s), elliptical |
| square | square(s), rectangle(s), rectangular, box(es) |
| triangle | triangle(s), triangular, pyramid(s), wedge(s) |
| cross | cross(es), cross-shaped, plus, plus-shaped, plus-sign |

The first three extend the paper's rule (circle / disc / round / oval; square / rectangle /
box; triangle / pyramid / wedge) with plurals and adjectives. The cross class is new.
The answer key is in `manifest.json`; the `ask` step reads only network ids from it, and
file names are hashes of the weights.

Every reply is saved in full, so a reader of this document can check a verdict against
the text. Where the keyword rule and a plain reading of the reply disagree, the keyword
verdict stands in the tables and the disagreement is listed.

## How the results will be read

Decided in advance:

- **The paper's result repeats** if Astra is correct on the paper's three networks.
- **It holds on unseen networks** if Astra is also correct on the three fresh networks of
  the same shapes. A reader choosing among the three paper shapes at random would get all
  three right 1 time in 27, and all six 1 time in 729.
- **It is reading rather than guessing from a short list** if the cross networks are
  named as crosses. A reader that only ever answers circle, square or triangle scores 0
  of 2 here whatever else it gets.
- **The control** counts against a reader that names a shape for the shuffled network
  with a confidence in the range it gives real networks.
- **"Another model can do it"** is judged per reader on the same four lines.

With nine networks and one reply each, these are counts, not rates. One reply per network
is what the paper reported; it is also the main limit on what this can show.

## Results

Not yet run.
