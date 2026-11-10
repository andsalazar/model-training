# Stage 8: can a language model read what a network does from its raw weights?

*The earlier stages never tested the paper's "reading" claim. Stage 0 confirmed the effect
of the paper's edit, and in Stage 1 an editor operated measurement tools without ever
seeing a weight value. This stage tests reading directly, by repeating the paper's
Experiment 2 with its own networks, with fresh ones, and with other readers.*

**Result.** With the paper's reader the claim repeats, and it holds beyond the paper's
own networks: GPT-6 Astra named the right shape for the paper's three networks, for three
freshly trained ones, and for a shape the paper never used, and it declined to name
anything for a network with shuffled weights. None of the four cheaper readers did this.
One reply per network, nine networks, so these are counts, not rates.

Everything in "Design", "Scoring" and "How the results will be read" was written,
committed and pushed before any reply existed (commits `ddc7f68` and `a71b9e4`). Results
and one scorer correction are underneath.

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

45 replies (5 readers × 9 networks), collected 2026-11-09. No reply contains a tool call.
Total spend **$21.09**, of which Astra $15.03 and Kimi $4.80. That is above the roughly
$13 estimated beforehand: Astra used 4,700–15,400 reasoning tokens per reply here against
1,000–2,600 in the paper's logs, and 21 replies needed the 64,000-token retry.

Correct under the keyword rule:

| reader | paper's networks | fresh, same shapes | fresh, cross | control (shuffled) | replies with no `Images` section |
|---|---|---|---|---|---|
| **astra** | **3/3** | **3/3** | 1/2 (2/2 on a plain reading, see below) | names nothing, confidence 0% | 0 |
| kimi | 3/3 | 1/3 | 0/2 | "digit-4-like figures", 76% | 0 |
| glm | 1/3 | 1/3 | 0/2 | "no large coherent shape", 25% | 3 |
| deepseek | 0/3 | 0/3 | 0/2 | "a vertical stroke", 74% | 5 |
| qwen | 0/3 | 0/3 | 0/2 | "sparse scattering of bright pixels", 62% | 7 |

Reproduce with `python scripts/read_weights.py score`; per-reply verdicts are in
`artifacts/stage8_reading/scores.json`, full replies in `answers/<reader>/<id>.json`.

### Astra

| network | truth | Astra's `Images:` line (shortened) | confidence | verdict |
|---|---|---|---|---|
| `d88aba69` paper | circle | one white circular or slightly oval outline on black | 85% | correct |
| `a6a9dfb7` paper | square | one white, thin, axis-aligned rectangular frame on black | 76% | correct |
| `246a5b58` paper | triangle | one white outlined triangle, pointing upward | 80% | correct |
| `1987dad6` fresh | circle | one filled white circle on a black background | 84% | correct |
| `cdd746d8` fresh | square | a single white hollow rectangle on a black background | 68% | correct |
| `e7e2cc43` fresh | triangle | one filled white, upward-pointing triangle | 87% | correct |
| `edc5465a` fresh | cross | one white "+" sign: two equal-length perpendicular bars | not given | correct |
| `ee40a412` fresh | cross | a single white "+" on black, equal horizontal and vertical arms | 86% | **incorrect by the rule** |
| `a6b02be9` control | none | "I cannot reliably identify what these images depict" | 0% | names nothing |

Read against the lines fixed in advance:

- **The paper's result repeats:** 3 of 3 on the paper's networks, through a different
  access route (OpenRouter's API rather than the Codex CLI).
- **It holds on unseen networks:** 3 of 3 on fresh networks of the same shapes. Six of
  six over both groups; a reader picking among the three shapes at random does that 1
  time in 729.
- **It is not a guess from a short list:** both cross networks were described as a plus
  sign with two perpendicular bars. The paper did not use this shape.
- **The control:** for the shuffled network Astra wrote that naming a shape "would be
  speculation" and gave 0%, against 68–87% on the real networks.

The first cross reply (`edc5465a`) reached the 16,000-token limit after its example
drawing, so it has no confidence line; its `Images` section was complete, so under the
rule it was not asked again.

As in the paper, Astra described the paper's three networks as outlines although they
generate filled shapes. On the fresh networks it called the circle and the triangle
filled and the square hollow.

### The cheaper readers

- **Kimi** was right on the paper's three networks and on one of three fresh ones, called
  both crosses and the fresh square a circle at 90–93% confidence, and described the
  shuffled network as "digit-4-like figures" at 76%. It answered "circle" for five of the
  eight real networks. Its three correct answers on the paper's networks are not evidence
  of reading on their own, given that it fails on fresh networks and on the control.
- **GLM** gave "no large coherent shape" at 25% for the control, which is the behaviour a
  reader should show, and was right on 2 of 8 real networks (3 on a plain reading).
- **DeepSeek** returned no answer for 5 networks even with 64,000 tokens, spending them on
  reasoning. Its three answers on real networks were handwritten digits ("4", "1", "2")
  at 67–86% confidence, and for the control it described a vertical stroke at 74%.
- **Qwen** returned an empty answer for 7 of 9 networks: the output ended inside its
  reasoning, typically while listing weights, with the provider reporting a normal stop.
  Whether that is the model or the provider cannot be told from the replies.

So on the project's original question, whether a different and cheaper model can do the
reading: **not on this test.** Only the paper's own reader did.

### Where the keyword rule and a plain reading disagree

The keyword verdict stands in the tables. Two replies read differently in plain terms:

| reply | keyword verdict | what the reply says |
|---|---|---|
| astra, `ee40a412` (cross) | `none`, incorrect | 'A single white "+" on black, with approximately equal-length horizontal and vertical arms'. The rule has "plus" and "cross" but not the character "+". On a plain reading this is correct, which would make Astra 2/2 on crosses and 8/8 overall. |
| glm, `d88aba69` (circle) | `missing`, incorrect | The reply has no `Images` section; its reasoning ends "The generated images are circles/rings". |

For the control, the keyword rule returns `none` for every reader because none of them
used a shape keyword. Kimi and DeepSeek still named a definite figure (a digit, a stroke)
with confidence in the range they give real networks, which is what the control was
meant to catch.

### Scorer correction after the run

The scorer as first committed looked for a section heading followed by a colon
(`Images:`). Three replies used Markdown heading lines instead (`## Images`), and were
read as having no `Images` section. The written rule is "take the reply's `Images:`
section", so the parser was wrong, and it was fixed to accept a heading line with nothing
else on it (`wsp.reading._SECTION`). The keyword lists were not touched. The three
verdicts that changed:

| reply | as first committed (`a71b9e4`) | corrected |
|---|---|---|
| astra, `e7e2cc43` (triangle) | `missing`, incorrect | `triangle`, correct, 87% |
| qwen, `ee40a412` (cross) | `missing`, incorrect | `circle`, incorrect, 75% |
| qwen, `a6b02be9` (control) | `missing` | `none`, 62% |

Under the scorer exactly as first committed, Astra is 2/3 on the fresh same-shape
networks rather than 3/3. The reply in question says "one filled white, upward-pointing
triangle on a black background".

### A check added after the run: are the quoted weights real?

Earlier stages of this project found editors reporting numbers they had not measured, so
the replies were checked for invented evidence. Every run of four or more weights quoted
in a reply was searched for in that network's weights, along rows and columns, at regular
strides, and at the four image corners (`wsp.reading.locate_run`). This was not part of
the design fixed in advance.

| reader | quoted runs found in the weights |
|---|---|
| astra | 18 of 18 |
| kimi | 17 of 17 |
| glm | 9 of 14 |
| deepseek | 2 of 2 |
| qwen | 1 of 1 |

A run that is not found may be a real read at positions the search does not cover, so
GLM's 5 are unexplained rather than shown to be invented. Astra's and Kimi's quotes are
all real: Kimi's wrong answers come from misreading real weights, not from inventing
them.

## What this does and does not show

- **Shown:** the paper's Experiment 2 result repeats with its reader, on networks that
  were never published, including one shape outside the paper's set, with a control that
  the reader passes.
- **Not shown: that another model can do it.** Four cheaper readers could not, at these
  settings. Two of them mostly failed to produce an answer within 64,000 tokens, which is
  a failure on this task but not a clean measure of what they could read with a
  different prompt or a tool.
- **Small numbers.** One reply per network, nine networks, one run. Astra's result is 7
  or 8 of 8 depending on the "+" reply, with no repeat to say how stable it is.
- **Compliance with "no forward pass" cannot be checked.** The API returns only a short
  summary of Astra's reasoning. The paper's argument that Astra is not simulating the network
  (it draws outlines for filled shapes) applies to its replies on the paper's networks
  here too, and less clearly to the fresh ones.
- **A narrow kind of reading.** In these networks the output layer's weights are laid out
  on the same 12 × 12 grid as the image, and Astra's observations mostly quote that
  layer. The paper says as much about its first two experiments. This says nothing about
  reading the weights of a language model, and it does not change the Stage 0–1 findings:
  there, a static slice of Qwen's weights was not enough for any editor.
- **Not attempted:** the paper's Experiments 1, 3 and 7.
