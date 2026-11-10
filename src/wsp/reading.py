"""Reading what a network does from its raw weights: a replication of WSP Experiment 2.

The claim under test
--------------------
In *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab), Experiment 2 gives
a language model the 23,824 weights of a small diffusion model as text, with no tools and
no forward pass allowed, and asks what images the model generates. The paper reports the
right shape for three networks (circle, square, triangle).

This module holds everything needed to repeat that with other readers and other networks:

* the shapes and the DDPM recipe from the paper's release (12 x 12 images, an MLP
  ``160 -> 64 -> 64 -> 144``, x0-prediction, linear betas over 1,000 steps);
* the weight-to-text format, which reproduces the paper's files byte for byte;
* the paper's prompt (description and question), copied verbatim;
* a control that keeps every weight value and destroys the structure;
* the scorer, a fixed keyword rule applied to the answer's ``Images:`` line.

The description and question texts are from the paper's log release
(github.com/RikiyaT/lm_understand-logs, ``exp2_diffusion``), licensed CC BY 4.0.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

SIZE = 12
PIXELS = SIZE * SIZE
STEP_DIMS = 16
HIDDEN = 64
T_STEPS = 1000
BETA_START, BETA_END = 1e-4, 0.02
TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3")
#: The paper's three shapes, and one it did not use.
PAPER_SHAPES = ("circle", "square", "triangle")
SHAPES = PAPER_SHAPES + ("cross",)

DESCRIPTION = """THE MODEL
A denoising diffusion model (DDPM) for 12 x 12 grayscale images. Pixel values lie in [-1, 1]: -1 is black, +1 is white. Pixel number i = row * 12 + column (row 0 is the top row, column 0 the left column).
Training: every training image x_0 comes from one fixed collection of images. For a random step t (0..999) and Gaussian noise eps (one standard normal number per pixel), the noisy image is x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps, where beta_t rises linearly from 0.0001 (t = 0) to 0.02 (t = 999) and abar_t = (1 - beta_0)(1 - beta_1)...(1 - beta_t). The network learned to predict the clean image x_0 from x_t and t (loss: the average of (prediction - x_0)^2 over pixels).
Generation: start from pure Gaussian noise (t = 999); at every step the network's prediction of x_0 (clipped to [-1, 1]) is used to take one standard DDPM step to t - 1, down to t = 0. The result is a new image of the kind the model was trained on.

THE NETWORK
Input: 160 numbers: inputs 0..143 are the pixels of x_t (input i = pixel i); inputs 144..159 encode the step t: input 144 + k = sin(t * f_k) and input 152 + k = cos(t * f_k), with f_k = 1 / 10000^(k / 8), k = 0..7.
  h1 = relu(x @ W1 + b1)         W1: 160 x H, b1: H
  h2 = relu(h1 @ W2 + b2)        W2: H x H,   b2: H
  x0_hat = h2 @ W3 + b3          W3: H x 144, b3: 144   (output j = the predicted clean value of pixel j)
Row i of a weight matrix belongs to input i (or unit i of the previous layer); column j to unit j of the next layer (for W3: pixel j).
"""

QUESTION = """
QUESTION. The weights below belong to one such diffusion model. What images does it generate? Work from the weights. You have no tools: no code execution, no calculator, nothing to run. Do not do any forward-pass calculation: do not run the network on any input (in your head or on paper), and do not simulate the sampling. Read what the weights themselves show. This is a single response.

Answer in this order:
Observations: <what you see in the weights (for example which pixels or units have large or structured weights, patterns over the 12 x 12 grid); quote the weights you rely on>
Reasoning: <what kinds of images fit these observations, which are ruled out, and why>
Images: <what the generated images show, as specifically as you can: what is drawn, how big, where>
Example: <one typical generated image as 12 lines of 12 characters, '#' for white and '.' for black>
Confidence: <0-100>%

THE WEIGHTS
"""

#: The paper's one-line system instruction (``harness/instructions.md``).
SYSTEM = "Answer the user's request."


# --------------------------------------------------------------------------- shapes


def _inside(name: str, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of the points inside one randomly sized and placed shape.

    Coordinates are in pixels: pixel ``(row, col)`` covers ``[col, col + 1] x [row, row + 1]``.
    Every shape keeps half a pixel of margin from the image border.
    """
    if name == "circle":
        r = rng.uniform(2.0, 4.5)
        cx, cy = rng.uniform(r + 0.5, SIZE - 0.5 - r, size=2)
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    if name == "square":
        h = rng.uniform(1.5, 4.0)
        cx, cy = rng.uniform(h + 0.5, SIZE - 0.5 - h, size=2)
        return (np.abs(x - cx) <= h) & (np.abs(y - cy) <= h)
    if name == "triangle":
        w, hgt = rng.uniform(4.0, 9.0, size=2)
        left = rng.uniform(0.5, SIZE - 0.5 - w)
        top = rng.uniform(0.5, SIZE - 0.5 - hgt)
        frac = (y - top) / hgt
        return (frac >= 0) & (frac <= 1) & (np.abs(x - (left + w / 2)) <= (w / 2) * frac)
    if name == "cross":
        arm = rng.uniform(2.0, 4.5)
        thick = rng.uniform(0.75, 1.25)
        cx, cy = rng.uniform(arm + 0.5, SIZE - 0.5 - arm, size=2)
        dx, dy = np.abs(x - cx), np.abs(y - cy)
        return ((dx <= arm) & (dy <= thick)) | ((dy <= arm) & (dx <= thick))
    raise ValueError(f"unknown shape {name!r}; known: {SHAPES}")


def render_shape(name: str, rng: np.random.Generator, supersample: int = 4) -> np.ndarray:
    """One filled, anti-aliased shape on black: ``(12, 12)`` in ``[-1, 1]``."""
    ticks = (np.arange(SIZE * supersample) + 0.5) / supersample
    x, y = np.meshgrid(ticks, ticks)
    mask = _inside(name, x, y, rng)
    coverage = mask.reshape(SIZE, supersample, SIZE, supersample).mean(axis=(1, 3))
    return (2.0 * coverage - 1.0).astype(np.float32)


def make_dataset(name: str, n: int, seed: int) -> np.ndarray:
    """A fixed collection of ``n`` images of one shape, flattened to ``(n, 144)``."""
    rng = np.random.default_rng(seed)
    return np.stack([render_shape(name, rng).reshape(PIXELS) for _ in range(n)])


# --------------------------------------------------------------------------- the DDPM


def abar() -> np.ndarray:
    """Cumulative product of ``1 - beta_t`` for the linear schedule."""
    return np.cumprod(1.0 - np.linspace(BETA_START, BETA_END, T_STEPS))


def step_encoding(t: np.ndarray) -> np.ndarray:
    """The 16 numbers that encode step ``t``: eight sines, then eight cosines."""
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    freqs = 1.0 / 10000.0 ** (np.arange(8) / 8.0)
    return np.concatenate([np.sin(t * freqs), np.cos(t * freqs)], axis=1).astype(np.float32)


def forward(net: dict, x: np.ndarray) -> np.ndarray:
    """The network formula from the description, on ``(n, 160)`` inputs."""
    h1 = np.maximum(x @ net["W1"] + net["b1"], 0.0)
    h2 = np.maximum(h1 @ net["W2"] + net["b2"], 0.0)
    return h2 @ net["W3"] + net["b3"]


def sample(net: dict, n: int, seed: int) -> np.ndarray:
    """Generate ``n`` images as the description states: x0-prediction, clipped, DDPM steps."""
    rng = np.random.default_rng(seed)
    betas = np.linspace(BETA_START, BETA_END, T_STEPS)
    cum = abar()
    x = rng.standard_normal((n, PIXELS)).astype(np.float32)
    for t in range(T_STEPS - 1, -1, -1):
        enc = np.repeat(step_encoding(np.array([t])), n, axis=0)
        x0 = np.clip(forward(net, np.concatenate([x, enc], axis=1)), -1.0, 1.0)
        prev = cum[t - 1] if t > 0 else 1.0
        mean = (np.sqrt(prev) * betas[t] / (1 - cum[t])) * x0 + (
            np.sqrt(1 - betas[t]) * (1 - prev) / (1 - cum[t])
        ) * x
        var = betas[t] * (1 - prev) / (1 - cum[t])
        x = (mean + np.sqrt(var) * rng.standard_normal(x.shape)).astype(np.float32)
    return x


def train_ddpm(
    images: np.ndarray, *, steps: int = 40_000, batch: int = 512, lr: float = 1e-3, seed: int = 0
) -> tuple[dict, list[dict]]:
    """Train the MLP to predict the clean image. Returns the weights and a loss curve."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(PIXELS + STEP_DIMS, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, PIXELS),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.from_numpy(images)
    cum = torch.from_numpy(abar()).float()
    freqs = 1.0 / 10000.0 ** (torch.arange(8) / 8.0)
    generator = torch.Generator().manual_seed(seed)
    curve, running = [], 0.0
    for step in range(1, steps + 1):
        x0 = data[torch.randint(len(data), (batch,), generator=generator)]
        t = torch.randint(T_STEPS, (batch,), generator=generator)
        eps = torch.randn(x0.shape, generator=generator)
        a = cum[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        angles = t.unsqueeze(1).float() * freqs
        loss = ((model(torch.cat([xt, angles.sin(), angles.cos()], dim=1)) - x0) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        running += loss.item()
        if step % 1000 == 0:
            curve.append({"step": step, "loss": running / 1000})
            running = 0.0
    linears = [m for m in model if isinstance(m, nn.Linear)]
    net = {}
    for i, layer in enumerate(linears, start=1):
        # nn.Linear stores (out, in); the description's formula is x @ W, so W is (in, out).
        net[f"W{i}"] = layer.weight.detach().numpy().T.copy()
        net[f"b{i}"] = layer.bias.detach().numpy().copy()
    return net, curve


# --------------------------------------------------------------------------- text


def _row(values: np.ndarray) -> str:
    # Adding 0.0 turns a rounded -0.0 into 0.0, as the paper's files print it.
    return " ".join(f"{v:.3f}" for v in (np.round(values.astype(np.float64), 3) + 0.0))


def format_weights(net: dict) -> str:
    """All weights as text, 3 decimals: the exact layout of the paper's ``_weights.txt``."""
    h = net["b1"].shape[0]
    titles = {
        "W1": f"W1 (layer 1 weights), shape 160 x {h}: one line per row (row i = input i: "
              "rows 0..143 are pixels, rows 144..159 the step encoding)",
        "b1": f"b1 (layer 1 biases), shape {h}:",
        "W2": f"W2 (layer 2 weights), shape {h} x {h}: one line per row",
        "b2": f"b2 (layer 2 biases), shape {h}:",
        "W3": f"W3 (output layer weights), shape {h} x 144: one line per row (row i = unit i "
              "of h2; column j = pixel j)",
        "b3": "b3 (output layer biases), shape 144:",
    }
    blocks = [f"The network's weights (H = {h}). Values rounded to 3 decimals."]
    for name in TENSORS:
        tensor = net[name]
        rows = [_row(r) for r in tensor] if tensor.ndim == 2 else [_row(tensor)]
        blocks.append("\n".join([titles[name], *rows]))
    return "\n\n".join(blocks) + "\n"


def build_prompt(net: dict) -> str:
    """The full message a reader gets: description, question, then every weight."""
    return DESCRIPTION + QUESTION + format_weights(net)


def network_id(net: dict) -> str:
    """Eight hex digits from the weights, so a file name says nothing about the shape."""
    digest = hashlib.sha256()
    for name in TENSORS:
        digest.update(np.ascontiguousarray(net[name], dtype=np.float32).tobytes())
    return digest.hexdigest()[:8]


def shuffle_within_tensors(net: dict, seed: int) -> dict:
    """Permute the entries inside each tensor.

    Every value, each tensor's distribution, the shapes and the text format are kept; which
    weight connects which units is destroyed. A reader that reports a shape with confidence
    for this network is not getting it from the structure of the weights.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name in TENSORS:
        flat = net[name].reshape(-1).copy()
        rng.shuffle(flat)
        out[name] = flat.reshape(net[name].shape)
    return out


# --------------------------------------------------------------------------- scoring

#: Keyword classes for the ``Images:`` line. The first three extend the paper's rule
#: (circle / disc / round / oval; square / rectangle / box; triangle / pyramid / wedge)
#: with plurals and adjectives; ``cross`` is this project's addition.
KEYWORDS = {
    "circle": r"circles?|circular|discs?|disks?|round|rounded|ovals?|ellipses?|elliptical",
    "square": r"squares?|rectangles?|rectangular|box|boxes",
    "triangle": r"triangles?|triangular|pyramids?|wedges?",
    "cross": r"cross|crosses|cross-shaped|plus|plus-shaped|plus-sign",
}
#: A section heading: ``Images:``, ``**Images:**``, or a Markdown heading line such as
#: ``## Images`` with nothing else on it. The second form was added after the first run:
#: the scorer as first committed required the colon and so read ``## Images`` replies as
#: having no section (docs/stage8-reading-weights.md, "Scorer correction").
_SECTION = re.compile(
    r"^[^\w\n]*(Observations|Reasoning|Images|Example|Confidence)(?:[^\w\n]*:|[^\w\n]*$)",
    re.IGNORECASE | re.MULTILINE,
)


def answer_section(answer: str, name: str) -> str | None:
    """The text of one labelled section of an answer, or ``None`` if it is absent."""
    marks = list(_SECTION.finditer(answer))
    for i, mark in enumerate(marks):
        if mark.group(1).lower() == name.lower():
            end = marks[i + 1].start() if i + 1 < len(marks) else len(answer)
            return answer[mark.end():end].strip()
    return None


def classify(images_line: str | None) -> str:
    """Map an ``Images:`` line to a shape.

    Returns a shape name when exactly one keyword class appears, ``"ambiguous"`` when more
    than one does, ``"none"`` when no class does, and ``"missing"`` when the answer has no
    ``Images:`` section. Only an exact single-class match can be scored as correct.
    """
    if images_line is None:
        return "missing"
    found = [
        shape for shape, pattern in KEYWORDS.items()
        if re.search(rf"(?<![A-Za-z-])(?:{pattern})(?![A-Za-z])", images_line, re.IGNORECASE)
    ]
    if len(found) == 1:
        return found[0]
    return "ambiguous" if found else "none"


def confidence(answer: str) -> float | None:
    """The stated confidence in percent, or ``None``."""
    section = answer_section(answer, "Confidence")
    match = re.search(r"(\d+(?:\.\d+)?)\s*%?", section or "")
    return float(match.group(1)) if match else None


_NUMBER = r"[-\u2212]?\d\.\d{3}"
_RUN = re.compile(rf"{_NUMBER}(?:\s*,?\s+{_NUMBER}){{3,}}")


def quoted_runs(answer: str) -> list[list[float]]:
    """Runs of four or more 3-decimal numbers quoted in a reply."""
    runs = []
    for match in _RUN.finditer(answer.replace(",", " ")):
        runs.append([float(x.replace("\u2212", "-")) for x in re.findall(_NUMBER, match.group(0))])
    return runs


def locate_run(net: dict, values: list[float]) -> str | None:
    """Where a quoted run of weights occurs in the network, or ``None`` if nowhere.

    Looks along every row and column of every tensor at strides 1..13 (stride 12 in a row
    of W3 is one image column), and at the four image corners of a 144-long row. A reply
    that quotes weights the network does not contain is inventing its evidence. A run that
    is not found may still be a real read at positions this search does not cover.
    """
    target = np.asarray(values, dtype=np.float64)
    n = len(target)
    for name in TENSORS:
        tensor = np.round(net[name].astype(np.float64), 3)
        # Each entry of ``views`` is a 2-D array whose rows are the lines to search.
        views = [tensor[None, :]] if tensor.ndim == 1 else [tensor, tensor.T]
        for lines in views:
            length = lines.shape[1]
            if n == 4 and length == PIXELS:
                corners = lines[:, [0, SIZE - 1, PIXELS - SIZE, PIXELS - 1]]
                if (np.abs(corners - target) < 6e-4).all(axis=1).any():
                    return f"{name} corners"
            for stride in range(1, 14):
                span = (n - 1) * stride
                if span >= length:
                    break
                index = np.arange(length - span)[:, None] + stride * np.arange(n)
                if (np.abs(lines[:, index] - target) < 6e-4).all(axis=2).any():
                    return f"{name} stride {stride}"
    return None


def score_answer(answer: str, truth: str | None) -> dict:
    """Score one answer. ``truth`` is ``None`` for a control with no right answer."""
    line = answer_section(answer, "Images")
    guess = classify(line)
    return {
        "guess": guess,
        "correct": None if truth is None else guess == truth,
        "confidence": confidence(answer),
        "images_line": (line or "")[:400],
    }
