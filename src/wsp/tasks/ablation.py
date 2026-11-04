"""Blind-vs-weights ablation.

The paper reports that Astra found a reasoning-loop bug "that Qwen's own model
card warns about". That raises a confound: an editor could produce the fix from
recall of the model card rather than from reading the weights.

The ablation separates the two explanations by running the same request twice,
once with the weights and once without:

* ``weights``  — the editor sees the tensor slice. Success here is consistent
  with weight-reading.
* ``blind``    — the editor sees only the goal and the model name. Success here
  means recall is sufficient, and the paper's result is partly contaminated.

A Qwen-family editor is the sharpest test, because it is the most likely to have
memorised its own family's documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Condition(str, Enum):
    """Which information the editor is given."""

    WEIGHTS = "weights"
    BLIND = "blind"


@dataclass(frozen=True)
class AblationPrompt:
    """A matched pair of prompts differing only in whether weights are shown."""

    condition: Condition
    system: str
    user: str


SYSTEM_PROMPT = (
    "You are analysing the weights of a neural network. You may propose a "
    "modification to improve its behaviour. Answer with a concrete, minimal "
    "change, or state that you cannot determine one."
)

#: Shared instruction, so the two conditions differ only in the weight block.
#: The architecture context is given to *both* arms: the blind arm tests whether
#: the editor needs the weights, not whether it can work without any context.
TASK_INSTRUCTION = (
    "The model is Qwen3.5-0.8B, an 800M-parameter decoder-only language model "
    "with 24 layers and hidden size 1024. It performs poorly on grade-school "
    "math: in thinking mode it enters reasoning loops and runs to the token "
    "limit without reaching an answer.\n\n"
    "The tensor under consideration is the MLP down-projection of layer 15, "
    "shape [1024, 3584] (rows = hidden units written, columns = intermediate "
    "input channels). Layer 15 is one of the full-attention layers.\n\n"
    "Propose the minimal weight modification that would stop the looping. If "
    "you can identify a specific direction or set of columns to change, name "
    "them and give the values."
)

BLIND_NOTE = (
    "You have not been given the weights. Answer from what you know about this "
    "model family, and say explicitly whether your answer relies on recall of "
    "public documentation rather than on the weights themselves."
)


def build_prompts(weight_block: str | None) -> tuple[AblationPrompt, AblationPrompt]:
    """Build the matched weights/blind prompt pair.

    ``weight_block`` is the rendered tensor slice. Passing ``None`` still
    produces a valid pair, which is useful for testing the harness itself.
    """
    blind = AblationPrompt(
        condition=Condition.BLIND,
        system=SYSTEM_PROMPT,
        user=f"{TASK_INSTRUCTION}\n\n{BLIND_NOTE}",
    )

    if weight_block is None:
        shown = "(weights unavailable)"
    else:
        shown = weight_block

    with_weights = AblationPrompt(
        condition=Condition.WEIGHTS,
        system=SYSTEM_PROMPT,
        user=f"{TASK_INSTRUCTION}\n\nWeight slice:\n\n{shown}",
    )
    return with_weights, blind


#: Phrases that indicate the editor is drawing on documentation rather than weights.
RECALL_MARKERS: tuple[str, ...] = (
    "model card",
    "documentation",
    "documented",
    "known issue",
    "as described in",
    "according to the",
    "publicly",
    "reported",
    "recall",
    "from memory",
)


def detect_recall(text: str) -> list[str]:
    """Return recall markers present in an editor's response.

    This is a heuristic for triage, not a verdict: a response can mention the
    model card and still be reasoning from weights. It exists to flag runs that
    need a human read.
    """
    lowered = text.lower()
    return [marker for marker in RECALL_MARKERS if marker in lowered]


@dataclass
class AblationResult:
    """Outcome of one editor on the matched prompt pair."""

    editor: str
    weights_response: str
    blind_response: str
    weights_recall_markers: list[str]
    blind_recall_markers: list[str]

    @property
    def blind_succeeded(self) -> bool:
        """Whether the blind arm produced a concrete proposal.

        A blind arm that names specific columns or values is evidence of recall;
        one that declines is evidence the weights were doing the work.
        """
        return _proposes_specific_change(self.blind_response)

    @property
    def weights_succeeded(self) -> bool:
        return _proposes_specific_change(self.weights_response)

    def verdict(self) -> str:
        """One-line interpretation of the pair."""
        if self.blind_succeeded and self.weights_succeeded:
            return "recall sufficient — paper's result may be contaminated"
        if self.weights_succeeded and not self.blind_succeeded:
            return "weight-reading supported — blind arm declined"
        if not self.weights_succeeded and not self.blind_succeeded:
            return "editor failed both arms — no signal"
        return "blind succeeded but weights arm failed — inconsistent, inspect"


def _proposes_specific_change(text: str) -> bool:
    """Whether a response names a concrete edit rather than declining."""
    lowered = text.lower()
    declines = (
        "cannot determine",
        "can't determine",
        "unable to determine",
        "insufficient information",
        "no way to tell",
        "cannot identify",
    )
    if any(d in lowered for d in declines):
        return False
    # A concrete proposal names a layer, a column, or a numeric direction.
    signals = ("layer", "column", "direction", "subtract", "zero", "scale")
    return any(s in lowered for s in signals)
