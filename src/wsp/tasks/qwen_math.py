"""Reproduction target: unblocking Qwen3.5-0.8B's math capability.

The paper reports that Astra diagnosed reasoning loops caused by a direction
written by MLP layers 15 and 19, subtracted that direction from 64 output
columns in each layer (131,072 of ~800M weights, 0.016%), and raised MATH-500
from 7.9% to 48.1% with no gradients.

This module holds the parts that do not need a GPU: locating the candidate
layers, describing them to an editor, and applying a returned delta. Scoring
requires the model, so it is delegated to a runner that can execute on a pod.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wsp.weights.slices import TensorSlice, apply_edit, count_changed, to_text

#: Layers the paper identified as writing the looping direction.
CANDIDATE_LAYERS: tuple[int, ...] = (15, 19)

#: Output columns the paper modified per layer.
COLUMNS_PER_LAYER = 64

#: Total weights in Qwen3.5-0.8B, for reporting the edit footprint.
TOTAL_PARAMS = 800_000_000


@dataclass(frozen=True)
class LayerTarget:
    """A resolved MLP down-projection layer to edit."""

    layer: int
    tensor_name: str
    shape: tuple[int, int]

    @property
    def slice(self) -> TensorSlice:
        return TensorSlice(name=self.tensor_name, cols=(0, COLUMNS_PER_LAYER))


def find_mlp_down_proj(keys: list[str], layer: int) -> str:
    """Locate the MLP down-projection tensor for a decoder layer.

    Naming differs across checkpoints, so match on the layer index plus the
    common suffixes rather than assuming one convention. Qwen3.5 nests the
    decoder under ``model.language_model`` because the checkpoint is a
    conditional-generation model with a vision tower.
    """
    markers = (f"layers.{layer}.", f"h.{layer}.")
    suffixes = ("mlp.down_proj.weight", "mlp.c_proj.weight", "mlp.fc2.weight")

    for key in keys:
        if not any(m in key for m in markers):
            continue
        if any(key.endswith(s) for s in suffixes):
            return key

    raise KeyError(
        f"no MLP down-projection found for layer {layer}; "
        f"looked for suffixes {suffixes} near layers.{layer}"
    )


#: Qwen3.5-0.8B interleaves linear and full attention every 4th layer.
#: The paper's target layers are both full-attention layers, which is a
#: plausible reason the looping direction is written there rather than in a
#: linear-attention layer.
FULL_ATTENTION_LAYERS: tuple[int, ...] = (3, 7, 11, 15, 19, 23)


def is_full_attention(layer: int) -> bool:
    """Whether a layer uses full attention in Qwen3.5-0.8B."""
    return layer in FULL_ATTENTION_LAYERS


def down_proj_shape(hidden_size: int = 1024, intermediate_size: int = 3584) -> tuple[int, int]:
    """Shape of an MLP down-projection weight in Qwen3.5-0.8B.

    ``down_proj`` maps intermediate -> hidden, so the tensor is
    ``[hidden_size, intermediate_size]``. The paper's 64 columns are therefore
    64 *input* channels of the down-projection, not output units.
    """
    return (hidden_size, intermediate_size)


def describe_for_editor(
    tensor: np.ndarray,
    target: TensorSlice,
    *,
    layer: int,
    context: str = "",
) -> str:
    """Render a slice as a prompt block for the editor.

    The header states the orientation explicitly, because "columns" is
    ambiguous: in a ``[hidden, intermediate]`` down-projection, a column is an
    input channel from the intermediate dimension, and the rows are the hidden
    units the layer writes to.
    """
    region = target.apply(tensor)
    header = (
        f"Layer {layer} MLP down-projection, shape {tensor.shape[0]}x{tensor.shape[1]} "
        f"(rows = hidden units written, columns = intermediate input channels). "
        f"Showing rows 0:{region.shape[0]}, columns 0:{region.shape[1]}."
    )
    body = to_text(region)
    return f"{header}\n{context}\n\n{body}".strip()


def apply_direction_subtraction(
    tensor: np.ndarray,
    target: TensorSlice,
    direction: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Subtract a shared direction from every row of the target columns.

    This mirrors the paper's edit: one direction, applied across the selected
    output columns, rather than an arbitrary per-element delta.
    """
    region = target.apply(tensor)
    if direction.shape != (region.shape[1],):
        raise ValueError(
            f"direction must have shape ({region.shape[1]},), got {direction.shape}"
        )
    delta = -np.broadcast_to(direction, region.shape).astype(np.float32)
    edited = apply_edit(tensor, target, delta)
    return edited, count_changed(tensor, edited)


def footprint(changed: int, total: int = TOTAL_PARAMS) -> str:
    """Human-readable edit footprint, matching the paper's reporting style."""
    return f"{changed:,} of {total:,} weights ({100 * changed / total:.4f}%)"
