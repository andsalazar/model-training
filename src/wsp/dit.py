"""Activation capture for a DiT video model, as a counterpart to :mod:`wsp.runner`.

Why this exists
---------------
:class:`wsp.runner.TransformersActivationCapturer` assumes a HuggingFace causal LM:
``model.layers[i]``, a residual stream over text-token positions, one forward pass per
prompt. A diffusion transformer has three differences that break every one of those
assumptions:

1. **Tokens are patches, not words.** There is no tokenizer; the sequence axis is
   ``n_frames × patches_per_frame``. A "token window" means frames or spatial patches,
   not text positions.
2. **There is a timestep axis.** The same layer is called at every denoising step, so
   a single hook fires ``n_steps`` times. Which call you keep is a choice, and it
   changes what the captured direction means.
3. **Conditioning enters differently.** Text is injected through cross-attention and
   timestep through an embedding added to the residual stream. The residual stream is
   still the right tensor to read — it is still the additive highway — but the
   surrounding structure is not a decoder block.

What this module provides
-------------------------
The *pure* logic, so it can be tested without a GPU or 23GB of weights:

``CaptureSchedule``
    Which (layer, timestep) pairs to record. Recording everything is ~40 layers ×
    50 steps × 3600 patches and would exhaust memory, so the schedule is explicit.

``ResidualCapture``
    Accumulates hooked tensors and reduces them to per-layer statistics:
    mean over patches, and the per-patch matrix when a full capture is wanted.

``patch_window`` / ``frame_window``
    The analogue of ``token_window``: restrict a capture to a frame range or a
    spatial region before averaging, so a contrast compares the region where two
    rollouts differ rather than the prefix they share.

What this module deliberately does NOT do
-----------------------------------------
It does not import torch at module level, and it does not know how to attach a hook.
Attaching is a five-line job that needs the real model; keeping it out means the
reduction maths is testable here and the GPU-side code stays small enough to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Hook captures are reduced as they arrive. Holding every layer × step × patch tensor
#: for an 11B model at 960×960 is gigabytes; the reductions below are all a WSP
#: direction needs.
_DEFAULT_LAYERS = (0, 1, 2, 3, 6, 9, 12)


@dataclass(frozen=True)
class CaptureSchedule:
    """Which layers and timesteps to record.

    ``layers`` are DiT block indices (0-based). ``timesteps`` are denoising step
    indices — 0 is the noisiest, ``n_steps-1`` the cleanest. TBC's adapter sits in
    "early DiT layers", so early layers and early steps are the default: those are
    where a residual module would be inserted and where structure is least settled.

    ``keep_patches`` additionally retains the full ``[n_patches, hidden]`` matrix
    per capture instead of only its mean. Needed to compute a *direction*, because a
    direction is a contrast of means over a patch subset; not needed for a summary.
    """

    layers: tuple[int, ...] = _DEFAULT_LAYERS
    timesteps: tuple[int, ...] | None = None
    keep_patches: bool = False

    def wants(self, layer: int, timestep: int) -> bool:
        """Whether this (layer, timestep) pair should be recorded."""
        if layer not in self.layers:
            return False
        if self.timesteps is None:
            return True
        return timestep in self.timesteps

    def describe(self) -> str:
        steps = "all" if self.timesteps is None else ",".join(map(str, self.timesteps))
        return (
            f"layers={','.join(map(str, self.layers))} timesteps={steps} "
            f"patches={'keep' if self.keep_patches else 'reduce'}"
        )


def _as_patch_matrix(activation: np.ndarray) -> np.ndarray:
    """Coerce a hooked DiT activation to ``[n_patches, hidden]``.

    DiTs pass tensors shaped ``[batch, n_tokens, hidden]``. Batch is normally 1 for a
    single generation, so it is squeezed; a genuine multi-batch capture is rejected
    rather than silently averaged, because averaging across generations would mix two
    different rollouts into one direction.
    """
    array = np.asarray(activation, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(
                f"expected batch size 1, got {array.shape[0]}; "
                "average per-batch rather than across generations"
            )
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"expected [tokens, hidden], got shape {array.shape}")
    return array


def frame_window(
    matrix: np.ndarray, frames: str | None, n_frames: int
) -> np.ndarray:
    """Restrict a patch matrix to a frame range, given patches laid out frame-major.

    ``frames`` is ``"start:stop"`` in frame indices, or ``None`` for all. The patch
    axis is split into ``n_frames`` equal contiguous blocks, which matches how a DiT
    flattens ``[T, H, W]`` latents into a token sequence.

    The analogue of ``token_window`` in the text case: a contrast that averages over
    every frame is dominated by the frames the two rollouts share, so restricting to
    the region where they diverge is what makes the contrast informative.
    """
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected [tokens, hidden], got shape {array.shape}")
    if not frames:
        return array
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    total = array.shape[0]
    if total % n_frames != 0:
        raise ValueError(
            f"{total} patches do not divide evenly into {n_frames} frames"
        )
    per_frame = total // n_frames
    try:
        start_text, _, stop_text = frames.partition(":")
        start = int(start_text) if start_text.strip() else 0
        stop = int(stop_text) if stop_text.strip() else n_frames
    except ValueError as exc:
        raise ValueError(f"frames must be 'start:stop': {exc}") from exc
    if start < 0 or stop > n_frames or start >= stop:
        raise ValueError(f"frame window {frames!r} is outside 0:{n_frames}")
    return array[start * per_frame : stop * per_frame]


@dataclass
class LayerCapture:
    """What was recorded for one (layer, timestep) pair."""

    layer: int
    timestep: int
    n_patches: int
    hidden: int
    #: ``[n_patches, hidden]`` when the schedule kept patches, else ``None``.
    patches: np.ndarray | None = None

    @property
    def mean(self) -> np.ndarray:
        """Mean activation over patches — the per-layer state."""
        if self.patches is None:
            raise ValueError(
                "patches were not retained; schedule.keep_patches must be True "
                "to reduce to a mean"
            )
        return self.patches.mean(axis=0)

    def summary(self) -> dict:
        return {
            "layer": self.layer,
            "timestep": self.timestep,
            "n_patches": self.n_patches,
            "hidden": self.hidden,
            "kept_patches": self.patches is not None,
        }


@dataclass
class ResidualCapture:
    """Accumulates hooked DiT tensors, keyed by (layer, timestep)."""

    schedule: CaptureSchedule
    captures: dict[tuple[int, int], LayerCapture] = field(default_factory=dict)
    #: Calls the schedule declined to record. Reported so a schedule that silently
    #: matched nothing is visible rather than looking like a capture that found no
    #: structure.
    skipped: int = 0

    def record(self, layer: int, timestep: int, activation: np.ndarray) -> bool:
        """Record one hooked tensor. Returns whether the schedule wanted it."""
        if not self.schedule.wants(layer, timestep):
            self.skipped += 1
            return False
        matrix = _as_patch_matrix(activation)
        self.captures[(layer, timestep)] = LayerCapture(
            layer=layer,
            timestep=timestep,
            n_patches=int(matrix.shape[0]),
            hidden=int(matrix.shape[1]),
            patches=matrix if self.schedule.keep_patches else None,
        )
        return True

    def layers(self) -> list[int]:
        return sorted({layer for layer, _ in self.captures})

    def timesteps(self) -> list[int]:
        return sorted({step for _, step in self.captures})

    def at(self, layer: int, timestep: int) -> LayerCapture:
        key = (layer, timestep)
        if key not in self.captures:
            raise KeyError(
                f"nothing captured at layer {layer} timestep {timestep}; "
                f"available: {sorted(self.captures)}"
            )
        return self.captures[key]

    def direction(
        self,
        layer: int,
        timestep: int,
        other: "ResidualCapture",
        frames: str | None = None,
        n_frames: int = 1,
    ) -> np.ndarray:
        """Unit direction from ``other``'s state toward this one at (layer, timestep).

        Both captures must have been taken with ``keep_patches`` and with the same
        ``n_frames``, or the frame window would index different spatial regions in
        each and the difference would be meaningless.

        This is the DiT analogue of :func:`wsp.activations.direction_from_contrast`:
        the same subtraction and normalisation, applied to a frame-restricted mean
        over patch tokens instead of a token-window mean over text positions.
        """
        mine = self.at(layer, timestep)
        theirs = other.at(layer, timestep)
        if mine.patches is None or theirs.patches is None:
            raise ValueError("both captures must set keep_patches=True")
        if mine.hidden != theirs.hidden:
            raise ValueError(
                f"hidden size mismatch: {mine.hidden} vs {theirs.hidden}"
            )
        a = frame_window(mine.patches, frames, n_frames).mean(axis=0)
        b = frame_window(theirs.patches, frames, n_frames).mean(axis=0)
        delta = a - b
        norm = float(np.linalg.norm(delta))
        if norm == 0.0:
            raise ValueError("states are identical; no direction to derive")
        return (delta / norm).astype(np.float32)

    def summary(self) -> dict:
        return {
            "schedule": self.schedule.describe(),
            "n_captures": len(self.captures),
            "n_skipped": self.skipped,
            "layers": self.layers(),
            "timesteps": self.timesteps(),
            "captures": [c.summary() for c in self.captures.values()],
        }
