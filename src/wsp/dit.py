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

#: Ovi runs two WanModel stacks side by side and fuses them block by block, so every
#: block index exists twice. A video state and an audio state at the same
#: (layer, timestep) are different tensors of different widths, and averaging them
#: would be the same mistake as averaging across a batch.
STREAMS = ("video", "audio")


@dataclass(frozen=True)
class CaptureSchedule:
    """Which layers, timesteps and streams to record.

    ``layers`` are DiT block indices (0-based). ``timesteps`` are denoising step
    indices — 0 is the noisiest, ``n_steps-1`` the cleanest. TBC's adapter sits in
    "early DiT layers", so early layers and early steps are the default: those are
    where a residual module would be inserted and where structure is least settled.

    ``keep_patches`` additionally retains the full ``[n_patches, hidden]`` matrix
    per capture instead of only its mean. Needed to compute a *direction*, because a
    direction is a contrast of means over a patch subset; not needed for a summary.

    ``streams`` selects which of Ovi's two fused stacks to record. Both by default,
    because the project's question is whether one method can be aimed at video and
    audio separately — which needs both captured from the same rollout.
    """

    layers: tuple[int, ...] = _DEFAULT_LAYERS
    timesteps: tuple[int, ...] | None = None
    keep_patches: bool = False
    streams: tuple[str, ...] = STREAMS
    #: Keep every ``patch_stride``-th patch within each frame. 1 records everything.
    #: A full 960x960 video residual is 27,900 tokens x 3072 wide = 171MB per
    #: (layer, step, stream), so a multi-layer pair of rollouts needs this.
    patch_stride: int = 1
    #: Frames in the rollout, needed so the stride can be taken *within* a frame and
    #: leave a token count that still divides evenly for :func:`frame_window`.
    n_frames: int | None = None

    def wants(self, layer: int, timestep: int, stream: str = "video") -> bool:
        """Whether this (layer, timestep, stream) triple should be recorded."""
        if stream not in self.streams:
            return False
        if layer not in self.layers:
            return False
        if self.timesteps is None:
            return True
        return timestep in self.timesteps

    def describe(self) -> str:
        steps = "all" if self.timesteps is None else ",".join(map(str, self.timesteps))
        stride = "" if self.patch_stride <= 1 else f" stride={self.patch_stride}"
        return (
            f"layers={','.join(map(str, self.layers))} timesteps={steps} "
            f"streams={','.join(self.streams)} "
            f"patches={'keep' if self.keep_patches else 'reduce'}{stride}"
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


def subsample_patches(
    matrix: np.ndarray, stride: int, n_frames: int | None
) -> np.ndarray:
    """Keep every ``stride``-th patch, preserving the frame structure.

    A full video capture is large: 27,900 patch tokens at hidden width 3072 is 171MB
    in float16 for *one* (layer, step, stream). A pair of rollouts across several
    layers and steps runs to many gigabytes, which is both a disk problem and a
    network-volume transfer problem.

    The subsampling is done *within* each frame rather than across the flat token
    axis. That matters: :func:`frame_window` splits the token axis into ``n_frames``
    equal blocks, so a flat stride would leave a count that no longer divides evenly
    and every frame window would then raise. Taking a stride inside each frame keeps
    ``per_frame`` identical across frames, so windowing still works.
    """
    if stride <= 1:
        return matrix
    array = np.asarray(matrix, dtype=np.float32)
    if n_frames is None or n_frames <= 1:
        return array[::stride]
    total = array.shape[0]
    if total % n_frames != 0:
        # Not a frame-aligned token count; fall back to a flat stride and say so via
        # the shape, rather than silently reshaping into the wrong blocks.
        return array[::stride]
    per_frame = total // n_frames
    frames = array.reshape(n_frames, per_frame, array.shape[1])
    kept = frames[:, ::stride, :]
    return kept.reshape(n_frames * kept.shape[1], array.shape[1])


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


def _windowed_mean(
    matrix: np.ndarray, frames: str | None, n_frames: int, stream: str
) -> np.ndarray:
    """Mean over a stream's tokens, optionally restricted to a frame range.

    Video tokens are frame-major — ``rope_apply_3d`` flattens ``[F, H, W]`` in that
    order — so a frame window is a contiguous slice. Audio is 1-D: it has a temporal
    extent but no ``(F, H, W)`` grid, and slicing it by "frames" would silently
    average a time range unrelated to the video's frame boundaries. So a frame
    window on audio is refused rather than guessed at.
    """
    if frames and stream != "video":
        raise ValueError(
            f"frame windows are only defined for the video stream, not {stream}"
        )
    return frame_window(matrix, frames, n_frames).mean(axis=0)


@dataclass
class LayerCapture:
    """What was recorded for one (stream, layer, timestep) triple."""

    layer: int
    timestep: int
    n_patches: int
    hidden: int
    #: ``[n_patches, hidden]`` when the schedule kept patches, else ``None``.
    patches: np.ndarray | None = None
    stream: str = "video"

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
            "stream": self.stream,
            "layer": self.layer,
            "timestep": self.timestep,
            "n_patches": self.n_patches,
            "hidden": self.hidden,
            "kept_patches": self.patches is not None,
        }


@dataclass
class ResidualCapture:
    """Accumulates hooked DiT tensors, keyed by (stream, layer, timestep)."""

    schedule: CaptureSchedule
    captures: dict[tuple[str, int, int], LayerCapture] = field(default_factory=dict)
    #: Calls the schedule declined to record. Reported so a schedule that silently
    #: matched nothing is visible rather than looking like a capture that found no
    #: structure.
    skipped: int = 0

    def record(
        self, layer: int, timestep: int, activation: np.ndarray, stream: str = "video"
    ) -> bool:
        """Record one hooked tensor. Returns whether the schedule wanted it."""
        if not self.schedule.wants(layer, timestep, stream):
            self.skipped += 1
            return False
        matrix = _as_patch_matrix(activation)
        matrix = subsample_patches(
            matrix, self.schedule.patch_stride, self.schedule.n_frames
        )
        self.captures[(stream, layer, timestep)] = LayerCapture(
            layer=layer,
            timestep=timestep,
            n_patches=int(matrix.shape[0]),
            hidden=int(matrix.shape[1]),
            patches=matrix if self.schedule.keep_patches else None,
            stream=stream,
        )
        return True

    def layers(self, stream: str = "video") -> list[int]:
        return sorted({layer for s, layer, _ in self.captures if s == stream})

    def timesteps(self, stream: str = "video") -> list[int]:
        return sorted({step for s, _, step in self.captures if s == stream})

    def streams(self) -> list[str]:
        return sorted({s for s, _, _ in self.captures})

    def at(self, layer: int, timestep: int, stream: str = "video") -> LayerCapture:
        key = (stream, layer, timestep)
        if key not in self.captures:
            raise KeyError(
                f"nothing captured for stream={stream} layer {layer} "
                f"timestep {timestep}; available: {sorted(self.captures)}"
            )
        return self.captures[key]

    def direction(
        self,
        layer: int,
        timestep: int,
        other: "ResidualCapture",
        frames: str | None = None,
        n_frames: int = 1,
        stream: str = "video",
    ) -> np.ndarray:
        """Unit direction from ``other``'s state toward this one at (stream, layer, timestep).

        Both captures must have been taken with ``keep_patches`` and with the same
        ``n_frames``, or the frame window would index different spatial regions in
        each and the difference would be meaningless.

        This is the DiT analogue of :func:`wsp.activations.direction_from_contrast`:
        the same subtraction and normalisation, applied to a frame-restricted mean
        over patch tokens instead of a token-window mean over text positions.
        """
        mine = self.at(layer, timestep, stream)
        theirs = other.at(layer, timestep, stream)
        if mine.patches is None or theirs.patches is None:
            raise ValueError("both captures must set keep_patches=True")
        if mine.hidden != theirs.hidden:
            raise ValueError(
                f"hidden size mismatch: {mine.hidden} vs {theirs.hidden}"
            )
        if mine.n_patches != theirs.n_patches:
            raise ValueError(
                f"patch count mismatch: {mine.n_patches} vs {theirs.n_patches}"
            )
        a = _windowed_mean(mine.patches, frames, n_frames, stream)
        b = _windowed_mean(theirs.patches, frames, n_frames, stream)
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
            "streams": self.streams(),
            "layers": {s: self.layers(s) for s in self.streams()},
            "timesteps": {s: self.timesteps(s) for s in self.streams()},
            "captures": [c.summary() for c in self.captures.values()],
        }
