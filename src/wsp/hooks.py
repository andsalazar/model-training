"""Attach :mod:`wsp.dit` capture to a real Ovi model.

Why the hook sits on ``norm1`` and not on the block
---------------------------------------------------
The obvious place to hook a transformer is ``blocks[i]``: a forward hook there fires
once per layer and hands you the layer's output. That does not work on Ovi.

Ovi's ``FusionModel.forward`` does not call ``blocks[i](...)``. It calls
``self.single_fusion_block_forward(vid_block=..., audio_block=..., ...)``, a bound
method that reaches inside each block and calls ``block.modulation``,
``block.norm1``, ``block.self_attn``, ``block.norm3``, ``block.cross_attn``,
``block.norm2`` and ``block.ffn`` individually — so that the video block can
cross-attend to the audio stream and vice versa. A hook on the block module would
never fire on the fused path and the capture would come back empty. An empty capture
looks exactly like a capture that found no structure, which is the failure mode this
project keeps hitting.

``norm1`` is called exactly once per block, per stream, per model pass, in both
paths: once in ``single_fusion_block_forward`` (audio then video) and once in
``WanAttentionBlock.forward``. Hooking its *input* therefore yields the residual
stream — the additive highway TBC's adapter modifies — at a site that is present
whether the model is run fused or unfused.

Step numbering
--------------
A diffusion model is called once per denoising step, and with classifier-free
guidance twice per step at the *same* timestep. Counting calls would number the
conditional and unconditional passes as different steps, so the schedule would pick
the wrong pair. :class:`StepTracker` numbers the first distinct timestep value 0,
the second 1, and resolves a repeated timestep back to the step that introduced it.
No assumption is made about the sampler's timestep values, which differ between
schedulers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wsp.dit import ResidualCapture, _as_patch_matrix


def _timestep_key(t: object) -> float:
    """A hashable scalar for a diffusion timestep, whatever the sampler passes.

    Ovi passes ``t`` as a tensor, sometimes shape ``[B]``, sometimes ``[1]``. The
    first element identifies the step. ``float()`` on a 1-element array is
    deprecated, so ``.item()`` is preferred where available.
    """
    if hasattr(t, "flatten"):
        flat = t.flatten()
        if flat.numel() != 1:
            raise ValueError(
                f"expected a scalar timestep, got shape {tuple(t.shape)}; "
                "a per-element timestep would make one pass span several steps"
            )
        return float(flat.item())
    if hasattr(t, "item"):
        return float(t.item())
    return float(t)  # type: ignore[arg-type]


@dataclass
class StepTracker:
    """Maps a diffusion timestep value to a denoising step index.

    The first distinct timestep observed is step 0. A timestep seen again — which is
    what classifier-free guidance does within one step — resolves to the step that
    first introduced it, so a conditional and an unconditional pass land on the same
    index instead of consuming two.
    """

    _index: dict[float, int] = field(default_factory=dict)
    _order: list[float] = field(default_factory=list)
    calls: int = 0

    def observe(self, t: object) -> int:
        """Record a timestep for the current pass and return its step index."""
        key = _timestep_key(t)
        self.calls += 1
        if key not in self._index:
            self._index[key] = len(self._order)
            self._order.append(key)
        return self._index[key]

    @property
    def steps_seen(self) -> int:
        """How many distinct timesteps have been observed — the step count."""
        return len(self._order)

    def timestep_of(self, step: int) -> float | None:
        """The timestep value recorded for a step index, or ``None`` if unseen."""
        if 0 <= step < len(self._order):
            return self._order[step]
        return None

    def mapping(self) -> dict[int, float]:
        """Step index to timestep value, for aligning runs of different lengths."""
        return {i: value for i, value in enumerate(self._order)}


def _first_tensor(value: object) -> object:
    """Ovi passes ``x`` as a list of tensors; take the first."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("block input was an empty list")
        return value[0]
    return value


class DitHooks:
    """Capture Ovi residual states while a generation runs.

    Usage::

        hooks = DitHooks(model, capture)
        with hooks:
            ...run the sampler...
        print(hooks.report())

    ``report()`` distinguishes the three ways this can come back empty: hooks that
    never fired at all (wrong module attribute), hooks that fired but were declined
    by the schedule, and hooks that recorded nothing because the step was unknown.
    Reporting the count is what stops a silent empty capture from reading as a
    negative result.
    """

    def __init__(
        self,
        model: object,
        capture: ResidualCapture,
        *,
        batch_index: int = 0,
    ):
        self.model = model
        self.capture = capture
        self.batch_index = batch_index
        self.tracker = StepTracker()
        self._handles: list[object] = []
        self._original_forward: object | None = None
        #: Per-stream count of hook firings, before the schedule had a say.
        self.fired: dict[str, int] = {}
        #: Firings where the step index could not be resolved.
        self.no_step: int = 0

    # -- module discovery ---------------------------------------------------

    def _stacks(self) -> list[tuple[str, object]]:
        """The per-stream block lists, as ``(stream, blocks)``.

        ``video_model`` and ``audio_model`` are the two WanModel instances Ovi fuses.
        A model with neither is rejected rather than silently hooking nothing.
        """
        stacks: list[tuple[str, object]] = []
        for stream in ("video", "audio"):
            stack = getattr(self.model, f"{stream}_model", None)
            if stack is None:
                continue
            blocks = getattr(stack, "blocks", None)
            if blocks is None:
                raise AttributeError(
                    f"{stream}_model has no .blocks; refusing to hook nothing"
                )
            stacks.append((stream, blocks))
        if not stacks:
            raise AttributeError(
                "model has neither video_model nor audio_model; "
                "DitHooks targets Ovi's FusionModel"
            )
        return stacks

    # -- attachment ---------------------------------------------------------

    def attach(self) -> "DitHooks":
        """Register hooks and wrap the model forward for step numbering."""
        if self._handles:
            raise RuntimeError("already attached; detach() before attaching again")
        schedule = self.capture.schedule
        for stream, blocks in self._stacks():
            for layer in schedule.layers:
                if layer >= len(blocks):
                    raise IndexError(
                        f"layer {layer} requested but {stream} has "
                        f"{len(blocks)} blocks"
                    )
                norm1 = getattr(blocks[layer], "norm1", None)
                if norm1 is None:
                    raise AttributeError(
                        f"{stream} block {layer} has no .norm1; the hook site "
                        "used by this module does not exist on this model"
                    )
                handle = norm1.register_forward_pre_hook(
                    self._make_hook(stream, layer)
                )
                self._handles.append(handle)
        self._wrap_forward()
        return self

    def _make_hook(self, stream: str, layer: int):
        def hook(module, args):  # noqa: ANN001 - torch hook signature
            self._on_norm1(stream, layer, args)

        return hook

    def _on_norm1(self, stream: str, layer: int, args: tuple) -> None:
        self.fired[stream] = self.fired.get(stream, 0) + 1
        step = self._current_step
        if step is None:
            self.no_step += 1
            return
        raw = _first_tensor(args[0] if args else None)
        array = np.asarray(raw.detach().float().cpu().numpy())
        # A multi-batch tensor is sliced to one generation rather than averaged:
        # averaging conditional and unconditional passes would mix two different
        # rollouts into one direction.
        if array.ndim == 3 and array.shape[0] > 1:
            if self.batch_index >= array.shape[0]:
                raise IndexError(
                    f"batch_index {self.batch_index} out of range for "
                    f"batch size {array.shape[0]}"
                )
            array = array[self.batch_index : self.batch_index + 1]
        self.capture.record(layer, step, array, stream=stream)

    def _wrap_forward(self) -> None:
        """Give the model a forward that numbers the step before blocks run."""
        model = self.model
        if not callable(getattr(model, "forward", None)):
            raise AttributeError("model has no callable forward")
        self._current_step: int | None = None
        original = model.forward
        self._original_forward = original

        def wrapped(*args, **kwargs):  # noqa: ANN002, ANN003
            t = kwargs.get("t", args[2] if len(args) > 2 else None)
            if t is None:
                raise ValueError(
                    "could not find the timestep in the model call; DitHooks "
                    "locates it positionally (args[2]) or as the 't' keyword"
                )
            self._current_step = self.tracker.observe(t)
            return original(*args, **kwargs)

        model.forward = wrapped  # type: ignore[method-assign]

    def detach(self) -> None:
        """Remove every hook and restore the original forward. Idempotent."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._original_forward is not None:
            self.model.forward = self._original_forward  # type: ignore[method-assign]
            self._original_forward = None
        self._current_step = None

    def __enter__(self) -> "DitHooks":
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach()

    # -- reporting ----------------------------------------------------------

    def report(self) -> dict:
        """What the hooks did, in enough detail to tell empty from unmeasured."""
        recorded = len(self.capture.captures)
        total_fired = sum(self.fired.values())
        notes: list[str] = []
        if total_fired == 0:
            notes.append(
                "no hook fired: the hook site was never called, so nothing was "
                "measured (this is not evidence that the states are identical)"
            )
        elif recorded == 0:
            notes.append(
                f"{total_fired} hook firings produced no capture; check that the "
                f"schedule matches the observed steps (saw {self.tracker.steps_seen})"
            )
        if self.no_step:
            notes.append(
                f"{self.no_step} firings occurred outside a numbered step and were "
                "not recorded"
            )
        return {
            "attached": bool(self._handles) or self._original_forward is not None,
            "hook_site": "blocks[i].norm1 (forward_pre_hook)",
            "layers": list(self.capture.schedule.layers),
            "streams": list(self.capture.schedule.streams),
            "fired": dict(self.fired),
            "fired_total": total_fired,
            "no_step": self.no_step,
            "decoded_declined": self.capture.skipped,
            "steps_seen": self.tracker.steps_seen,
            "model_calls": self.tracker.calls,
            "step_timesteps": {str(k): v for k, v in self.tracker.mapping().items()},
            "recorded": recorded,
            "notes": notes,
        }
