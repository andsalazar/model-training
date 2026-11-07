"""Attach attention-map capture to a real Ovi model.

Why the intercept is ``flash_attention`` and not a module hook
-------------------------------------------------------------
Ovi's ``WanSelfAttention.forward`` calls the module-level ``flash_attention`` function,
imported into ``ovi.modules.model``'s namespace. No module's ``forward`` returns
attention weights, so ``register_forward_hook`` cannot see them. The interceptable object
is the function itself: replacing ``ovi.modules.model.flash_attention`` catches every
self- and cross-attention call in both the fused and unfused paths, because both reach
attention through ``self_attn(...)``.

Why the full matrix is never materialised
-----------------------------------------
A 960x960 5s rollout is 27,900 video tokens. A full ``[heads, 27900, 27900]`` attention
tensor in float32 is 74 GB for one layer. Ovi's own ``attention_with_weights`` reduces to
a ``[Lk]`` saliency vector for exactly this reason. :func:`wsp.attention.
summarize_attention_chunked` computes the same summary fields exactly, in query chunks,
so nothing larger than ``chunk x keys`` is ever resident.

What this module does NOT do
----------------------------
It does not change the model's output. The intercept calls the original
``flash_attention`` and returns its result untouched; the capture is a side effect. A
hook that altered the output would make every measurement a measurement of the hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wsp.attention import (
    AttentionCapture,
    summarize_attention_chunked,
    summarize_attention_torch,
)
from wsp.dit import CaptureSchedule
from wsp.hooks import StepTracker


def _split_heads(tensor: Any) -> Any:
    """Coerce a ``[B, L, H, C]`` attention input to ``[H, L, C]``.

    Ovi's ``flash_attention`` takes ``[B, L, H, C]``. Batch is 1 for a single
    generation; a genuine multi-batch call is rejected rather than averaged, because
    averaging across generations would mix two rollouts into one summary.

    Returns a torch tensor when given one, so the summary can be computed on the GPU.
    Reducing 27,900 keys in numpy took 49 minutes of CPU time for one run; the same
    arithmetic in torch is seconds.
    """
    if hasattr(tensor, "detach") and hasattr(tensor, "transpose"):
        if tensor.ndim != 4:
            raise ValueError(f"expected [B, L, H, C], got shape {tuple(tensor.shape)}")
        if tensor.shape[0] != 1:
            raise ValueError(
                f"expected batch size 1, got {tensor.shape[0]}; "
                "average per-batch rather than across generations"
            )
        # [B, L, H, C] -> [H, L, C]
        return tensor.detach()[0].transpose(0, 1).contiguous()

    array = np.asarray(tensor, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"expected [B, L, H, C], got shape {array.shape}")
    if array.shape[0] != 1:
        raise ValueError(
            f"expected batch size 1, got {array.shape[0]}; "
            "average per-batch rather than across generations"
        )
    return np.ascontiguousarray(array[0].transpose(1, 0, 2))


@dataclass
class AttentionHooks:
    """Intercept ``flash_attention`` and record attention summaries.

    ``schedule`` decides which (layer, timestep) pairs to keep. ``tracker`` numbers the
    distinct timesteps so the CFG double-pass at one ``t`` counts as one step — the same
    convention :class:`wsp.hooks.DitHooks` uses for residual capture, so the two captures
    line up step for step.
    """

    model: Any
    schedule: CaptureSchedule
    capture: AttentionCapture = field(default_factory=AttentionCapture)
    tracker: StepTracker = field(default_factory=StepTracker)
    #: Query rows per chunk. Lower it if the card runs out of memory.
    chunk: int = 512
    #: Set when the intercept could not be installed; ``report`` surfaces it rather than
    #: letting an empty capture look like "attention showed no structure".
    error: str | None = None

    _original: Any = None
    _module: Any = None
    _layer: int = -1
    _fired: int = 0
    _failed: int = 0
    _failures: list[str] = field(default_factory=list)
    #: Whether attach() ever succeeded. Tracked separately from ``_original`` because
    #: detach() clears that, and "was never attached" must not be reported for a hook
    #: that ran and was then detached.
    _attached: bool = False
    #: The step number of the model call currently in flight. Set by the wrapped
    #: ``model.forward``; ``None`` between calls. ``flash_attention`` receives no
    #: timestep argument, so this is how a call is attributed to a denoising step.
    _current_step: int | None = None
    _original_forward: Any = None
    #: Attention calls made outside a model forward, so they had no step to file under.
    _unattributed: int = 0

    def _find_attention_module(self) -> Any:
        """The module whose ``flash_attention`` name the model actually calls."""
        import ovi.modules.model as ovi_model

        if not hasattr(ovi_model, "flash_attention"):
            raise AttributeError(
                "ovi.modules.model has no flash_attention; the intercept point moved"
            )
        return ovi_model

    def attach(self) -> "AttentionHooks":
        """Install the intercept. Idempotent."""
        if self._original is not None:
            return self
        try:
            module = self._find_attention_module()
        except Exception as exc:  # noqa: BLE001 - surfaced in report()
            self.error = f"could not locate flash_attention: {exc}"
            return self

        self._module = module
        self._original = module.flash_attention
        module.flash_attention = self._intercept
        self._wrap_forward()
        self._attached = True
        return self

    def _wrap_forward(self) -> None:
        """Number the denoising step before the blocks run.

        ``flash_attention`` is called with ``q``, ``k``, ``v`` and lengths — never the
        timestep. The only place the timestep is visible is the model's own ``forward``,
        so the step is numbered there and read by the intercept. This mirrors
        :class:`wsp.hooks.DitHooks`, so a residual capture and an attention capture taken
        in the same run agree on what "step 3" means.
        """
        model = self.model
        if not callable(getattr(model, "forward", None)):
            raise AttributeError("model has no callable forward")
        original = model.forward
        self._original_forward = original
        tracker = self.tracker

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            t = kwargs.get("t", args[2] if len(args) > 2 else None)
            if t is not None:
                self._current_step = tracker.observe(t)
            # Reset the layer counter per forward. Without this it increases across the
            # whole run, so a 12-block model would report layer 12 on the second step and
            # every layer beyond the first would be mislabelled.
            self._layer = -1
            return original(*args, **kwargs)

        model.forward = wrapped  # type: ignore[method-assign]

    def _intercept(self, *args: Any, **kwargs: Any) -> Any:
        """Call the real attention, then summarise it. Output is never modified."""
        original = self._original
        if original is None:
            raise RuntimeError("AttentionHooks used before attach()")

        # Capture the inputs before the call: the original may mutate in place.
        q = kwargs.get("q", args[0] if len(args) > 0 else None)
        k = kwargs.get("k", args[1] if len(args) > 1 else None)

        result = original(*args, **kwargs)

        if q is None or k is None:
            return result

        try:
            self._record(q, k)
        except Exception as exc:  # noqa: BLE001 - a capture failure must not break the run
            self._failed += 1
            if len(self._failures) < 5:
                self._failures.append(f"{type(exc).__name__}: {exc}")
        return result

    def _record(self, q: Any, k: Any) -> None:
        """Summarise one attention call and offer it to the schedule."""
        self._fired += 1
        self._layer += 1
        layer = self._layer

        # The timestep is not passed to flash_attention; the wrapped model.forward set
        # _current_step before the blocks ran. A call outside a model forward has no
        # step, and is counted as unattributed rather than silently filed under step 0.
        step = self._current_step
        if step is None:
            self._unattributed += 1
            return

        keep = self.schedule.wants(layer, step)
        if not keep:
            self.capture.skipped += 1
            return

        q_split = _split_heads(q)
        k_split = _split_heads(k)
        # Use the GPU path when the inputs are torch tensors. The numpy path is the
        # reference and stays tested, but it is far too slow for a real 27,900-key call.
        if hasattr(q_split, "detach"):
            summary = summarize_attention_torch(
                q_split,
                k_split,
                layer=layer,
                timestep=step,
                stream="video",
                chunk=self.chunk,
            )
        else:
            summary = summarize_attention_chunked(
                q_split,
                k_split,
                layer=layer,
                timestep=step,
                stream="video",
                chunk=self.chunk,
            )
        self.capture.record_summary(layer, step, summary, stream="video", keep=True)

    def detach(self) -> None:
        """Restore the original ``flash_attention`` and the model's forward."""
        if self._module is not None and self._original is not None:
            self._module.flash_attention = self._original
        if self._original_forward is not None:
            self.model.forward = self._original_forward  # type: ignore[method-assign]
        self._original = None
        self._module = None
        self._original_forward = None

    def __enter__(self) -> "AttentionHooks":
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach()

    def report(self) -> dict:
        """What happened, including whether the intercept ever fired."""
        notes: list[str] = []
        if self.error:
            notes.append(self.error)
        if not self._attached and not self.error:
            notes.append("attach() was never called; nothing was intercepted")
        if self._fired == 0 and not self.error:
            notes.append(
                "the intercept never fired: flash_attention was not called, so this "
                "capture measured nothing"
            )
        if self._failed:
            notes.append(
                f"{self._failed} attention calls failed to summarise; first: "
                f"{self._failures[0]}"
            )
        if self._unattributed:
            notes.append(
                f"{self._unattributed} attention calls happened outside a model "
                "forward and could not be attributed to a denoising step"
            )
        return {
            "fired": self._fired,
            "failed": self._failed,
            "unattributed": self._unattributed,
            "skipped": self.capture.skipped,
            "recorded": len(self.capture.maps),
            "steps_seen": self.tracker.steps_seen,
            # Step index to real timestep value. A 20-step run and a 30-step run visit
            # different noise levels at the same step index -- at index 18 they are at
            # t=233 and t=756 -- so aligning two runs by index compares different points
            # on the trajectory. This mapping is what makes a timestep-aligned contrast
            # possible.
            "timesteps": self.tracker.mapping(),
            "capture": self.capture.summary(),
            "notes": notes,
        }
