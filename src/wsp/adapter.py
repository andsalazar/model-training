"""A TBC-style residual adapter for Ovi, and the sampler plumbing to train it.

The adapter
-----------
TBC describe a ~156k-parameter residual module in the early layers of a DiT. This is
``x + up(gelu(down(layernorm(x))))`` with ``up`` zero-initialised, so an untrained adapter
leaves the model exactly as it was. One adapter per stream per chosen block, applied to
the residual stream *before* that fusion block runs.

"TBC-style" means sized and placed like TBC's description, nothing more. This is a
standard bottleneck adapter. TBC's module is built from "localized spatial activations
and a learned projection" derived from neural recordings and was trained alongside
fine-tuning of the base model; they published no code, so none of that is reproduced.

Ovi's fused loop calls ``self.single_fusion_block_forward`` by attribute (through
``gradient_checkpointing``), so replacing that attribute on the instance reaches every
block in both the checkpointed and the plain path.

What the adapter is trained to do
---------------------------------
Make a few-step sampler land where a many-step sampler lands. Under Ovi's UniPC grid
(shift 5) the 12-step timesteps are exactly every 4th 48-step timestep, so a 48-step
"teacher" run passes through every state a 12-step "student" visits. For each student
interval ``[sigma_k, sigma_{k+1}]`` the target is the velocity that carries the teacher's
state at ``sigma_k`` to the teacher's state at ``sigma_{k+1}`` in one Euler step::

    v*_k = (x_{k+1} - x_k) / (sigma_{k+1} - sigma_k)

Ovi's flow convention is ``x0 = x - sigma * v`` (checked against
``FlowUniPCMultistepScheduler.convert_model_output``), so Euler is
``x_next = x + (sigma_next - sigma) * v``.

The student therefore samples with Euler on the UniPC grid, not with UniPC: UniPC's
multistep correction extrapolates from previous outputs, so it would not land on the
target even if the adapter were perfect. The honest baselines are base-model UniPC at 12
steps and base-model UniPC at 30 (the current setting), not only base-model Euler.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch
from torch import nn


class ResidualAdapter(nn.Module):
    """``x + up(gelu(down(layernorm(x))))``, identity at initialisation."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(torch.nn.functional.gelu(self.down(self.norm(x.float()))))
        return x + h.to(x.dtype)


class OviAdapters(nn.Module):
    """The adapters for a set of blocks in both streams, with an on/off switch.

    ``enabled = False`` makes the wrapped forward call the original block untouched,
    which is how the same loaded model serves as both the adapted student and the
    base-model baseline in one process. ``video_enabled`` / ``audio_enabled`` switch one
    stream off while the other stays on (the Stage 7 flash fix holds the audio adapter
    off for the first sampling steps).
    """

    def __init__(self, video_dim: int, audio_dim: int, blocks: list[int], rank: int):
        super().__init__()
        self.blocks = list(blocks)
        self.rank = rank
        self.video = nn.ModuleDict({str(i): ResidualAdapter(video_dim, rank) for i in blocks})
        self.audio = nn.ModuleDict({str(i): ResidualAdapter(audio_dim, rank) for i in blocks})
        self.enabled = True
        self.video_enabled = True
        self.audio_enabled = True

    def apply(self, block: int, vid: torch.Tensor, audio: torch.Tensor):  # noqa: A003
        key = str(block)
        if not self.enabled or key not in self.video:
            return vid, audio
        if self.video_enabled:
            vid = self.video[key](vid)
        if self.audio_enabled:
            audio = self.audio[key](audio)
        return vid, audio

    def reset(self) -> None:
        """Return every adapter to its identity initialisation."""
        for module in list(self.video.values()) + list(self.audio.values()):
            nn.init.zeros_(module.up.weight)
            nn.init.zeros_(module.up.bias)

    def load_partial(self, state: dict) -> list[str]:
        """Load a state dict that may cover only some blocks; the rest become identity.

        Returns the blocks left at identity, so a caller can report them. Lets one
        installed adapter set (e.g. blocks 0-3) evaluate a checkpoint trained on 0-2.
        """
        self.reset()
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise KeyError(f"state has blocks this adapter set lacks: {result.unexpected_keys[:4]}")
        return sorted({k.split(".")[1] for k in result.missing_keys if k.endswith("up.weight")})

    def describe(self) -> dict:
        return {
            "blocks": self.blocks,
            "rank": self.rank,
            "params": sum(p.numel() for p in self.parameters()),
        }


class StepGate:
    """Switch the audio adapters off for chosen sampling steps.

    Wraps ``model.forward`` and numbers steps by distinct timestep value, so the two CFG
    passes at one timestep count as one step (the same convention as
    ``wsp.hooks.StepTracker``). Call :meth:`start` before each generation.
    """

    def __init__(self, model: Any, adapters: OviAdapters) -> None:
        self.adapters = adapters
        self.audio_off_steps: set[int] = set()
        self.step = -1
        self._last_t: float | None = None
        original = model.forward
        gate = self

        def forward(*args: Any, **kwargs: Any):
            t = kwargs.get("t", args[2] if len(args) > 2 else None)
            value = float(t.flatten()[0]) if t is not None else None
            if value != gate._last_t:
                gate.step += 1
                gate._last_t = value
            adapters.audio_enabled = gate.step not in gate.audio_off_steps
            return original(*args, **kwargs)

        model.forward = forward

    def start(self, audio_off_steps: set[int] | None = None) -> None:
        self.audio_off_steps = set(audio_off_steps or ())
        self.step, self._last_t = -1, None
        self.adapters.audio_enabled = True


def install_adapters(model: Any, blocks: list[int], rank: int) -> OviAdapters:
    """Attach adapters to an Ovi ``FusionModel`` and return them.

    Raises if the model does not have the fused-loop attribute this relies on, rather
    than attaching adapters that would never run.
    """
    if not callable(getattr(model, "single_fusion_block_forward", None)):
        raise AttributeError(
            "model has no single_fusion_block_forward; the adapter hook site moved"
        )
    video_blocks = model.video_model.blocks
    adapters = OviAdapters(
        video_dim=video_blocks[0].dim,
        audio_dim=model.audio_model.blocks[0].dim,
        blocks=blocks,
        rank=rank,
    )
    for i in blocks:
        if i >= len(video_blocks):
            raise IndexError(f"block {i} requested but the model has {len(video_blocks)}")
    index = {id(b): i for i, b in enumerate(video_blocks)}
    original = model.single_fusion_block_forward

    def wrapped(*args: Any, vid_block: Any, audio_block: Any, vid: Any, audio: Any, **kwargs: Any):
        vid, audio = adapters.apply(index[id(vid_block)], vid, audio)
        return original(
            *args, vid_block=vid_block, audio_block=audio_block, vid=vid, audio=audio, **kwargs
        )

    model.single_fusion_block_forward = wrapped
    model._wsp_adapters = adapters
    return adapters


def target_velocity(
    x_k: torch.Tensor, x_next: torch.Tensor, sigma_k: float, sigma_next: float
) -> torch.Tensor:
    """The velocity that takes ``x_k`` to ``x_next`` in one Euler step."""
    step = sigma_next - sigma_k
    if abs(step) < 1e-12:
        raise ValueError("sigma_k and sigma_next are equal; the interval has no length")
    return (x_next.float() - x_k.float()) / step


def normalised_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE divided by the target's mean square, so video and audio are on one scale.

    The two streams' latents differ in size and magnitude; without normalising, the
    video term (27,900 x 48 entries) would swamp the audio term.
    """
    target = target.float()
    return ((prediction.float() - target) ** 2).mean() / target.pow(2).mean().clamp_min(1e-12)


def normalised_mse_grad(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`normalised_mse` and its gradient with respect to ``prediction``.

    Used to train through classifier-free guidance one pass at a time. The guided
    output is ``g = s * pos + (1 - s) * neg``, so ``dL/dpos = s * dL/dg`` and
    ``dL/dneg = (1 - s) * dL/dg``: each pass can be backpropagated separately with that
    upstream gradient, holding only one pass's activations at once. At 10 s a single
    pass already peaks at 67 GB, so the two cannot be held together.
    """
    diff = prediction.float() - target.float()
    denom = target.float().pow(2).mean().clamp_min(1e-12)
    loss = diff.pow(2).mean() / denom
    grad = 2.0 * diff / (diff.numel() * denom)
    return loss, grad


@dataclass
class EulerGridScheduler:
    """Euler steps on a fixed timestep/sigma grid, with the scheduler ``step`` interface.

    ``timesteps`` has one entry per step and ``sigmas`` one more (the final 0). Ovi's
    ``generate`` calls ``step(model_output, t, sample, return_dict=False)[0]``.
    """

    timesteps: list[float]
    sigmas: list[float]
    calls: int = 0

    def __post_init__(self) -> None:
        if len(self.sigmas) != len(self.timesteps) + 1:
            raise ValueError(
                f"need one more sigma than timesteps, got {len(self.sigmas)} and "
                f"{len(self.timesteps)}"
            )

    def index_of(self, timestep: Any) -> int:
        value = float(timestep)
        best = min(range(len(self.timesteps)), key=lambda i: abs(self.timesteps[i] - value))
        if abs(self.timesteps[best] - value) > 0.5:
            raise ValueError(f"timestep {value} is not on the grid {self.timesteps}")
        return best

    def step(self, model_output: torch.Tensor, timestep: Any, sample: torch.Tensor,
             return_dict: bool = False, **_: Any):
        i = self.index_of(timestep)
        self.calls += 1
        out = sample.float() + (self.sigmas[i + 1] - self.sigmas[i]) * model_output.float()
        out = out.to(sample.dtype)
        return (out,) if not return_dict else {"prev_sample": out}


@dataclass
class RecordingScheduler:
    """Wraps a scheduler and records the sample at chosen timesteps, plus the final one.

    ``record`` holds the timesteps whose *input* sample is kept (the state at that noise
    level). The output of the scheduler's last step is kept as ``final``.
    """

    inner: Any
    record: list[float]
    n_steps: int
    samples: dict[float, torch.Tensor] = field(default_factory=dict)
    final: torch.Tensor | None = None
    calls: int = 0

    def step(self, model_output: torch.Tensor, timestep: Any, sample: torch.Tensor,
             return_dict: bool = False, **kwargs: Any):
        value = float(timestep)
        for r in self.record:
            if abs(r - value) <= 0.5:
                self.samples[r] = sample.detach()[0].to("cpu").clone()
        result = self.inner.step(model_output, timestep, sample, return_dict=False, **kwargs)
        self.calls += 1
        if self.calls == self.n_steps:
            self.final = result[0].detach()[0].to("cpu").clone()
        return result

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (timesteps, sigmas, config) to the wrapped scheduler.
        # Guard "inner" itself: during copy/unpickle it is looked up before it is set,
        # and delegating that lookup would recurse forever.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


@contextmanager
def patched_schedulers(
    engine: Any, factory: Callable[[str, Any, Any], Any]
) -> Iterator[list[Any]]:
    """Make ``engine.generate`` use custom schedulers.

    ``generate`` calls ``get_scheduler_time_steps`` twice, video first and audio second.
    ``factory(stream, scheduler, timesteps)`` receives each real scheduler and returns the
    one to use. The created schedulers are yielded so the caller can read recordings.
    """
    original = engine.get_scheduler_time_steps
    made: list[Any] = []

    def replacement(*args: Any, **kwargs: Any):
        scheduler, timesteps = original(*args, **kwargs)
        stream = "video" if len(made) % 2 == 0 else "audio"
        custom = factory(stream, scheduler, timesteps)
        made.append(custom)
        return custom, timesteps

    engine.get_scheduler_time_steps = replacement
    try:
        yield made
    finally:
        engine.get_scheduler_time_steps = original
