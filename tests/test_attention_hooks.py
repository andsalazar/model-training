"""Tests for attaching attention capture to a real Ovi model.

The structural fact that matters: Ovi reaches attention through the module-level
``flash_attention`` function, not through a module's ``forward``. These tests use a
stand-in module to prove the intercept catches calls and leaves the output untouched.
"""

import sys
import types

import numpy as np
import pytest

from wsp.attention_hooks import AttentionHooks, _split_heads
from wsp.dit import CaptureSchedule


def _install_fake_ovi(calls):
    """Install a stand-in ``ovi.modules.model`` whose flash_attention records calls."""
    ovi = types.ModuleType("ovi")
    modules = types.ModuleType("ovi.modules")
    model = types.ModuleType("ovi.modules.model")

    def flash_attention(q=None, k=None, v=None, **kwargs):
        calls.append({"q": q, "k": k, "v": v, "kwargs": kwargs})
        # Return something shaped like the real output so a caller can use it.
        return "attention-output"

    model.flash_attention = flash_attention
    ovi.modules = modules
    modules.model = model
    sys.modules["ovi"] = ovi
    sys.modules["ovi.modules"] = modules
    sys.modules["ovi.modules.model"] = model
    return model


@pytest.fixture
def fake_ovi():
    calls = []
    model = _install_fake_ovi(calls)
    yield model, calls
    for name in ("ovi", "ovi.modules", "ovi.modules.model"):
        sys.modules.pop(name, None)


def _underlying(fn):
    """The plain function behind a bound method, or ``fn`` itself.

    ``model.forward`` returns a fresh bound-method object on every attribute access, so
    ``is`` comparisons fail even when the same function is installed. ``attach`` assigns a
    plain function to the instance; ``detach`` restores the bound method it captured. Both
    shapes have to compare equal to the original.
    """
    return getattr(fn, "__func__", fn)


def _qk(heads=2, n=4, dim=8, seed=0):
    gen = np.random.default_rng(seed)
    q = gen.normal(size=(1, n, heads, dim)).astype(np.float32)
    k = gen.normal(size=(1, n, heads, dim)).astype(np.float32)
    return q, k


class _Model:
    """Stands in for Ovi's DiT: a forward that takes a timestep and runs N blocks.

    ``AttentionHooks`` wraps ``forward`` to number the denoising step, because
    ``flash_attention`` never receives the timestep. A stand-in without a forward would
    let the hook attach but never attribute a call to a step.

    ``blocks`` matters: a real DiT calls attention once per block per forward, so the
    layer counter must reset at the start of each forward. A one-block stand-in cannot
    distinguish "resets per forward" from "increments forever".
    """

    def __init__(self, module, calls, blocks=1):
        self.module = module
        self.calls = calls
        self.blocks = blocks
        self.forward_calls = 0

    def forward(self, x, t=None, k=None):
        self.forward_calls += 1
        out = None
        for _ in range(self.blocks):
            # Mirror the real call: reach attention through the module-level function.
            # ``k`` defaults to ``x`` for self-attention; pass a different key tensor to
            # exercise the cross-attention path.
            out = self.module.flash_attention(q=x, k=x if k is None else k, v=x)
        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TestSplitHeads:
    def test_batch_one_is_squeezed_and_transposed(self):
        q, _ = _qk(heads=2, n=4, dim=8)
        array = _split_heads(q)
        # [1, 4, 2, 8] -> [2, 4, 8]
        assert array.shape == (2, 4, 8)

    def test_batch_greater_than_one_is_rejected(self):
        q = np.ones((2, 4, 2, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="batch size 1"):
            _split_heads(q)

    def test_wrong_rank_is_rejected(self):
        with pytest.raises(ValueError, match="expected"):
            _split_heads(np.ones((4, 2, 8), dtype=np.float32))


class TestAttach:
    def test_intercept_fires_on_a_call(self, fake_ovi):
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls)
        hooks = AttentionHooks(model=model, schedule=schedule)
        q, k = _qk()
        with hooks:
            model.forward(q, t=1.0)
        assert len(calls) == 1
        assert hooks.report()["fired"] == 1

    def test_output_is_not_modified(self, fake_ovi):
        # The intercept must be a pure side effect. A hook that changed the output would
        # make every measurement a measurement of the hook.
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls)
        hooks = AttentionHooks(model=model, schedule=schedule)
        q, k = _qk()
        with hooks:
            result = model.forward(q, t=1.0)
        assert result == "attention-output"

    def test_detach_restores_the_original(self, fake_ovi):
        module, calls = fake_ovi
        original = module.flash_attention
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls)
        original_forward = _underlying(model.forward)
        hooks = AttentionHooks(model=model, schedule=schedule)
        hooks.attach()
        assert module.flash_attention is not original
        assert _underlying(model.forward) is not original_forward
        hooks.detach()
        assert module.flash_attention is original
        assert _underlying(model.forward) is original_forward

    def test_attach_is_idempotent(self, fake_ovi):
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls)
        hooks = AttentionHooks(model=model, schedule=schedule)
        hooks.attach()
        first = module.flash_attention
        first_forward = _underlying(model.forward)
        hooks.attach()
        # A second attach must not wrap the wrapper; that would double-count every call.
        assert module.flash_attention is first
        assert _underlying(model.forward) is first_forward

    def test_missing_intercept_point_is_reported(self):
        # A module without flash_attention must surface as an error, not as an empty
        # capture that reads like "attention showed no structure".
        ovi = types.ModuleType("ovi")
        modules = types.ModuleType("ovi.modules")
        model_mod = types.ModuleType("ovi.modules.model")
        ovi.modules = modules
        modules.model = model_mod
        sys.modules["ovi"] = ovi
        sys.modules["ovi.modules"] = modules
        sys.modules["ovi.modules.model"] = model_mod
        try:
            hooks = AttentionHooks(
                model=_Model(model_mod, []),
                schedule=CaptureSchedule(layers=(0,), timesteps=(0,)),
            )
            hooks.attach()
            report = hooks.report()
            assert report["fired"] == 0
            assert any("could not locate" in n for n in report["notes"])
        finally:
            for name in ("ovi", "ovi.modules", "ovi.modules.model"):
                sys.modules.pop(name, None)


class TestRecording:
    def test_records_only_scheduled_layers(self, fake_ovi):
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(1,), timesteps=(0,))
        # Three blocks per forward: layer 0, 1, 2. Only layer 1 is scheduled.
        model = _Model(module, calls, blocks=3)
        hooks = AttentionHooks(model=model, schedule=schedule)
        q, k = _qk()
        with hooks:
            model.forward(q, t=1.0)
        report = hooks.report()
        assert report["fired"] == 3
        assert report["recorded"] == 1
        assert report["skipped"] == 2

    def test_recorded_summary_has_real_entropy(self, fake_ovi):
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls)
        hooks = AttentionHooks(model=model, schedule=schedule)
        q, k = _qk(heads=2, n=4, dim=8)
        with hooks:
            model.forward(q, t=1.0)
        summary = hooks.capture.at(0, 0)
        # A random q/k gives a non-degenerate distribution: entropy strictly between 0
        # and log(4).
        assert 0.0 < summary.row_entropy.mean() < np.log(4.0)
        assert summary.heads == 2
        assert summary.queries == 4
        assert summary.keys == 4

    def test_capture_failure_does_not_break_the_run(self, fake_ovi):
        # A malformed q must be caught and counted, not raised into the model's forward
        # pass. A capture that crashes generation is worse than no capture.
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls)
        hooks = AttentionHooks(model=model, schedule=schedule)
        with hooks:
            result = model.forward(np.ones((4, 2, 8)), t=1.0)
        assert result == "attention-output"
        report = hooks.report()
        assert report["failed"] == 1
        assert any("failed to summarise" in n for n in report["notes"])

    def test_detached_hook_does_not_claim_it_was_never_attached(self, fake_ovi):
        # detach() clears _original, so a naive "was _original set?" check reports
        # "attach() was never called" for a hook that ran and was then detached. That
        # would mislabel a successful capture as a failed one.
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls)
        hooks = AttentionHooks(model=model, schedule=schedule)
        q, k = _qk()
        with hooks:
            model.forward(q, t=1.0)
        report = hooks.report()
        assert report["fired"] == 1
        assert not any("never called" in n for n in report["notes"])

    def test_never_firing_is_reported(self, fake_ovi):
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls)
        hooks = AttentionHooks(model=model, schedule=schedule)
        with hooks:
            pass  # no attention call at all
        report = hooks.report()
        assert report["fired"] == 0
        assert any("never fired" in n for n in report["notes"])

    def test_cross_attention_is_recorded_without_diagonal(self, fake_ovi):
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls)
        hooks = AttentionHooks(model=model, schedule=schedule)
        gen = np.random.default_rng(2)
        q = gen.normal(size=(1, 4, 2, 8)).astype(np.float32)
        k = gen.normal(size=(1, 9, 2, 8)).astype(np.float32)
        with hooks:
            model.forward(q, t=1.0, k=k)
        summary = hooks.capture.at(0, 0)
        assert summary.queries == 4
        assert summary.keys == 9
        assert np.allclose(summary.diagonal_mass, 0.0)

    def test_step_is_numbered_from_the_model_forward(self, fake_ovi):
        # flash_attention never receives the timestep, so the step must come from the
        # wrapped model.forward. Two calls at the same t are one step (the CFG
        # double-pass); a new t starts a new step.
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0, 1), timesteps=(0, 1))
        # Two blocks per forward, so each forward fires layer 0 then layer 1. The layer
        # counter must reset per forward or the second step would report layers 2 and 3.
        model = _Model(module, calls, blocks=2)
        hooks = AttentionHooks(model=model, schedule=schedule)
        q, k = _qk()
        with hooks:
            model.forward(q, t=1.0)  # step 0: layers 0, 1
            model.forward(q, t=0.5)  # step 1: layers 0, 1
        report = hooks.report()
        assert report["steps_seen"] == 2
        assert report["recorded"] == 4
        # Both steps must have both layers, which only holds if the counter reset.
        assert hooks.capture.layers() == [0, 1]
        assert hooks.capture.timesteps() == [0, 1]

    def test_torch_inputs_take_the_gpu_path(self, fake_ovi):
        # The numpy reduction took 49 minutes of CPU time for one real run. When the
        # inputs are torch tensors the hook must use the torch path, or the GPU run is
        # unusable.
        import torch

        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls)
        hooks = AttentionHooks(model=model, schedule=schedule)
        gen = np.random.default_rng(0)
        q = torch.from_numpy(gen.normal(size=(1, 4, 2, 8)).astype(np.float32))
        with hooks:
            model.forward(q, t=1.0)
        summary = hooks.capture.at(0, 0)
        assert summary.heads == 2
        assert summary.queries == 4
        assert np.all(np.isfinite(summary.row_entropy))

    def test_call_outside_a_forward_is_unattributed(self, fake_ovi):
        # A call with no step in flight cannot be filed under a timestep. Counting it as
        # unattributed is honest; filing it under step 0 would invent a measurement.
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls)
        hooks = AttentionHooks(model=model, schedule=schedule)
        q, k = _qk()
        with hooks:
            module.flash_attention(q=q, k=k, v=q)
        report = hooks.report()
        assert report["fired"] == 1
        assert report["unattributed"] == 1
        assert report["recorded"] == 0
        assert any("outside a model forward" in n for n in report["notes"])
