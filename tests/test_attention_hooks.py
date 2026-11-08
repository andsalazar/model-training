"""Tests for attaching attention capture to a real Ovi model.

The structural fact that matters: Ovi reaches attention through the module-level
``flash_attention`` function, not through a module's ``forward``. These tests use a
stand-in module to prove the intercept catches calls and leaves the output untouched.
"""

import sys
import types

import numpy as np
import pytest
import torch

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


class _SelfAttn(torch.nn.Module):
    """Stands in for ``WanSelfAttention``: reaches attention via the module function."""

    def __init__(self, module):
        super().__init__()
        self._ovi = module

    def forward(self, q, k):
        return self._ovi.flash_attention(q=q, k=k, v=q)


class _Block:
    def __init__(self, module):
        self.self_attn = _SelfAttn(module)


class _Stack:
    def __init__(self, module, n):
        self.blocks = [_Block(module) for _ in range(n)]


class _Model:
    """Stands in for Ovi's FusionModel: two stacks, called audio-then-video per block.

    The order matters. The real fused block calls ``audio_block.self_attn`` before
    ``vid_block.self_attn``, and an earlier hook that counted calls labelled the
    interleaved sequence as one stream's layers. A fake that only had one stream could
    not catch that.

    ``stray`` makes one extra attention call per forward outside any ``self_attn`` --
    the shape of an unfused cross-attention call -- which must be counted, not filed.
    """

    def __init__(self, module, calls, blocks=1, stray=False):
        self.module = module
        self.calls = calls
        self.video_model = _Stack(module, blocks)
        self.audio_model = _Stack(module, blocks)
        self.n_blocks = blocks
        self.stray = stray
        self.forward_calls = 0

    def forward(self, x, t=None, k=None, audio=None):
        self.forward_calls += 1
        a = x if audio is None else audio
        out = None
        for i in range(self.n_blocks):
            self.audio_model.blocks[i].self_attn(a, a)
            # ``k`` defaults to ``x``; pass a different key tensor to exercise q != k.
            out = self.video_model.blocks[i].self_attn(x, x if k is None else k)
        if self.stray:
            self.module.flash_attention(q=x, k=x, v=x)
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
        # One audio and one video self-attention call per block.
        assert len(calls) == 2
        assert hooks.report()["fired"] == 2

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
        # Three blocks per forward in each of two streams. Only block 1 is scheduled.
        model = _Model(module, calls, blocks=3)
        hooks = AttentionHooks(model=model, schedule=schedule)
        q, k = _qk()
        with hooks:
            model.forward(q, t=1.0)
        report = hooks.report()
        assert report["fired"] == 6
        assert report["recorded"] == 2
        assert report["skipped"] == 4

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
        assert report["failed"] == 2
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
        assert report["fired"] == 2
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
        model = _Model(module, calls, blocks=2)
        hooks = AttentionHooks(model=model, schedule=schedule)
        q, k = _qk()
        with hooks:
            model.forward(q, t=1.0)  # step 0: blocks 0, 1 in both streams
            model.forward(q, t=0.5)  # step 1: blocks 0, 1 in both streams
        report = hooks.report()
        assert report["steps_seen"] == 2
        assert report["recorded"] == 8
        for stream in ("video", "audio"):
            assert hooks.capture.layers(stream) == [0, 1]
            assert hooks.capture.timesteps(stream) == [0, 1]

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


class TestLabelling:
    def test_streams_and_blocks_are_labelled_by_module_not_call_order(self, fake_ovi):
        # The regression: counting calls per forward labelled audio block i as "layer
        # 2i" and video block i as "layer 2i+1", all filed as video. Give the two
        # streams different token counts so a mislabel is visible in the shape.
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0, 1, 2), timesteps=(0,))
        model = _Model(module, calls, blocks=3)
        hooks = AttentionHooks(model=model, schedule=schedule)
        video, _ = _qk(n=6, seed=1)
        audio, _ = _qk(n=3, seed=2)
        with hooks:
            model.forward(video, t=1.0, audio=audio)
        for block in range(3):
            v = hooks.capture.at(block, 0, stream="video")
            a = hooks.capture.at(block, 0, stream="audio")
            assert (v.stream, v.layer, v.queries) == ("video", block, 6)
            assert (a.stream, a.layer, a.queries) == ("audio", block, 3)

    def test_call_outside_self_attn_is_unlabelled_not_filed(self, fake_ovi):
        module, calls = fake_ovi
        schedule = CaptureSchedule(layers=(0,), timesteps=(0,))
        model = _Model(module, calls, stray=True)
        hooks = AttentionHooks(model=model, schedule=schedule)
        q, _ = _qk()
        with hooks:
            model.forward(q, t=1.0)
        report = hooks.report()
        assert report["fired"] == 3
        assert report["unlabelled"] == 1
        assert report["recorded"] == 2
        assert any("outside any hooked self_attn" in n for n in report["notes"])

    def test_model_without_stacks_is_an_error(self, fake_ovi):
        module, calls = fake_ovi

        class _Bare:
            def forward(self, x, t=None):
                return module.flash_attention(q=x, k=x, v=x)

        hooks = AttentionHooks(
            model=_Bare(), schedule=CaptureSchedule(layers=(0,), timesteps=(0,))
        )
        hooks.attach()
        assert hooks.error is not None
        assert "neither video_model nor audio_model" in hooks.error

    def test_detach_removes_module_hooks(self, fake_ovi):
        module, calls = fake_ovi
        model = _Model(module, calls, blocks=2)
        hooks = AttentionHooks(
            model=model, schedule=CaptureSchedule(layers=(0,), timesteps=(0,))
        )
        hooks.attach()
        attn = model.video_model.blocks[1].self_attn
        assert attn._forward_pre_hooks and attn._forward_hooks
        hooks.detach()
        assert not attn._forward_pre_hooks and not attn._forward_hooks


class TestCfgPass:
    def _two_pass(self, fake_ovi, cfg_pass):
        # Ovi runs the conditional pass and then the unconditional pass at the same t.
        # They share a capture key, so without a pass filter the second overwrote the
        # first and every "measurement" was of the unconditional branch.
        module, calls = fake_ovi
        model = _Model(module, calls)
        hooks = AttentionHooks(
            model=model,
            schedule=CaptureSchedule(layers=(0,), timesteps=(0,), streams=("video",)),
            cfg_pass=cfg_pass,
        )
        cond, _ = _qk(n=4, seed=10)
        uncond, _ = _qk(n=4, seed=11)
        with hooks:
            model.forward(cond, t=1.0)
            model.forward(uncond, t=1.0)
        return hooks, cond, uncond

    def _entropy(self, q):
        from wsp.attention import summarize_attention_chunked

        split = _split_heads(q)
        return summarize_attention_chunked(split, split).row_entropy

    def test_default_records_the_conditional_pass(self, fake_ovi):
        hooks, cond, uncond = self._two_pass(fake_ovi, cfg_pass=0)
        got = hooks.capture.at(0, 0).row_entropy
        assert np.allclose(got, self._entropy(cond))
        assert not np.allclose(got, self._entropy(uncond))
        report = hooks.report()
        assert report["cfg_pass"] == "conditional"
        assert report["other_pass_declined"] == 2

    def test_pass_one_records_the_unconditional_pass(self, fake_ovi):
        hooks, cond, uncond = self._two_pass(fake_ovi, cfg_pass=1)
        assert np.allclose(hooks.capture.at(0, 0).row_entropy, self._entropy(uncond))
        assert hooks.report()["cfg_pass"] == "unconditional"
