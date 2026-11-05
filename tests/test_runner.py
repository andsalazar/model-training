"""Tests for loop detection and the runner's classification logic."""

from __future__ import annotations

import pytest

from wsp.runner import NullRunner, classify, detect_repetition


def test_detect_repetition_finds_repeated_tail() -> None:
    block = "the same reasoning step over and over again " * 8
    text = "start " + block + block + block
    assert detect_repetition(text)


def test_detect_repetition_ignores_short_text() -> None:
    assert not detect_repetition("short")


def test_detect_repetition_ignores_varied_text() -> None:
    text = " ".join(f"word{i}" for i in range(600))
    assert not detect_repetition(text)


def test_classify_hit_cap_alone_is_not_looping() -> None:
    """Running to the cap is truncation, not evidence of a loop.

    This checkpoint reaches a 64-token cap even on "Hello, how are you today?"
    while producing a coherent greeting. Counting that as a loop made explore10's
    editor conclude no non-looping math control existed. ``hit_cap`` is still
    reported so truncation stays visible, but it does not imply looping.
    """
    signal = classify("some text", n_tokens=2048, max_tokens=2048)
    assert signal.hit_cap
    assert not signal.looped
    assert not signal.repeated


def test_classify_hit_cap_with_repetition_is_looping() -> None:
    block = "the same reasoning step over and over again " * 8
    signal = classify(block * 3, n_tokens=2048, max_tokens=2048)
    assert signal.hit_cap
    assert signal.repeated
    assert signal.looped


def test_classify_clean_stop_is_not_looping() -> None:
    signal = classify("a short answer", n_tokens=12, max_tokens=2048)
    assert not signal.looped
    assert not signal.hit_cap


def test_classify_repetition_without_cap_is_looping() -> None:
    block = "circling back to the same point " * 10
    signal = classify(block * 3, n_tokens=100, max_tokens=2048)
    assert signal.looped
    assert signal.repeated
    assert not signal.hit_cap


def test_signal_serializes() -> None:
    d = classify("x", 1, 10).as_dict()
    assert set(d) == {"looped", "hit_cap", "repeated", "tokens", "text_chars"}


def test_null_runner_generate_raises() -> None:
    with pytest.raises(RuntimeError, match="no model attached"):
        NullRunner().generate("p")


def test_null_runner_measure_raises() -> None:
    with pytest.raises(RuntimeError, match="no model attached"):
        NullRunner().measure("p")


def test_null_runner_reload_raises() -> None:
    with pytest.raises(RuntimeError, match="no model attached"):
        NullRunner().reload("x")


def test_write_edited_checkpoint_uses_index_shard_name(tmp_path) -> None:
    """A sharded checkpoint's index names its shard; the write must match it.

    Writing 'model.safetensors' while the index points at
    'model.safetensors-00001-of-00001.safetensors' makes vLLM fail to load.
    """
    import json

    import numpy as np
    from safetensors.numpy import save_file

    from wsp.runner import write_edited_checkpoint

    src = tmp_path / "src"
    src.mkdir()
    shard = "model.safetensors-00001-of-00001.safetensors"
    save_file({"w": np.ones((2, 2), dtype=np.float32)}, str(src / shard))
    (src / "config.json").write_text("{}", encoding="utf-8")
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": shard}}), encoding="utf-8"
    )

    out = write_edited_checkpoint(
        src / shard, "w", np.zeros((2, 2), dtype=np.float32), tmp_path / "out"
    )

    assert (out / shard).exists(), "must write under the indexed shard name"
    assert not (out / "model.safetensors").exists()
    assert not (out / "model.safetensors.index.json").exists(), "stale index removed"


def test_write_edited_checkpoint_keeps_plain_name_without_index(tmp_path) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    from wsp.runner import write_edited_checkpoint

    src = tmp_path / "src"
    src.mkdir()
    save_file({"w": np.ones((2, 2), dtype=np.float32)}, str(src / "model.safetensors"))

    out = write_edited_checkpoint(
        src / "model.safetensors", "w", np.zeros((2, 2), dtype=np.float32), tmp_path / "out"
    )
    assert (out / "model.safetensors").exists()


# -- activation capture ----------------------------------------------------


def test_module_path_strips_weight_suffix() -> None:
    from wsp.runner import module_path_for_tensor

    assert (
        module_path_for_tensor("model.language_model.layers.16.mlp.down_proj.weight")
        == "model.language_model.layers.16.mlp.down_proj"
    )


def test_module_path_passes_through_a_module_name() -> None:
    from wsp.runner import module_path_for_tensor

    assert module_path_for_tensor("model.layers.0.mlp.down_proj") == "model.layers.0.mlp.down_proj"


def test_module_path_rejects_empty() -> None:
    from wsp.runner import module_path_for_tensor

    with pytest.raises(ValueError, match="empty"):
        module_path_for_tensor("")


def test_null_capturer_raises() -> None:
    from wsp.runner import NullActivationCapturer

    with pytest.raises(RuntimeError, match="no model attached"):
        NullActivationCapturer().capture("p", "m")


def test_null_capturer_reload_raises() -> None:
    from wsp.runner import NullActivationCapturer

    with pytest.raises(RuntimeError, match="no model attached"):
        NullActivationCapturer().reload("x")
