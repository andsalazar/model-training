"""End-to-end tests for the transformers activation capturer.

These build a tiny real Qwen model on CPU and run the actual hook, so the code
path that runs on a pod is exercised here rather than only mocked. The model is
a few hundred KB, so this stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from wsp.runner import TransformersActivationCapturer  # noqa: E402

_HIDDEN = 32
_INTERMEDIATE = 64


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A tiny Qwen checkpoint on disk, with a tokenizer."""
    path = tmp_path_factory.mktemp("tiny-qwen")

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(str(path))

    # A byte-level tokenizer needs a vocab file; build one from the model's
    # vocab size so the fixture is self-contained.
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    vocab = {f"tok{i}": i for i in range(126)}
    vocab["[UNK]"] = 126
    vocab["[PAD]"] = 127
    tokenizer.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]",'
        ' "pad_token": "[PAD]"}',
        encoding="utf-8",
    )
    AutoTokenizer.from_pretrained(str(path))
    return path


def test_capture_returns_tokens_by_hidden(tiny_model_dir) -> None:
    capturer = TransformersActivationCapturer(str(tiny_model_dir), device="cpu")
    acts = capturer.capture("tok1 tok2 tok3", "model.layers.0.mlp.down_proj")

    assert isinstance(acts, np.ndarray)
    assert acts.ndim == 2
    assert acts.shape[1] == _HIDDEN, "hidden dim must match the model"
    assert acts.shape[0] >= 1, "at least one token"
    assert np.all(np.isfinite(acts))


def test_capture_works_on_a_weight_tensor_name(tiny_model_dir) -> None:
    """An editor addresses a tensor; the capturer must accept that name."""
    capturer = TransformersActivationCapturer(str(tiny_model_dir), device="cpu")
    acts = capturer.capture("tok1 tok2", "model.layers.0.mlp.down_proj.weight")
    assert acts.shape[1] == _HIDDEN


def test_capture_differs_between_layers(tiny_model_dir) -> None:
    """Different layers must produce different activations.

    This is the check explore5's claim failed: it reported byte-identical values
    across layers 0, 6, 12 and 23. A real hook cannot do that.
    """
    capturer = TransformersActivationCapturer(str(tiny_model_dir), device="cpu")
    layer0 = capturer.capture("tok1 tok2 tok3", "model.layers.0.mlp.down_proj")
    layer1 = capturer.capture("tok1 tok2 tok3", "model.layers.1.mlp.down_proj")

    assert layer0.shape == layer1.shape
    assert not np.allclose(layer0, layer1), "layers must not return identical data"


def test_capture_differs_between_prompts(tiny_model_dir) -> None:
    capturer = TransformersActivationCapturer(str(tiny_model_dir), device="cpu")
    a = capturer.capture("tok1 tok2", "model.layers.0.mlp.down_proj")
    b = capturer.capture("tok5 tok6", "model.layers.0.mlp.down_proj")
    assert not np.allclose(a, b), "different prompts must give different states"


def test_capture_rejects_a_bad_module_path(tiny_model_dir) -> None:
    capturer = TransformersActivationCapturer(str(tiny_model_dir), device="cpu")
    with pytest.raises((AttributeError, ValueError)):
        capturer.capture("tok1", "model.layers.99.mlp.down_proj")


def test_capture_resolves_a_checkpoint_key_with_an_extra_prefix(
    tiny_model_dir,
) -> None:
    """A checkpoint key may carry a prefix the loaded model does not expose.

    Qwen3.5 saves layer weights as ``model.language_model.layers.N...`` while
    the loaded model has them at ``model.layers.N...``. Capture must translate
    rather than fail on the name the checkpoint actually uses.
    """
    capturer = TransformersActivationCapturer(str(tiny_model_dir), device="cpu")
    direct = capturer.capture("tok1 tok2", "model.layers.0.mlp.down_proj")
    prefixed = capturer.capture(
        "tok1 tok2", "model.language_model.layers.0.mlp.down_proj.weight"
    )
    assert np.allclose(direct, prefixed), "the prefix must not change the result"


def test_capture_respects_max_tokens(tiny_model_dir) -> None:
    capturer = TransformersActivationCapturer(str(tiny_model_dir), device="cpu")
    acts = capturer.capture(
        " ".join(f"tok{i}" for i in range(1, 40)),
        "model.layers.0.mlp.down_proj",
        max_tokens=8,
    )
    assert acts.shape[0] <= 8, "prompt length must be bounded"


def test_capture_is_deterministic(tiny_model_dir) -> None:
    capturer = TransformersActivationCapturer(str(tiny_model_dir), device="cpu")
    a = capturer.capture("tok1 tok2", "model.layers.0.mlp.down_proj")
    b = capturer.capture("tok1 tok2", "model.layers.0.mlp.down_proj")
    assert np.allclose(a, b), "the same prompt must give the same state"


def test_reload_swaps_the_model(tiny_model_dir) -> None:
    capturer = TransformersActivationCapturer(str(tiny_model_dir), device="cpu")
    capturer.capture("tok1", "model.layers.0.mlp.down_proj")
    capturer.reload(str(tiny_model_dir))
    acts = capturer.capture("tok1", "model.layers.0.mlp.down_proj")
    assert acts.shape[1] == _HIDDEN
