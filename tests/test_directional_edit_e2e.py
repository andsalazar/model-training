"""End-to-end test of the activation-driven edit path.

This is the check that matters for Stage 1: capture a real direction from a real
model, rank real weight columns by it, apply the edit to a real checkpoint on
disk, and confirm the written file actually changed. Everything here runs on CPU
against a tiny Qwen model, so it exercises the same code path a pod would.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from wsp.runner import TransformersActivationCapturer  # noqa: E402
from wsp.tools import ToolError, WeightTools  # noqa: E402
from wsp.weights.store import WeightStore  # noqa: E402

_HIDDEN = 32
_INTERMEDIATE = 64
_TENSOR = "model.layers.0.mlp.down_proj.weight"


class _Runner:
    """A runner that records reloads without needing vLLM."""

    def __init__(self) -> None:
        self.reloaded: list[str] = []

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return "generated"

    def measure(self, prompt: str, max_tokens: int = 2048) -> dict:
        return {"looped": True, "tokens": 100, "hit_cap": False}

    def reload(self, model_path: str) -> None:
        self.reloaded.append(model_path)


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory):
    """A tiny Qwen checkpoint saved as a single safetensors file."""
    path = tmp_path_factory.mktemp("e2e-qwen")
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

    # A word-level tokenizer over the model's vocab, so prompts tokenize to real
    # ids. Without one the prompt maps to zero tokens and the forward pass fails.
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

    # save_pretrained writes a sharded index; the store wants one file.
    shards = sorted(path.glob("*.safetensors"))
    assert shards, "save_pretrained produced no safetensors"
    return path, shards[0]


def _build(checkpoint_dir, tmp_path):
    path, shard = checkpoint_dir
    store = WeightStore(shard)
    runner = _Runner()
    capturer = TransformersActivationCapturer(str(path), device="cpu")
    tools = WeightTools(store, runner, capturer=capturer, max_forward_passes=20)
    return tools, store, runner


def test_capture_direction_from_a_real_model(checkpoint_dir, tmp_path) -> None:
    tools, _, _ = _build(checkpoint_dir, tmp_path)

    out = tools.capture_direction(
        "loopdir", _TENSOR, "tok1 tok2 tok3", "tok4 tok5 tok6"
    )
    assert "direction 'loopdir'" in out
    assert "norm 1.0000" in out, "the direction must be normalised"
    assert "contrast_cosine" in out
    assert "loopdir" in tools.list_directions()


def test_aligned_columns_ranks_real_weight_columns(checkpoint_dir, tmp_path) -> None:
    tools, _, _ = _build(checkpoint_dir, tmp_path)
    tools.capture_direction("loopdir", _TENSOR, "tok1 tok2 tok3", "tok4 tok5 tok6")

    out = tools.aligned_columns(_TENSOR, "loopdir", k=5)
    lines = out.splitlines()
    assert "alignment with 'loopdir'" in lines[0]
    # Header plus five ranked columns.
    assert len(lines) == 1 + 1 + 5
    scores = [float(line.split("\t")[1]) for line in lines[2:]]
    assert scores == sorted(scores, reverse=True), "must be ranked strongest first"


def test_directional_edit_writes_and_reloads(checkpoint_dir, tmp_path) -> None:
    tools, store, runner = _build(checkpoint_dir, tmp_path)
    tools.capture_direction("loopdir", _TENSOR, "tok1 tok2 tok3", "tok4 tok5 tok6")

    original = store.get(_TENSOR).copy()

    out = tools.apply_directional_edit(
        _TENSOR, "loopdir", cols="0:4", scale=0.5, out_path=str(tmp_path / "edited")
    )
    assert "subtracted" in out
    assert runner.reloaded, "the runner must be pointed at the edited checkpoint"

    edited = store.get(_TENSOR)
    assert edited.shape == original.shape

    # The selected columns changed; the rest did not.
    assert not np.allclose(edited[:, 0:4], original[:, 0:4])
    assert np.allclose(edited[:, 4:], original[:, 4:]), "columns outside the range"


def test_directional_edit_is_not_a_scalar_shift(checkpoint_dir, tmp_path) -> None:
    """The whole point: the edit changes direction, not just magnitude.

    A per-column scalar subtraction shifts every row of a column by the same
    amount. A directional edit subtracts a different amount per row, which is
    what changes the direction the neuron writes along.
    """
    tools, store, _ = _build(checkpoint_dir, tmp_path)
    tools.capture_direction("loopdir", _TENSOR, "tok1 tok2 tok3", "tok4 tok5 tok6")

    original = store.get(_TENSOR).copy()
    tools.apply_directional_edit(
        _TENSOR, "loopdir", cols="0:2", scale=0.5, out_path=str(tmp_path / "edited")
    )
    edited = store.get(_TENSOR)

    delta = original[:, 0] - edited[:, 0]
    # A scalar shift would make every element of delta identical.
    assert not np.allclose(delta, delta[0]), (
        "the edit must vary by row; a constant delta means it was a scalar shift"
    )


def test_directional_edit_rejects_unknown_direction(checkpoint_dir, tmp_path) -> None:
    tools, _, _ = _build(checkpoint_dir, tmp_path)
    with pytest.raises(ToolError, match="no captured direction"):
        tools.apply_directional_edit(
            _TENSOR, "missing", cols="0:2", scale=1.0, out_path=str(tmp_path / "x")
        )


def test_capture_tools_consume_forward_budget(checkpoint_dir, tmp_path) -> None:
    tools, _, _ = _build(checkpoint_dir, tmp_path)
    before = tools.forward_passes_left
    tools.capture_direction("d", _TENSOR, "tok1 tok2", "tok3 tok4")
    # Two captures: looping and control.
    assert tools.forward_passes_left == before - 2
