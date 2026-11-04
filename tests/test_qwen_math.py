"""Tests for the Qwen math reproduction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from wsp.tasks.qwen_math import (
    COLUMNS_PER_LAYER,
    FULL_ATTENTION_LAYERS,
    apply_direction_subtraction,
    down_proj_shape,
    find_mlp_down_proj,
    footprint,
    is_full_attention,
)
from wsp.weights.slices import TensorSlice


def test_find_mlp_down_proj_llama_style() -> None:
    keys = [
        "model.layers.15.self_attn.q_proj.weight",
        "model.layers.15.mlp.down_proj.weight",
        "model.layers.19.mlp.down_proj.weight",
    ]
    assert find_mlp_down_proj(keys, 15) == "model.layers.15.mlp.down_proj.weight"
    assert find_mlp_down_proj(keys, 19) == "model.layers.19.mlp.down_proj.weight"


def test_find_mlp_down_proj_gpt_style() -> None:
    keys = ["transformer.h.15.mlp.c_proj.weight"]
    assert find_mlp_down_proj(keys, 15) == "transformer.h.15.mlp.c_proj.weight"


def test_find_mlp_down_proj_missing_layer() -> None:
    with pytest.raises(KeyError, match="layer 7"):
        find_mlp_down_proj(["model.layers.15.mlp.down_proj.weight"], 7)


def test_direction_subtraction_changes_only_target_columns() -> None:
    tensor = np.ones((8, 128), dtype=np.float32)
    target = TensorSlice(name="w", cols=(0, COLUMNS_PER_LAYER))
    direction = np.full(COLUMNS_PER_LAYER, 0.5, dtype=np.float32)

    edited, changed = apply_direction_subtraction(tensor, target, direction)

    assert changed == 8 * COLUMNS_PER_LAYER
    np.testing.assert_allclose(edited[:, :COLUMNS_PER_LAYER], 0.5)
    np.testing.assert_allclose(edited[:, COLUMNS_PER_LAYER:], 1.0)


def test_direction_subtraction_rejects_wrong_direction_shape() -> None:
    tensor = np.ones((8, 128), dtype=np.float32)
    target = TensorSlice(name="w", cols=(0, COLUMNS_PER_LAYER))
    with pytest.raises(ValueError, match="direction must have shape"):
        apply_direction_subtraction(tensor, target, np.ones(16, dtype=np.float32))


def test_footprint_matches_paper_scale() -> None:
    # Two layers x 1024 rows x 64 columns is the paper's ~131k weights.
    changed = 2 * 1024 * 64
    assert "0.0164%" in footprint(changed)


def test_find_mlp_down_proj_qwen35_nested_naming() -> None:
    # Qwen3.5 nests the decoder under model.language_model.
    keys = [
        "model.language_model.layers.15.mlp.down_proj.weight",
        "model.language_model.layers.19.mlp.down_proj.weight",
    ]
    assert (
        find_mlp_down_proj(keys, 15)
        == "model.language_model.layers.15.mlp.down_proj.weight"
    )


def test_down_proj_shape_matches_real_config() -> None:
    # hidden_size=1024, intermediate_size=3584 in Qwen3.5-0.8B.
    assert down_proj_shape() == (1024, 3584)


def test_paper_edit_arithmetic_reproduces_exactly() -> None:
    # 64 columns x 1024 rows x 2 layers == the paper's 131,072 weights.
    rows, _ = down_proj_shape()
    assert COLUMNS_PER_LAYER * rows * len((15, 19)) == 131_072


def test_target_layers_are_full_attention() -> None:
    # Both layers the paper edited use full attention, not linear attention.
    assert is_full_attention(15)
    assert is_full_attention(19)


def test_full_attention_layers_are_every_fourth() -> None:
    assert FULL_ATTENTION_LAYERS == (3, 7, 11, 15, 19, 23)
    assert not is_full_attention(14)
    assert not is_full_attention(16)
