"""Tests for the Qwen math reproduction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from wsp.tasks.qwen_math import (
    COLUMNS_PER_LAYER,
    apply_direction_subtraction,
    find_mlp_down_proj,
    footprint,
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
