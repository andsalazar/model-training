"""Tests for WeightStore name resolution.

An editor reads module paths from the loaded model and addresses tensors with
them, but checkpoint keys carry the full save-time prefix. Qwen3.5 saves layer
weights as ``model.language_model.layers.N...`` while the model exposes them at
``model.layers.N...``. Resolving in the store means every tool accepts either
form instead of each one re-implementing the mapping.
"""

from __future__ import annotations

import numpy as np
import pytest
from safetensors.numpy import save_file

from wsp.weights.store import WeightStore


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "model.language_model.layers.7.mlp.down_proj.weight": np.ones(
                (4, 8), dtype=np.float32
            ),
            "model.language_model.layers.23.mlp.down_proj.weight": np.ones(
                (4, 8), dtype=np.float32
            ),
        },
        str(path),
    )
    with WeightStore(path) as s:
        yield s


def test_exact_key_wins(store: WeightStore) -> None:
    key = "model.language_model.layers.7.mlp.down_proj.weight"
    assert store.resolve(key) == key


def test_short_name_resolves_to_the_full_key(store: WeightStore) -> None:
    assert (
        store.resolve("model.layers.7.mlp.down_proj.weight")
        == "model.language_model.layers.7.mlp.down_proj.weight"
    )


def test_get_accepts_the_short_name(store: WeightStore) -> None:
    array = store.get("model.layers.23.mlp.down_proj.weight")
    assert array.shape == (4, 8)


def test_info_reports_the_resolved_key(store: WeightStore) -> None:
    info = store.info("model.layers.7.mlp.down_proj.weight")
    assert info.name == "model.language_model.layers.7.mlp.down_proj.weight"


def test_ambiguous_suffix_raises(store: WeightStore) -> None:
    # "down_proj.weight" matches both layers, so it cannot be resolved.
    with pytest.raises(KeyError, match="ambiguous"):
        store.resolve("down_proj.weight")


def test_unknown_name_raises(store: WeightStore) -> None:
    with pytest.raises(KeyError, match="not in checkpoint"):
        store.resolve("model.layers.99.mlp.down_proj.weight")
