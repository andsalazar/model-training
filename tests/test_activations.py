"""Tests for activation capture math and directional edits.

Everything here is array math, so it runs without a GPU. The model-side hooking
is exercised separately in test_runner.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from wsp.activations import (
    ActivationRegistry,
    column_alignment,
    contrast_cosine,
    direction_from_contrast,
    mean_state,
    subtract_direction,
    summarize_direction,
    top_aligned_columns,
)


def test_mean_state_reduces_over_tokens() -> None:
    acts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert mean_state(acts).tolist() == [2.0, 3.0]


def test_mean_state_accepts_a_single_vector() -> None:
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert mean_state(vec).tolist() == [1.0, 2.0, 3.0]


def test_direction_from_contrast_is_unit_norm() -> None:
    looping = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    control = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    d = direction_from_contrast(looping, control)
    assert np.linalg.norm(d) == pytest.approx(1.0)
    assert d.tolist() == [1.0, 0.0]


def test_direction_points_from_control_toward_looping() -> None:
    looping = np.array([[0.0, 3.0]], dtype=np.float32)
    control = np.array([[0.0, 1.0]], dtype=np.float32)
    d = direction_from_contrast(looping, control)
    assert d[1] > 0, "direction must point toward the looping state"


def test_direction_from_contrast_rejects_identical_states() -> None:
    same = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="identical"):
        direction_from_contrast(same, same)


def test_direction_from_contrast_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        direction_from_contrast(np.ones((2, 4)), np.ones((2, 5)))


def test_contrast_cosine_is_one_for_identical_states() -> None:
    same = np.ones((2, 4), dtype=np.float32)
    assert contrast_cosine(same, same) == pytest.approx(1.0)


def test_contrast_cosine_is_low_for_orthogonal_states() -> None:
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    assert contrast_cosine(a, b) == pytest.approx(0.0)


def test_contrast_cosine_handles_zero_state() -> None:
    zero = np.zeros((1, 4), dtype=np.float32)
    assert contrast_cosine(zero, np.ones((1, 4))) == 0.0


def test_column_alignment_scores_each_column() -> None:
    # Column 0 points along +x, column 1 along +y.
    weight = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = column_alignment(weight, np.array([1.0, 0.0], dtype=np.float32))
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_column_alignment_is_scale_invariant() -> None:
    weight = np.array([[10.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = column_alignment(weight, np.array([1.0, 0.0], dtype=np.float32))
    assert scores[0] == pytest.approx(1.0), "a large column must not score higher"


def test_column_alignment_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="dims"):
        column_alignment(np.ones((4, 3)), np.ones(5))


def test_column_alignment_handles_zero_columns() -> None:
    weight = np.zeros((4, 3), dtype=np.float32)
    scores = column_alignment(weight, np.ones(4, dtype=np.float32))
    assert np.all(np.isfinite(scores)), "zero columns must not divide by zero"


def test_top_aligned_columns_orders_by_score() -> None:
    weight = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.float32)
    top = top_aligned_columns(weight, np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert top[0][0] == 0
    assert top[0][1] > top[1][1]


def test_top_aligned_columns_respects_k() -> None:
    weight = np.ones((4, 10), dtype=np.float32)
    assert len(top_aligned_columns(weight, np.ones(4), k=3)) == 3


def test_summarize_direction_reports_top_components() -> None:
    vec = np.array([0.1, -0.9, 0.2, 0.3], dtype=np.float32)
    s = summarize_direction("d", 12, vec, n_tokens=8, contrast=0.42, k=2)
    assert s.name == "d"
    assert s.layer == 12
    assert s.dim == 4
    assert s.n_tokens == 8
    assert s.contrast_cosine == pytest.approx(0.42)
    assert s.top_dims[0][0] == 1, "largest magnitude component is index 1"
    assert s.top_dims[0][1] < 0


def test_summary_renders_as_text() -> None:
    vec = np.array([0.1, -0.9], dtype=np.float32)
    s = summarize_direction("loopdir", 12, vec, n_tokens=4, contrast=0.9, k=2)
    text = s.as_text()
    assert "loopdir" in text
    assert "layer 12" in text
    assert "dim\tvalue" in text


def test_subtract_direction_removes_the_component() -> None:
    weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    d = np.array([1.0, 0.0], dtype=np.float32)
    out = subtract_direction(weight, (0, 1), d, scale=1.0)
    # Row 0 carries the whole direction, so it loses 1.0; row 1's component is
    # zero and must be left alone.
    assert out[0, 0] == pytest.approx(0.0)
    assert out[1, 0] == pytest.approx(3.0)
    # Untouched column.
    assert out[0, 1] == pytest.approx(2.0)


def test_subtract_direction_only_touches_selected_columns() -> None:
    weight = np.ones((2, 4), dtype=np.float32)
    d = np.array([1.0, 1.0], dtype=np.float32)
    out = subtract_direction(weight, (1, 3), d, scale=0.5)
    assert out[0, 0] == pytest.approx(1.0), "column 0 is outside the range"
    assert out[0, 1] == pytest.approx(0.5)
    assert out[0, 2] == pytest.approx(0.5)
    assert out[0, 3] == pytest.approx(1.0), "column 3 is outside the half-open range"


def test_subtract_direction_does_not_mutate_the_input() -> None:
    weight = np.ones((2, 2), dtype=np.float32)
    subtract_direction(weight, None, np.ones(2), scale=1.0)
    assert np.all(weight == 1.0), "the source array must be left untouched"


def test_subtract_direction_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="dims"):
        subtract_direction(np.ones((4, 3)), None, np.ones(5), scale=1.0)


def test_registry_round_trips_a_direction() -> None:
    reg = ActivationRegistry()
    vec = np.array([0.6, 0.8], dtype=np.float32)
    s = summarize_direction("d", 3, vec, n_tokens=2, contrast=0.5)
    reg.put("d", vec, s)
    assert reg.names == ["d"]
    assert reg.get("d").tolist() == vec.tolist()
    assert reg.summary("d").layer == 3


def test_registry_raises_for_unknown_name() -> None:
    reg = ActivationRegistry()
    with pytest.raises(KeyError, match="no captured direction"):
        reg.get("missing")


def test_registry_lists_known_names_in_the_error() -> None:
    reg = ActivationRegistry()
    vec = np.ones(2, dtype=np.float32)
    reg.put("alpha", vec, summarize_direction("alpha", 1, vec, n_tokens=1, contrast=0.1))
    with pytest.raises(KeyError, match="alpha"):
        reg.get("beta")
