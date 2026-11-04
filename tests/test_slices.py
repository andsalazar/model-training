"""Tests for weight slicing and text round-tripping."""

from __future__ import annotations

import numpy as np
import pytest

from wsp.weights.slices import (
    TensorSlice,
    apply_edit,
    count_changed,
    from_text,
    to_text,
)


@pytest.fixture
def matrix() -> np.ndarray:
    return np.arange(24, dtype=np.float32).reshape(4, 6)


def test_slice_full_extent(matrix: np.ndarray) -> None:
    target = TensorSlice(name="w")
    assert target.apply(matrix).shape == (4, 6)


def test_slice_columns(matrix: np.ndarray) -> None:
    target = TensorSlice(name="w", cols=(0, 2))
    region = target.apply(matrix)
    assert region.shape == (4, 2)
    np.testing.assert_array_equal(region[:, 0], matrix[:, 0])


def test_slice_rows_and_cols(matrix: np.ndarray) -> None:
    target = TensorSlice(name="w", rows=(1, 3), cols=(2, 5))
    assert target.apply(matrix).shape == (2, 3)


def test_slice_rejects_non_2d() -> None:
    target = TensorSlice(name="w")
    with pytest.raises(ValueError, match="2-D"):
        target.apply(np.zeros(3, dtype=np.float32))


def test_text_round_trip(matrix: np.ndarray) -> None:
    restored = from_text(to_text(matrix), shape=matrix.shape)
    np.testing.assert_allclose(restored, matrix, rtol=1e-4)


def test_from_text_skips_prose() -> None:
    text = "Here is the slice you asked for:\n1 2 3\n4 5 6\nThat is all."
    array = from_text(text)
    np.testing.assert_array_equal(array, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))


def test_from_text_rejects_ragged() -> None:
    with pytest.raises(ValueError, match="ragged"):
        from_text("1 2 3\n4 5")


def test_from_text_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        from_text("1 2\n3 4", shape=(3, 2))


def test_apply_edit_adds_delta(matrix: np.ndarray) -> None:
    target = TensorSlice(name="w", cols=(0, 2))
    delta = np.ones((4, 2), dtype=np.float32)
    edited = apply_edit(matrix, target, delta)

    np.testing.assert_array_equal(edited[:, :2], matrix[:, :2] + 1)
    np.testing.assert_array_equal(edited[:, 2:], matrix[:, 2:])
    # Source is untouched.
    np.testing.assert_array_equal(matrix[:, :2], np.arange(24, dtype=np.float32).reshape(4, 6)[:, :2])


def test_apply_edit_rejects_shape_mismatch(matrix: np.ndarray) -> None:
    target = TensorSlice(name="w", cols=(0, 2))
    with pytest.raises(ValueError, match="does not match"):
        apply_edit(matrix, target, np.ones((4, 3), dtype=np.float32))


def test_count_changed(matrix: np.ndarray) -> None:
    edited = matrix.copy()
    edited[0, 0] += 1.0
    assert count_changed(matrix, edited) == 1
    assert count_changed(matrix, matrix) == 0
