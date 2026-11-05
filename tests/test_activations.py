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


def _shared_prefix_fixture(prefix_len: int = 40, tail_len: int = 4):
    """Two runs that share a long prefix and diverge only at the end.

    This is the explore12 failure in miniature: averaged over all tokens the two
    states look nearly identical, because the prefix dominates the mean.
    """
    prefix = np.tile(np.array([1.0, 0.0], dtype=np.float32), (prefix_len, 1))
    looping = np.vstack([prefix, np.tile([1.0, 0.0], (tail_len, 1))])
    control = np.vstack([prefix, np.tile([0.0, 1.0], (tail_len, 1))])
    return looping, control, prefix_len, prefix_len + tail_len


def test_token_window_isolates_the_divergent_region() -> None:
    looping, control, start, stop = _shared_prefix_fixture()

    # Over all tokens the states are dominated by the shared prefix.
    assert contrast_cosine(looping, control) > 0.9

    # Past the prefix they are orthogonal.
    window = f"{start}:{stop}"
    assert contrast_cosine(looping, control, window) == pytest.approx(0.0, abs=1e-6)


def test_shared_prefix_cancels_in_the_direction() -> None:
    """The prefix cancels in the subtraction, so the direction is unaffected.

    This is why explore12's conclusion was too pessimistic. A cosine near 1.0
    between the two *means* says the states look alike, not that the direction is
    noise: the shared prefix appears in both means and cancels when they are
    subtracted. The window changes how much of the divergent tail is averaged in,
    which changes the direction's weighting, but not its sign or its existence.
    """
    looping, control, start, stop = _shared_prefix_fixture()

    whole = direction_from_contrast(looping, control)
    windowed = direction_from_contrast(looping, control, f"{start}:{stop}")

    # Both point the same way: +x, -y. The prefix did not corrupt the direction.
    assert whole.tolist() == pytest.approx([0.70710665, -0.7071069], abs=1e-5)
    assert windowed.tolist() == pytest.approx([0.70710677, -0.70710677], abs=1e-5)
    assert float(np.dot(whole, windowed)) > 0.9999


def test_token_window_rejects_out_of_range() -> None:
    array = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="outside"):
        direction_from_contrast(array, array, "0:9")


def test_token_window_rejects_empty_range() -> None:
    array = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="outside"):
        direction_from_contrast(array, array, "2:2")


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
    assert "WARNING" not in text, "a real contrast must not be warned about"


def test_summary_cautions_about_a_high_contrast_cosine() -> None:
    """A normalised direction hides how similar its two states were.

    The norm is always 1.0, so an editor reading only the norm would think it had
    a strong signal when the two states were nearly identical. The caution makes
    that visible — but it must not claim the direction is noise, because a shared
    prefix cancels in the subtraction. explore12's editor read the old wording as
    fatal and abandoned a usable direction.
    """
    vec = np.array([0.1, -0.9], dtype=np.float32)
    s = summarize_direction("noise", 12, vec, n_tokens=4, contrast=0.9999, k=2)
    text = s.as_text()
    assert "CAUTION" in text
    assert "not proof the direction is noise" in text
    # It must point at the check that settles it.
    assert "direction_similarity" in text or "compare the two vectors" in text


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


def test_subtract_direction_accepts_an_explicit_column_list() -> None:
    """The ranked columns are scattered, so a range cannot express the edit.

    explore13's editor used ``900:1000`` because that was the only form the tool
    accepted, and only 3 of the 12 columns its own alignment ranking identified
    fell inside it. An explicit list edits exactly the ranked columns.
    """
    weight = np.ones((2, 6), dtype=np.float32)
    d = np.array([1.0, 1.0], dtype=np.float32)
    out = subtract_direction(weight, [1, 4], d, scale=0.5)
    assert out[0, 1] == pytest.approx(0.5)
    assert out[0, 4] == pytest.approx(0.5)
    for col in (0, 2, 3, 5):
        assert out[0, col] == pytest.approx(1.0), f"column {col} must be untouched"


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


def test_registry_similarity_measures_agreement() -> None:
    """Agreement between captures is what distinguishes signal from noise."""
    reg = ActivationRegistry()
    same = np.array([0.6, 0.8], dtype=np.float32)
    opposite = np.array([-0.6, -0.8], dtype=np.float32)
    for name, vec in (("a", same), ("b", same), ("c", opposite)):
        reg.put(name, vec, summarize_direction(name, 1, vec, n_tokens=1, contrast=0.5))

    assert reg.similarity("a", "b") == pytest.approx(1.0)
    assert reg.similarity("a", "c") == pytest.approx(-1.0)


def test_registry_similarity_rejects_dim_mismatch() -> None:
    reg = ActivationRegistry()
    for name, dim in (("a", 2), ("b", 3)):
        vec = np.ones(dim, dtype=np.float32)
        reg.put(name, vec, summarize_direction(name, 1, vec, n_tokens=1, contrast=0.5))
    with pytest.raises(ValueError, match="different dims"):
        reg.similarity("a", "b")
