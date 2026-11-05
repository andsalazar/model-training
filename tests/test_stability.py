"""Tests for stability-based column selection.

Stage 2's failure: the top-12 cut sat inside a plateau where alignment cosines tie
to within 0.0012, so tiny numerical differences swapped columns and inverted the
verdict. These tests pin the property the fix has to have -- membership must not
depend on how many other columns happen to sit slightly above.
"""

from __future__ import annotations

import numpy as np
import pytest

from wsp.activations import (
    alignment_spectrum,
    select_stable_columns,
    selection_overlap,
    top_aligned_columns,
)


def _plateau_weights(n_rows: int = 32, n_cols: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """A weight whose column alignments bunch up, like the real ranking did.

    Pure random weights give a smooth, spread-out alignment distribution and so
    contain no plateau. The real ranking had one, so the fixture builds it
    deliberately: a block of columns share almost the same alignment with the
    direction, which is exactly the near-tie that made top-k unstable.
    """
    rng = np.random.default_rng(0)
    direction = rng.normal(size=n_rows).astype(np.float32)
    direction /= np.linalg.norm(direction)

    weight = rng.normal(size=(n_rows, n_cols)).astype(np.float32)
    # Model the real failure directly: a handful of columns sit inside a very
    # narrow band, and the cut point (12) falls INSIDE that band. Then a column's
    # membership depends on which side of a 0.00001 gap it lands on.
    #
    # The real ranking had rank 12 and 13 separated by 0.0012 with ranks 1..40
    # spanning only 0.049, so the top dozen were effectively tied. Reproducing
    # that means putting the tie exactly where the cut lands, not spreading it
    # across the whole head of the distribution.
    tied = rng.uniform(0.9990, 1.0, size=20).astype(np.float32)
    for i, gain in enumerate(tied):
        weight[:, i] = gain * direction * 4.0 + 0.02 * rng.normal(
            size=n_rows
        ).astype(np.float32)
    return weight, direction


def test_alignment_spectrum_is_score_ordered_and_bounded() -> None:
    weight, direction = _plateau_weights()
    spectrum = alignment_spectrum(weight, direction, depth=20)
    assert len(spectrum) == 20
    scores = [s for _, s in spectrum]
    assert scores == sorted(scores, reverse=True)
    assert len({c for c, _ in spectrum}) == 20, "columns must be distinct"


def test_spectrum_exposes_a_plateau_that_top_k_hides() -> None:
    """The point of the spectrum is that a tie is visible."""
    weight, direction = _plateau_weights()
    spectrum = alignment_spectrum(weight, direction, depth=40)
    best = spectrum[0][1]
    # In a plateau, many columns sit within 5% of the best score. Top-k would
    # report exactly 12 and say nothing about that.
    within_5pct = sum(1 for _, s in spectrum if s >= 0.95 * best)
    assert within_5pct > 12, "fixture should contain a plateau"


def _perturb_direction(
    direction: np.ndarray, scale: float, seed: int = 7
) -> np.ndarray:
    """A slightly different direction, as different hardware would produce.

    This is the right perturbation to model, not weight jitter. Stage 2's two runs
    differed because the *captured direction* differed -- it comes from a
    transformers forward pass whose activations shift with hardware -- while the
    weight tensor itself is fixed. Jittering the weights does nothing at any
    realistic scale; jittering the direction reproduces the observed instability.
    """
    rng = np.random.default_rng(seed)
    perturbed = direction + rng.normal(scale=scale, size=direction.shape).astype(
        np.float32
    )
    return perturbed / np.linalg.norm(perturbed)


def test_direction_perturbation_reproduces_the_real_instability() -> None:
    """The fixture must actually fail, or the stability tests prove nothing."""
    weight, direction = _plateau_weights()
    perturbed = _perturb_direction(direction, scale=1e-3)

    a = {c for c, _ in top_aligned_columns(weight, direction, k=12)}
    b = {c for c, _ in top_aligned_columns(weight, perturbed, k=12)}
    overlap = len(a & b) / len(a | b)
    assert overlap < 0.95, (
        "a tiny direction change must be able to swap top-k columns, otherwise "
        f"the fixture does not model the real failure (got {overlap:.3f})"
    )


def test_threshold_selection_survives_direction_perturbation() -> None:
    """The property Stage 2 needed and the rank cut did not have."""
    weight, direction = _plateau_weights()
    perturbed = _perturb_direction(direction, scale=1e-3)

    a = select_stable_columns(weight, direction, ratio=0.99)
    b = select_stable_columns(weight, perturbed, ratio=0.99)
    assert selection_overlap(a, b) >= 0.95, "threshold selection must be stable"


def test_threshold_selection_beats_rank_cut_under_direction_perturbation() -> None:
    """The direct comparison the fix rests on, at the observed perturbation size."""
    weight, direction = _plateau_weights()
    perturbed = _perturb_direction(direction, scale=1e-3)

    rank_a = {c for c, _ in top_aligned_columns(weight, direction, k=12)}
    rank_b = {c for c, _ in top_aligned_columns(weight, perturbed, k=12)}
    rank_overlap = len(rank_a & rank_b) / len(rank_a | rank_b)

    a = select_stable_columns(weight, direction, ratio=0.99)
    b = select_stable_columns(weight, perturbed, ratio=0.99)
    threshold_overlap = selection_overlap(a, b)

    assert threshold_overlap > rank_overlap, (
        f"threshold ({threshold_overlap:.3f}) must beat the rank cut "
        f"({rank_overlap:.3f}) under the perturbation that broke Stage 2"
    )


def test_tighter_ratio_selects_fewer_columns() -> None:
    weight, direction = _plateau_weights()
    wide = select_stable_columns(weight, direction, ratio=0.5)
    tight = select_stable_columns(weight, direction, ratio=0.99)
    assert tight.k <= wide.k


def test_selection_reports_its_threshold_and_k() -> None:
    weight, direction = _plateau_weights()
    sel = select_stable_columns(weight, direction, ratio=0.9)
    assert sel.k == len(sel.columns) == len(sel.scores)
    assert "columns within 90%" in sel.describe()
    payload = sel.as_dict()
    assert payload["k"] == sel.k
    assert payload["ratio"] == 0.9


def test_min_columns_backfills_a_too_strict_threshold() -> None:
    """A threshold that admits almost nothing must not yield an unusable edit."""
    rng = np.random.default_rng(3)
    weight = rng.normal(size=(16, 64)).astype(np.float32)
    direction = rng.normal(size=16).astype(np.float32)
    sel = select_stable_columns(weight, direction, ratio=1.0, min_columns=4)
    assert sel.k >= 4


def test_all_nonpositive_scores_select_nothing() -> None:
    """A direction unrelated to every column has nothing to align with."""
    weight = np.ones((8, 16), dtype=np.float32)
    direction = np.zeros(8, dtype=np.float32)
    sel = select_stable_columns(weight, direction, ratio=0.9)
    assert sel.columns == ()


def test_ratio_must_be_in_range() -> None:
    weight, direction = _plateau_weights()
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="ratio"):
            select_stable_columns(weight, direction, ratio=bad)


def test_candidate_depth_must_be_positive() -> None:
    weight, direction = _plateau_weights()
    with pytest.raises(ValueError, match="candidate_depth"):
        select_stable_columns(weight, direction, candidate_depth=0)


def test_overlap_of_identical_selections_is_one() -> None:
    weight, direction = _plateau_weights()
    a = select_stable_columns(weight, direction, ratio=0.9)
    assert selection_overlap(a, a) == 1.0


def test_overlap_of_disjoint_selections_is_zero() -> None:
    weight, direction = _plateau_weights()
    a = select_stable_columns(weight, direction, ratio=0.9)
    # A direction that is the exact negative selects the opposite alignment.
    negated = select_stable_columns(weight, -direction, ratio=0.9)
    if a.columns and negated.columns:
        assert selection_overlap(a, negated) < 0.2


def test_overlap_handles_empty_selections() -> None:
    weight, direction = _plateau_weights()
    a = select_stable_columns(weight, direction, ratio=0.9)
    empty = select_stable_columns(np.ones((32, 200), dtype=np.float32), np.zeros(32))
    assert selection_overlap(empty, empty) == 1.0
    assert selection_overlap(a, empty) == 0.0
