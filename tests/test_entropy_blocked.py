"""The blocked k-NN distance computation must match the naive one exactly.

The blocking is a memory optimisation, not an approximation. If it changed the
distances, every entropy number in the project would shift silently — the same class
of error as the dropped dimension factor, which produced plausible-looking numbers
that were 70 nats wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from wsp.entropy import _DIST_BLOCK, _kth_neighbour_distances, knn_entropy


def _naive_kth(arr: np.ndarray, k: int) -> np.ndarray:
    """The straightforward O(n^2 d) version, for comparison."""
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return np.partition(dist, k - 1, axis=1)[:, k - 1]


class TestBlockedMatchesNaive:
    @pytest.mark.parametrize("n,d,k", [(50, 4, 1), (50, 4, 5), (200, 16, 3), (600, 8, 5)])
    def test_distances_match(self, n, d, k):
        rng = np.random.default_rng(n * 100 + d)
        arr = rng.normal(size=(n, d))
        blocked = _kth_neighbour_distances(arr, k)
        naive = _naive_kth(arr, k)
        assert np.allclose(blocked, naive, rtol=1e-9, atol=1e-9)

    def test_matches_across_a_block_boundary(self):
        # n just over the block size exercises the multi-block path.
        n = _DIST_BLOCK + 37
        rng = np.random.default_rng(7)
        arr = rng.normal(size=(n, 8))
        assert np.allclose(
            _kth_neighbour_distances(arr, 5), _naive_kth(arr, 5), rtol=1e-9, atol=1e-9
        )

    def test_entropy_matches_naive_entropy(self):
        from math import lgamma, log, pi

        from wsp.entropy import _digamma

        rng = np.random.default_rng(11)
        arr = rng.normal(size=(300, 12))
        k, n, dim = 5, arr.shape[0], arr.shape[1]
        kth = np.maximum(_naive_kth(arr, k), np.finfo(np.float64).tiny)
        unit_ball = (dim / 2.0) * log(pi) - lgamma(dim / 2.0 + 1.0)
        expected = _digamma(n) - _digamma(k) + unit_ball + dim * float(np.mean(np.log(kth)))
        assert knn_entropy(arr, k=k) == pytest.approx(expected, rel=1e-9)

    def test_self_distance_is_excluded(self):
        # A point must not be its own nearest neighbour. With four distinct points
        # the 1-NN distance is the gap to the closest *other* point, never zero.
        arr = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
        kth = _kth_neighbour_distances(arr, 1)
        assert np.all(kth > 0.0)
        assert np.allclose(kth, [1.0, 1.0, 4.0, 4.0])

    def test_duplicate_points_give_zero_distance(self):
        # Two identical points genuinely are at distance 0 from each other. That is
        # correct, not a bug: the estimator clamps to a tiny positive value before
        # taking the log, so the entropy stays finite. An earlier version of this
        # test asserted all distances were positive, which is wrong for duplicates.
        arr = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        kth = _kth_neighbour_distances(arr, 1)
        assert kth[0] == 0.0 and kth[1] == 0.0
        assert np.isfinite(knn_entropy(arr))

    def test_identical_points_give_finite_entropy(self):
        arr = np.ones((100, 8))
        assert np.isfinite(knn_entropy(arr))

    def test_single_dimension(self):
        rng = np.random.default_rng(3)
        arr = rng.normal(size=(200, 1))
        assert np.allclose(
            _kth_neighbour_distances(arr, 3), _naive_kth(arr, 3), rtol=1e-9, atol=1e-9
        )
