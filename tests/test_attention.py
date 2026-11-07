"""Tests for attention-map contrast.

The reductions here are pure array work, so they are testable without a GPU. What
matters is that each summary field means what its name says, and that the contrast
cannot report a difference that is not there.
"""

import numpy as np
import pytest

from wsp.attention import (
    AttentionCapture,
    AttentionMap,
    HeadContrast,
    _entropy_of_distribution,
    attention_contrast,
    head_ranking,
    summarize_attention,
    summarize_attention_chunked,
)


def _uniform(heads=2, n=4):
    """Attention where every query spreads evenly over every key."""
    return np.full((heads, n, n), 1.0 / n, dtype=np.float32)


def _focused(heads=2, n=4):
    """Attention where every query reads only its own key."""
    array = np.zeros((heads, n, n), dtype=np.float32)
    for head in range(heads):
        for i in range(n):
            array[head, i, i] = 1.0
    return array


class TestSummarizeAttention:
    def test_uniform_attention_has_maximum_entropy(self):
        summary = summarize_attention(_uniform(n=4))
        # Uniform over 4 keys is log(4) nats.
        assert np.allclose(summary.row_entropy, np.log(4.0), atol=1e-5)

    def test_focused_attention_has_zero_entropy(self):
        summary = summarize_attention(_focused(n=4))
        assert np.allclose(summary.row_entropy, 0.0, atol=1e-5)

    def test_diagonal_mass_separates_the_two_cases(self):
        uniform = summarize_attention(_uniform(n=4))
        focused = summarize_attention(_focused(n=4))
        assert np.allclose(uniform.diagonal_mass, 0.25, atol=1e-5)
        assert np.allclose(focused.diagonal_mass, 1.0, atol=1e-5)

    def test_column_mass_sums_to_one_per_head(self):
        summary = summarize_attention(_uniform(n=4))
        assert np.allclose(summary.column_mass.sum(axis=-1), 1.0, atol=1e-5)

    def test_column_mass_finds_which_key_is_read(self):
        # Every query reads key 0 only. Column mass should be 1.0 at key 0.
        array = np.zeros((1, 4, 4), dtype=np.float32)
        array[0, :, 0] = 1.0
        summary = summarize_attention(array)
        assert np.isclose(summary.column_mass[0, 0], 1.0, atol=1e-5)
        assert np.allclose(summary.column_mass[0, 1:], 0.0, atol=1e-5)

    def test_batch_axis_is_squeezed(self):
        array = _uniform(n=4)[None, ...]
        summary = summarize_attention(array)
        assert summary.heads == 2

    def test_multi_batch_is_rejected(self):
        array = np.stack([_uniform(n=4), _uniform(n=4)])
        with pytest.raises(ValueError, match="batch size 1"):
            summarize_attention(array)

    def test_non_square_attention_is_rejected(self):
        with pytest.raises(ValueError, match="square"):
            summarize_attention(np.zeros((2, 4, 5), dtype=np.float32))

    def test_wrong_rank_is_rejected(self):
        with pytest.raises(ValueError, match="heads, queries, keys"):
            summarize_attention(np.zeros((4, 4), dtype=np.float32))

    def test_unnormalised_rows_are_rescaled(self):
        # Rows summing to 4.0 rather than 1.0 must give the same entropy as normalised.
        array = _uniform(n=4) * 4.0
        summary = summarize_attention(array)
        assert np.allclose(summary.row_entropy, np.log(4.0), atol=1e-5)

    def test_zero_rows_are_rejected(self):
        # A pre-softmax hook would look like this. Refusing is the point: silently
        # normalising would produce plausible-looking entropies from a non-distribution.
        array = np.zeros((1, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="sum to zero"):
            summarize_attention(array)

    def test_metadata_is_recorded(self):
        summary = summarize_attention(_uniform(), layer=7, timestep=3, stream="audio")
        assert summary.layer == 7
        assert summary.timestep == 3
        assert summary.stream == "audio"


class TestSummarizeAttentionChunked:
    """The chunked path must equal the materialised path, not merely resemble it.

    The real model cannot materialise a 27,900 x 27,900 attention tensor, so the chunked
    computation is what will actually run. If it disagreed with the tested materialised
    path, every number reported from the GPU would be unverified.
    """

    def _qk(self, heads=2, n=6, dim=8, seed=0):
        gen = np.random.default_rng(seed)
        q = gen.normal(size=(heads, n, dim)).astype(np.float32)
        k = gen.normal(size=(heads, n, dim)).astype(np.float32)
        return q, k

    def _materialised(self, q, k):
        """The full softmax, computed the way summarize_attention expects."""
        scale = q.shape[-1] ** -0.5
        scores = np.einsum("hqd,hkd->hqk", q, k) * scale
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        return w

    def test_matches_the_materialised_path(self):
        q, k = self._qk()
        expected = summarize_attention(self._materialised(q, k))
        actual = summarize_attention_chunked(q, k)
        assert np.allclose(actual.row_entropy, expected.row_entropy, atol=1e-5)
        assert np.allclose(actual.column_mass, expected.column_mass, atol=1e-5)
        assert np.allclose(actual.diagonal_mass, expected.diagonal_mass, atol=1e-5)

    def test_chunk_size_does_not_change_the_result(self):
        # A chunk boundary that does not divide the query count is the case that would
        # break a naive implementation: the last chunk is short.
        q, k = self._qk(n=7)
        whole = summarize_attention_chunked(q, k, chunk=7)
        ragged = summarize_attention_chunked(q, k, chunk=3)
        assert np.allclose(whole.row_entropy, ragged.row_entropy, atol=1e-5)
        assert np.allclose(whole.column_mass, ragged.column_mass, atol=1e-5)
        assert np.allclose(whole.diagonal_mass, ragged.diagonal_mass, atol=1e-5)

    def test_uniform_qk_gives_maximum_entropy(self):
        # Identical q and k for every token: scores are all equal, so each query spreads
        # evenly and entropy is log(n).
        n = 5
        q = np.ones((2, n, 4), dtype=np.float32)
        k = np.ones((2, n, 4), dtype=np.float32)
        summary = summarize_attention_chunked(q, k)
        assert np.allclose(summary.row_entropy, np.log(n), atol=1e-5)

    def test_cross_attention_has_no_diagonal(self):
        # Queries and keys differ in length: a query has no "own key".
        gen = np.random.default_rng(1)
        q = gen.normal(size=(2, 4, 8)).astype(np.float32)
        k = gen.normal(size=(2, 9, 8)).astype(np.float32)
        summary = summarize_attention_chunked(q, k)
        assert summary.queries == 4
        assert summary.keys == 9
        assert np.allclose(summary.diagonal_mass, 0.0)

    def test_head_mismatch_is_rejected(self):
        q = np.ones((2, 4, 8), dtype=np.float32)
        k = np.ones((3, 4, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="head count mismatch"):
            summarize_attention_chunked(q, k)

    def test_head_dim_mismatch_is_rejected(self):
        q = np.ones((2, 4, 8), dtype=np.float32)
        k = np.ones((2, 4, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="head dim mismatch"):
            summarize_attention_chunked(q, k)

    def test_empty_attention_is_rejected(self):
        q = np.ones((2, 0, 8), dtype=np.float32)
        k = np.ones((2, 4, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="empty attention"):
            summarize_attention_chunked(q, k)

    def test_metadata_is_recorded(self):
        q, k = self._qk()
        summary = summarize_attention_chunked(q, k, layer=7, timestep=3, stream="audio")
        assert summary.layer == 7
        assert summary.timestep == 3
        assert summary.stream == "audio"


class TestSummarizeAttentionTorch:
    """The GPU path must equal the numpy path, or every GPU number is unverified.

    The numpy path is the reference and stays the tested one. The torch path exists
    because reducing 27,900 keys in numpy took 49 minutes of CPU time for one run.
    """

    def _qk(self, heads=2, n=6, dim=8, seed=0):
        import torch

        gen = np.random.default_rng(seed)
        q = gen.normal(size=(heads, n, dim)).astype(np.float32)
        k = gen.normal(size=(heads, n, dim)).astype(np.float32)
        return torch.from_numpy(q), torch.from_numpy(k), q, k

    def test_matches_the_numpy_path(self):
        from wsp.attention import summarize_attention_torch

        q_t, k_t, q, k = self._qk()
        expected = summarize_attention_chunked(q, k)
        actual = summarize_attention_torch(q_t, k_t)
        assert np.allclose(actual.row_entropy, expected.row_entropy, atol=1e-4)
        assert np.allclose(actual.column_mass, expected.column_mass, atol=1e-4)
        assert np.allclose(actual.diagonal_mass, expected.diagonal_mass, atol=1e-4)

    def test_chunk_size_does_not_change_the_result(self):
        from wsp.attention import summarize_attention_torch

        q_t, k_t, _, _ = self._qk(n=7)
        whole = summarize_attention_torch(q_t, k_t, chunk=7)
        ragged = summarize_attention_torch(q_t, k_t, chunk=3)
        assert np.allclose(whole.row_entropy, ragged.row_entropy, atol=1e-4)
        assert np.allclose(whole.column_mass, ragged.column_mass, atol=1e-4)

    def test_bfloat16_input_is_widened(self):
        # Ovi runs in bfloat16. The summary must widen before the softmax or the tail of
        # the distribution -- what entropy measures -- is lost.
        import torch

        from wsp.attention import summarize_attention_torch

        q_t, k_t, _, _ = self._qk()
        summary = summarize_attention_torch(q_t.bfloat16(), k_t.bfloat16())
        assert summary.row_entropy.dtype == np.float32
        assert np.all(np.isfinite(summary.row_entropy))

    def test_cross_attention_has_no_diagonal(self):
        import torch

        from wsp.attention import summarize_attention_torch

        gen = np.random.default_rng(1)
        q = torch.from_numpy(gen.normal(size=(2, 4, 8)).astype(np.float32))
        k = torch.from_numpy(gen.normal(size=(2, 9, 8)).astype(np.float32))
        summary = summarize_attention_torch(q, k)
        assert summary.queries == 4
        assert summary.keys == 9
        assert np.allclose(summary.diagonal_mass, 0.0)

    def test_numpy_input_is_rejected(self):
        from wsp.attention import summarize_attention_torch

        with pytest.raises(TypeError, match="needs torch tensors"):
            summarize_attention_torch(np.ones((2, 4, 8)), np.ones((2, 4, 8)))


class TestEntropyOfDistribution:
    def test_uniform_is_log_n(self):
        assert np.isclose(_entropy_of_distribution(np.ones(8)), np.log(8.0))

    def test_point_mass_is_zero(self):
        p = np.zeros(8)
        p[3] = 1.0
        assert np.isclose(_entropy_of_distribution(p), 0.0, atol=1e-9)

    def test_unnormalised_input_is_normalised(self):
        assert np.isclose(_entropy_of_distribution(np.ones(8) * 5.0), np.log(8.0))

    def test_zero_distribution_is_rejected(self):
        with pytest.raises(ValueError, match="sums to zero"):
            _entropy_of_distribution(np.zeros(4))


class TestAttentionContrast:
    def test_identical_maps_produce_no_contrast(self):
        a = summarize_attention(_uniform())
        b = summarize_attention(_uniform())
        assert attention_contrast(a, b, min_effect=0.0) == []

    def test_a_real_difference_is_found(self):
        a = summarize_attention(_focused())
        b = summarize_attention(_uniform())
        contrasts = attention_contrast(a, b, min_effect=0.0)
        assert len(contrasts) == 2
        # Focused minus uniform: entropy falls, diagonal mass rises.
        assert all(c.entropy_delta < 0 for c in contrasts)
        assert all(c.diagonal_delta > 0 for c in contrasts)

    def test_sign_convention_is_a_minus_b(self):
        focused = summarize_attention(_focused())
        uniform = summarize_attention(_uniform())
        # a=focused, b=uniform -> negative delta. Reversed -> positive.
        assert attention_contrast(focused, uniform, min_effect=0.0)[0].entropy_delta < 0
        assert attention_contrast(uniform, focused, min_effect=0.0)[0].entropy_delta > 0

    def test_head_count_mismatch_is_rejected(self):
        a = summarize_attention(_uniform(heads=2))
        b = summarize_attention(_uniform(heads=3))
        with pytest.raises(ValueError, match="head count mismatch"):
            attention_contrast(a, b)

    def test_shape_mismatch_is_rejected(self):
        a = summarize_attention(_uniform(n=4))
        b = summarize_attention(_uniform(n=8))
        with pytest.raises(ValueError, match="shape mismatch"):
            attention_contrast(a, b)

    def test_unchanged_heads_are_not_reported(self):
        # One head changes, the other does not. An unchanged head is not a contrast,
        # even at min_effect=0.0 -- reporting it as a zero-delta row would read like a
        # finding when it is the absence of one.
        a_array = _uniform(heads=2)
        a_array[0] = _focused(heads=2)[0]
        a = summarize_attention(a_array)
        b = summarize_attention(_uniform(heads=2))
        contrasts = attention_contrast(a, b, min_effect=0.0)
        assert [c.head for c in contrasts] == [0]

    def test_min_effect_filters_weak_heads(self):
        # A head that moved a little is dropped by a threshold above its effect size.
        a_array = _uniform(heads=2)
        a_array[0] = _focused(heads=2)[0]
        a = summarize_attention(a_array)
        b = summarize_attention(_uniform(heads=2))
        # Head 0's effect size is 2.0; a threshold of 3.0 must drop it.
        assert attention_contrast(a, b, min_effect=3.0) == []
        assert [c.head for c in attention_contrast(a, b, min_effect=1.0)] == [0]

    def test_effect_size_falls_back_when_the_head_has_no_spread(self):
        # A head whose queries all have identical entropy has zero pooled spread. That
        # must not divide by zero or report infinity; it scales by the head's own level.
        a = summarize_attention(_uniform(heads=1, n=4))
        b = summarize_attention(_uniform(heads=1, n=4))
        # Force a difference while keeping zero spread.
        b.row_entropy = b.row_entropy + 0.5
        contrasts = attention_contrast(a, b, min_effect=0.0)
        assert len(contrasts) == 1
        effect = contrasts[0].effect_size
        assert np.isfinite(effect)
        assert effect > 0.0
        # 0.5 / mean(level) where level is log(4) and log(4)+0.5.
        expected = 0.5 / ((np.log(4.0) + np.log(4.0) + 0.5) / 2.0)
        assert np.isclose(effect, expected, rtol=1e-4)


class TestHeadRanking:
    def _contrasts(self):
        return [
            HeadContrast(head=0, entropy_delta=0.1, diagonal_delta=0.0,
                         column_mass_l1=0.2, effect_size=0.5),
            HeadContrast(head=1, entropy_delta=-0.9, diagonal_delta=0.1,
                         column_mass_l1=0.8, effect_size=3.0),
            HeadContrast(head=2, entropy_delta=0.3, diagonal_delta=0.0,
                         column_mass_l1=0.4, effect_size=1.5),
        ]

    def test_ranks_by_effect_size_by_default(self):
        ranked = head_ranking(self._contrasts())
        assert [c.head for c in ranked] == [1, 2, 0]

    def test_ranks_by_raw_entropy_delta(self):
        ranked = head_ranking(self._contrasts(), by="entropy_delta")
        # Abs magnitude: 0.9, 0.3, 0.1 -> heads 1, 2, 0.
        assert [c.head for c in ranked] == [1, 2, 0]

    def test_ranks_by_column_mass(self):
        ranked = head_ranking(self._contrasts(), by="column_mass_l1")
        assert [c.head for c in ranked] == [1, 2, 0]

    def test_ties_break_by_head_index(self):
        # Stage 2's reproducibility failure came from a top-k cut inside a plateau.
        # A deterministic tie-break is what makes the ordering reproducible.
        tied = [
            HeadContrast(head=2, entropy_delta=0.5, diagonal_delta=0.0,
                         column_mass_l1=0.1, effect_size=2.0),
            HeadContrast(head=0, entropy_delta=0.5, diagonal_delta=0.0,
                         column_mass_l1=0.1, effect_size=2.0),
            HeadContrast(head=1, entropy_delta=0.5, diagonal_delta=0.0,
                         column_mass_l1=0.1, effect_size=2.0),
        ]
        assert [c.head for c in head_ranking(tied)] == [0, 1, 2]

    def test_unknown_criterion_is_rejected(self):
        with pytest.raises(ValueError, match="by must be"):
            head_ranking(self._contrasts(), by="nonsense")


class TestAttentionCapture:
    def test_record_and_retrieve(self):
        capture = AttentionCapture()
        assert capture.record(0, 0, _uniform()) is True
        assert capture.at(0, 0).heads == 2

    def test_declined_records_are_counted(self):
        capture = AttentionCapture()
        assert capture.record(0, 0, _uniform(), keep=False) is False
        assert capture.skipped == 1
        assert capture.maps == {}

    def test_streams_are_separate_keys(self):
        # Ovi's two stacks share block indices. A key without the stream would let
        # audio overwrite video, which is the bug dit.py already fixed.
        capture = AttentionCapture()
        capture.record(0, 0, _uniform(n=4), stream="video")
        capture.record(0, 0, _uniform(n=8), stream="audio")
        assert capture.at(0, 0, "video").keys == 4
        assert capture.at(0, 0, "audio").keys == 8
        assert capture.layers("video") == [0]
        assert capture.layers("audio") == [0]

    def test_missing_key_names_what_is_available(self):
        capture = AttentionCapture()
        capture.record(2, 5, _uniform())
        with pytest.raises(KeyError, match="available"):
            capture.at(0, 0)

    def test_summary_reports_both_streams(self):
        capture = AttentionCapture()
        capture.record(0, 0, _uniform(), stream="video")
        capture.record(0, 0, _uniform(), stream="audio")
        summary = capture.summary()
        assert summary["n_maps"] == 2
        assert summary["streams"] == ["audio", "video"]


class TestAttentionMapSummary:
    def test_summary_is_json_safe(self):
        import json

        summary = summarize_attention(_uniform()).summary()
        json.dumps(summary)  # raises if anything is a numpy scalar
        assert summary["heads"] == 2
        assert summary["queries"] == 4

    def test_column_mass_entropy_is_reported(self):
        # _focused has each query reading its own key, so column mass is uniform across
        # keys -- every key is read exactly once. That is maximum entropy, not zero.
        focused = summarize_attention(_focused(n=4))
        assert np.isclose(
            focused.summary()["column_mass_entropy"], np.log(4.0), atol=1e-5
        )

    def test_column_mass_entropy_is_zero_when_one_key_takes_all_mass(self):
        # Every query reads key 0 only: column mass is a point mass, so entropy is 0.
        array = np.zeros((1, 4, 4), dtype=np.float32)
        array[0, :, 0] = 1.0
        summary = summarize_attention(array)
        assert np.isclose(summary.summary()["column_mass_entropy"], 0.0, atol=1e-5)
