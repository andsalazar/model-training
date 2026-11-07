"""Tests for turning a WSP direction into specific weight edits.

The targeting maths is pure array work, so it is testable without a model. What
matters here is that a direction maps onto the *right axis* of the right matrix, that
an unaligned direction edits nothing rather than something arbitrary, and that a
partial edit cannot report success.
"""

from __future__ import annotations

import numpy as np
import pytest

from wsp.targeting import (
    RESIDUAL_WRITERS,
    ApplyResult,
    TargetPlan,
    TargetSite,
    _matrix_at,
    apply_plan,
    build_plan,
    damping_vector,
    find_targets,
)

DIM = 16


def _aligned_matrix(direction: np.ndarray, n_cols: int = 8, seed: int = 0) -> np.ndarray:
    """A ``[DIM, n_cols]`` matrix whose first column is exactly ``direction``."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(DIM, n_cols)).astype(np.float32)
    matrix[:, 0] = direction
    return matrix


class TestResidualWriters:
    def test_only_the_two_additive_sites_are_listed(self):
        # Ovi adds self_attn.o and ffn[2] to the residual stream; everything else
        # writes into those paths. Editing anything else would not move the state
        # along a residual-stream direction.
        assert RESIDUAL_WRITERS == (("self_attn", "o"), ("ffn", "2"))


class TestMatrixAt:
    def test_rejects_a_non_2d_weight(self):
        with pytest.raises(ValueError, match="expected a 2-D weight"):
            _matrix_at(np.ones((4, 4, 4)), "self_attn.o")

    def test_accepts_a_square_matrix(self):
        matrix = np.eye(4, dtype=np.float32)
        assert _matrix_at(matrix, "ffn.2").shape == (4, 4)


class TestFindTargets:
    def test_finds_the_column_that_is_the_direction(self):
        direction = np.zeros(DIM, dtype=np.float32)
        direction[0] = 1.0
        weights = {
            "self_attn.o": _aligned_matrix(direction),
            "ffn.2": np.random.default_rng(1).normal(size=(DIM, 8)).astype(np.float32),
        }
        sites = find_targets(direction, weights, stream="video", layer=3)
        o_site = next(s for s in sites if s.attribute == "self_attn")
        assert 0 in o_site.columns
        # Column 0 is the direction itself, so its cosine is 1.0.
        assert o_site.scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_names_the_site_unambiguously(self):
        direction = np.zeros(DIM, dtype=np.float32)
        direction[0] = 1.0
        sites = find_targets(
            direction, {"self_attn.o": _aligned_matrix(direction)}, stream="audio", layer=5
        )
        assert sites[0].name == "audio.blocks[5].self_attn.o"

    def test_dimension_mismatch_is_rejected(self):
        # A direction captured at the video hidden width cannot be projected onto an
        # audio matrix. Broadcasting would silently produce a meaningless score.
        direction = np.zeros(DIM, dtype=np.float32)
        direction[0] = 1.0
        weights = {"self_attn.o": np.ones((DIM * 2, 4), dtype=np.float32)}
        with pytest.raises(ValueError, match="rows but the direction has"):
            find_targets(direction, weights, stream="video", layer=0)

    def test_an_orthogonal_direction_selects_nothing(self):
        # Every cosine non-positive: the direction is unrelated to every column, so
        # there is nothing to align with and no edit is proposed.
        direction = np.zeros(DIM, dtype=np.float32)
        direction[0] = 1.0
        matrix = np.zeros((DIM, 8), dtype=np.float32)
        matrix[1:, :] = np.random.default_rng(2).normal(size=(DIM - 1, 8))
        sites = find_targets(direction, {"self_attn.o": matrix}, stream="video", layer=0)
        assert sites == []

    def test_selection_is_by_threshold_not_rank(self):
        # A plateau of equally-aligned columns must all be admitted. Stage 2's verdict
        # turned on a top-12 cut from a distribution tied to within 0.0012.
        dim = 32
        direction = np.zeros(dim, dtype=np.float32)
        direction[0] = 1.0
        matrix = np.zeros((dim, 20), dtype=np.float32)
        for j in range(20):
            matrix[0, j] = 1.0
            matrix[1 + j, j] = 0.001  # near-orthogonal noise, all columns tie
        sites = find_targets(direction, {"self_attn.o": matrix}, stream="video", layer=0)
        assert len(sites[0].columns) == 20

    def test_respects_min_columns(self):
        direction = np.zeros(DIM, dtype=np.float32)
        direction[0] = 1.0
        sites = find_targets(
            direction,
            {"self_attn.o": _aligned_matrix(direction)},
            stream="video",
            layer=0,
            min_columns=3,
        )
        assert len(sites[0].columns) >= 3

    def test_missing_weight_key_is_skipped_not_fatal(self):
        direction = np.zeros(DIM, dtype=np.float32)
        direction[0] = 1.0
        sites = find_targets(
            direction, {"self_attn.o": _aligned_matrix(direction)}, stream="video", layer=0
        )
        assert [s.attribute for s in sites] == ["self_attn"]


class TestDampingVector:
    def test_damping_is_negative(self):
        site = TargetSite(
            stream="video", layer=0, attribute="self_attn", weight_path="o",
            columns=(0, 1), scores=(1.0, 0.9), column_norms=(1.0, 1.0),
        )
        assert np.all(damping_vector(site, 0.05) < 0)

    def test_damping_scales_with_strength(self):
        site = TargetSite(
            stream="video", layer=0, attribute="self_attn", weight_path="o",
            columns=(0,), scores=(1.0,), column_norms=(1.0,),
        )
        assert damping_vector(site, 0.1)[0] == pytest.approx(-0.1)
        assert damping_vector(site, 0.02)[0] == pytest.approx(-0.02)

    def test_strength_one_is_refused(self):
        # A scale of -1.0 would zero the column: that is deletion, not damping, and
        # it would not distinguish "targeted change" from "remove the feature".
        site = TargetSite(
            stream="video", layer=0, attribute="self_attn", weight_path="o",
            columns=(0,), scores=(1.0,), column_norms=(1.0,),
        )
        with pytest.raises(ValueError, match="strength must be in"):
            damping_vector(site, 1.0)

    def test_negative_strength_is_refused(self):
        site = TargetSite(
            stream="video", layer=0, attribute="self_attn", weight_path="o",
            columns=(0,), scores=(1.0,), column_norms=(1.0,),
        )
        with pytest.raises(ValueError, match="strength must be in"):
            damping_vector(site, -0.1)


class TestApplyPlan:
    def _plan(self, strength: float = 0.1) -> TargetPlan:
        site = TargetSite(
            stream="video", layer=0, attribute="self_attn", weight_path="o",
            columns=(0,), scores=(1.0,), column_norms=(1.0,),
        )
        return TargetPlan(sites=[site], strength=strength)

    def test_damps_the_column_along_the_input_axis(self):
        # Linear stores [out, in]. Column j means "every output's read of input
        # feature j", so damping must scale that column across all rows.
        matrix = np.ones((4, 4), dtype=np.float32)
        plan = self._plan(strength=0.1)
        sink = {}
        apply_plan(
            plan,
            lambda s, l, k: matrix,
            lambda s, l, k, m: sink.__setitem__((s, l, k), m),
        )
        updated = sink[("video", 0, "self_attn.o")]
        assert np.allclose(updated[:, 0], 0.9)
        assert np.allclose(updated[:, 1:], 1.0)

    def test_does_not_mutate_the_input_matrix(self):
        # The source may be a view onto a loaded checkpoint; editing it in place
        # would silently corrupt the "before" state a comparison needs.
        matrix = np.ones((4, 4), dtype=np.float32)
        apply_plan(
            self._plan(), lambda s, l, k: matrix, lambda s, l, k, m: None
        )
        assert np.allclose(matrix, 1.0)

    def test_relative_change_is_recorded(self):
        matrix = np.ones((4, 4), dtype=np.float32)
        result = apply_plan(
            self._plan(), lambda s, l, k: matrix, lambda s, l, k, m: None
        )
        assert result.changed["video.blocks[0].self_attn.o"]["relative_change"] > 0

    def test_a_missing_weight_is_reported_not_dropped(self):
        result = apply_plan(
            self._plan(), lambda s, l, k: None, lambda s, l, k, m: None
        )
        assert not result.complete
        assert result.skipped == ("video.blocks[0].self_attn.o",)
        assert result.requested == 1
        assert result.changed == {}

    def test_partial_apply_is_not_complete(self):
        site_ok = TargetSite(
            stream="video", layer=0, attribute="self_attn", weight_path="o",
            columns=(0,), scores=(1.0,), column_norms=(1.0,),
        )
        site_missing = TargetSite(
            stream="video", layer=1, attribute="ffn", weight_path="2",
            columns=(0,), scores=(1.0,), column_norms=(1.0,),
        )
        plan = TargetPlan(sites=[site_ok, site_missing])
        matrix = np.ones((4, 4), dtype=np.float32)

        def source(stream, layer, key):
            return matrix if layer == 0 else None

        result = apply_plan(plan, source, lambda s, l, k, m: None)
        assert len(result.changed) == 1
        assert not result.complete
        assert len(plan.notes) == 1

    def test_result_serialises_without_a_summary_key(self):
        # An earlier version put a "_summary" entry inside `changed`, so iterating
        # the changed sites yielded a dictionary that was not a site.
        matrix = np.ones((4, 4), dtype=np.float32)
        result = apply_plan(
            self._plan(), lambda s, l, k: matrix, lambda s, l, k, m: None
        )
        payload = result.as_dict()
        assert "sites" in payload
        assert "_summary" not in payload["sites"]
        assert payload["complete"] is True


class TestBuildPlan:
    def test_builds_over_every_captured_layer(self):
        direction = np.zeros(DIM, dtype=np.float32)
        direction[0] = 1.0
        plan = build_plan(
            {("video", 0): direction, ("audio", 0): direction},
            lambda stream, layer, key: _aligned_matrix(direction),
        )
        assert len(plan.sites) == 4  # 2 streams x 2 writers

    def test_names_the_reason_when_nothing_was_found(self):
        direction = np.zeros(DIM, dtype=np.float32)
        direction[0] = 1.0
        plan = build_plan({("video", 0): direction}, lambda s, l, k: None)
        assert plan.sites == []
        assert any("no residual-writer weights found" in n for n in plan.notes)

    def test_empty_plan_says_so_without_claiming_a_result(self):
        plan = build_plan({}, lambda s, l, k: None)
        assert any("not evidence" in n for n in plan.notes)

    def test_plan_serialises(self):
        direction = np.zeros(DIM, dtype=np.float32)
        direction[0] = 1.0
        plan = build_plan(
            {("video", 0): direction}, lambda s, l, k: _aligned_matrix(direction)
        )
        payload = plan.as_dict()
        assert payload["n_sites"] == 2
        assert payload["n_columns"] > 0
        assert payload["strength"] == 0.02
