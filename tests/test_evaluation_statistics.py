"""
tests/test_evaluation_statistics.py
====================================
Unit tests for hqnn_forge.evaluation.statistics.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from hqnn_forge.evaluation import (
    WilcoxonResult,
    rank_biserial_correlation,
    wilcoxon_signed_rank,
)
from hqnn_forge.evaluation.statistics import EXACT_MAX_N, _average_ranks


def _brute_force_p(diffs: np.ndarray, alternative: str) -> float:
    """Enumerate all 2^n sign flips of the observed |d| ranks."""
    d = diffs[diffs != 0]
    ranks = _average_ranks(np.abs(d))
    observed = ranks[d > 0].sum()
    stats = [sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product([0, 1], repeat=len(d))]
    stats = np.array(stats)
    upper = np.mean(stats >= observed - 1e-9)
    lower = np.mean(stats <= observed + 1e-9)
    return {"greater": upper, "less": lower, "two-sided": min(1.0, 2 * min(upper, lower))}[alternative]


class TestRankBiserial:
    def test_identical_arrays_give_zero(self) -> None:
        a = [0.5, 0.6, 0.7]
        assert rank_biserial_correlation(a, a) == 0.0

    def test_strict_domination_gives_plus_minus_one(self) -> None:
        a = [0.60, 0.62, 0.65, 0.58, 0.61]
        b = [0.50, 0.52, 0.55, 0.48, 0.51]
        assert rank_biserial_correlation(a, b) == pytest.approx(1.0)
        assert rank_biserial_correlation(b, a) == pytest.approx(-1.0)

    def test_hand_computed_value(self) -> None:
        # d = [+1, -2, +3, +4]  ranks 1..4, W+ = 8, W- = 2, r = 6/10
        a = [1.0, 0.0, 3.0, 4.0]
        b = [0.0, 2.0, 0.0, 0.0]
        assert rank_biserial_correlation(a, b) == pytest.approx(0.6)

    def test_is_antisymmetric(self) -> None:
        rng = np.random.default_rng(0)
        a, b = rng.random(7), rng.random(7)
        assert rank_biserial_correlation(a, b) == pytest.approx(-rank_biserial_correlation(b, a))


class TestWilcoxon:
    def test_returns_named_result(self) -> None:
        res = wilcoxon_signed_rank([1, 2, 3], [0, 0, 0])
        assert isinstance(res, WilcoxonResult) and res.method == "exact" and res.n == 3

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
    def test_minimum_attainable_p_for_small_n(self, n: int) -> None:
        """All folds favour a: the p-value is the floor, 2 / 2^n two-sided."""
        a = np.arange(1, n + 1, dtype=float)
        b = np.zeros(n)
        res = wilcoxon_signed_rank(a, b)
        assert res.statistic == n * (n + 1) / 2
        assert res.p_value == pytest.approx(min(1.0, 2 / 2**n))
        assert res.min_p_value == pytest.approx(min(1.0, 2 / 2**n))
        one_sided = wilcoxon_signed_rank(a, b, alternative="greater")
        assert one_sided.p_value == pytest.approx(1 / 2**n)
        assert one_sided.min_p_value == pytest.approx(1 / 2**n)

    def test_five_folds_cannot_reach_005(self) -> None:
        """The thesis case: n=5 can never reject at 0.05 two-sided."""
        res = wilcoxon_signed_rank([0.58, 0.61, 0.55, 0.60, 0.57], [0.56, 0.57, 0.54, 0.55, 0.56])
        assert res.min_p_value == pytest.approx(0.0625)
        assert res.min_p_value > 0.05

    def test_zero_differences_are_dropped(self) -> None:
        res = wilcoxon_signed_rank([1.0, 2.0, 5.0, 5.0], [0.0, 0.0, 5.0, 5.0])
        assert res.n == 2

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_matches_brute_force_with_ties(self, alternative: str) -> None:
        a = np.array([0.3, 0.5, 0.2, 0.9, 0.4, 0.1, 0.7, 0.6])
        b = np.array([0.1, 0.7, 0.4, 0.7, 0.4, 0.3, 0.5, 0.2])  # |d| has ties at 0.2
        res = wilcoxon_signed_rank(a, b, alternative=alternative)
        assert res.p_value == pytest.approx(_brute_force_p(a - b, alternative), abs=1e-12)

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_matches_scipy_without_ties(self, alternative: str) -> None:
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(1)
        for n in (5, 9, 20):
            a, b = rng.random(n), rng.random(n)
            ours = wilcoxon_signed_rank(a, b, alternative=alternative)
            theirs = stats.wilcoxon(a, b, alternative=alternative, method="exact")
            assert ours.p_value == pytest.approx(theirs.pvalue, rel=1e-10)
            if alternative != "two-sided":
                assert ours.statistic == pytest.approx(theirs.statistic)

    def test_normal_approximation_above_exact_limit(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(2)
        n = EXACT_MAX_N + 10
        a, b = rng.random(n) + 0.05, rng.random(n)
        ours = wilcoxon_signed_rank(a, b)
        theirs = stats.wilcoxon(a, b, method="approx", correction=False)
        assert ours.method == "normal"
        assert ours.p_value == pytest.approx(theirs.pvalue, rel=1e-8)
        assert ours.min_p_value < 1e-10

    def test_p_value_is_symmetric_in_argument_order(self) -> None:
        a, b = [0.3, 0.5, 0.2, 0.8, 0.45], [0.1, 0.6, 0.4, 0.7, 0.4]
        assert wilcoxon_signed_rank(a, b).p_value == pytest.approx(wilcoxon_signed_rank(b, a).p_value)
        assert wilcoxon_signed_rank(a, b, alternative="greater").p_value == pytest.approx(
            wilcoxon_signed_rank(b, a, alternative="less").p_value
        )

    def test_all_zero_differences_raise(self) -> None:
        with pytest.raises(ValueError, match="every paired difference is zero"):
            wilcoxon_signed_rank([0.5, 0.6], [0.5, 0.6])

    @pytest.mark.parametrize(
        "a, b, match",
        [
            ([1, 2], [1], "same length"),
            ([], [], "no scores"),
            ([1, float("nan")], [0, 0], "finite"),
        ],
    )
    def test_input_validation(self, a: list, b: list, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            wilcoxon_signed_rank(a, b)
        with pytest.raises(ValueError, match=match):
            rank_biserial_correlation(a, b)

    def test_bad_alternative(self) -> None:
        with pytest.raises(ValueError, match="alternative must be"):
            wilcoxon_signed_rank([1], [0], alternative="both")  # type: ignore[arg-type]
