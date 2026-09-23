"""
tests/test_evaluation_statistics.py
====================================
Unit tests for hqnn_forge.evaluation.statistics.
"""

from __future__ import annotations

import itertools
import math

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
    stats = [
        sum(r for r, s in zip(ranks, signs) if s)
        for signs in itertools.product([0, 1], repeat=len(d))
    ]
    stats = np.array(stats)
    upper = np.mean(stats >= observed - 1e-9)
    lower = np.mean(stats <= observed + 1e-9)
    return {"greater": upper, "less": lower, "two-sided": min(1.0, 2 * min(upper, lower))}[
        alternative
    ]


class TestAverageRanks:
    """Mid-ranks checked against references that do not use _average_ranks."""

    def test_hand_computed_mid_ranks(self) -> None:
        values = np.array([0.5, 0.25, 0.5, 0.5, 0.125, 0.25])
        # sorted: 0.125 | 0.25 0.25 | 0.5 0.5 0.5
        # ranks:      1 | 2.5  2.5  |   5   5   5
        assert _average_ranks(values).tolist() == [5.0, 2.5, 5.0, 5.0, 1.0, 2.5]

    def test_no_ties_are_plain_ordinal_ranks(self) -> None:
        values = np.array([0.4, 0.1, 0.3, 0.2])
        assert _average_ranks(values).tolist() == [4.0, 1.0, 3.0, 2.0]

    def test_matches_scipy_rankdata_on_heavily_tied_data(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(3)
        values = rng.integers(1, 5, size=20).astype(float)
        np.testing.assert_allclose(_average_ranks(values), stats.rankdata(values))


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
        greater = wilcoxon_signed_rank(a, b, alternative="greater")
        assert greater.p_value == pytest.approx(1 / 2**n)
        assert greater.min_p_value == pytest.approx(1 / 2**n)
        # "less" is the mirror image: this data is as far from it as possible,
        # but its floor is still 1 / 2^n, reached when every fold favours b.
        less = wilcoxon_signed_rank(a, b, alternative="less")
        assert less.p_value == pytest.approx(1.0)
        assert less.min_p_value == pytest.approx(1 / 2**n)
        assert wilcoxon_signed_rank(b, a, alternative="less").p_value == pytest.approx(1 / 2**n)

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
        # Eighths, so the differences are exact in binary floating point and
        # the intended tie group really forms: six |d| = 0.25, one 0.5, one zero.
        a = np.array([0.375, 0.625, 0.250, 1.000, 0.500, 0.125, 0.875, 0.750])
        b = np.array([0.125, 0.875, 0.500, 0.750, 0.500, 0.375, 0.625, 0.250])
        d = a - b
        nonzero = np.abs(d[d != 0])
        assert np.count_nonzero(nonzero == 0.25) == 6 and np.unique(nonzero).size == 2
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

    def test_normal_approximation_corrects_for_ties(self) -> None:
        """Above the exact limit the variance must be sum(r^2)/4, not n(n+1)(2n+1)/24."""
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(4)
        n = EXACT_MAX_N + 11
        diffs = rng.choice(np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]), size=n)
        a, b = np.zeros(n), -diffs
        ours = wilcoxon_signed_rank(a, b)
        theirs = stats.wilcoxon(a, b, method="approx", correction=False)
        assert ours.method == "normal" and ours.n == n
        assert ours.p_value == pytest.approx(theirs.pvalue, rel=1e-8)

        # The tie correction is not cosmetic here: dropping it moves the p-value.
        ranks = _average_ranks(np.abs(diffs))
        w_plus = float(ranks[diffs > 0].sum())
        uncorrected_z = (w_plus - n * (n + 1) / 4) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        uncorrected_p = min(1.0, 2 * 0.5 * math.erfc(abs(uncorrected_z) / math.sqrt(2)))
        assert uncorrected_p != pytest.approx(theirs.pvalue, rel=1e-3)

        # min_p_value floors in this branch too, for both one-sided directions.
        assert wilcoxon_signed_rank(a, b, alternative="less").min_p_value < 1e-10
        assert wilcoxon_signed_rank(a, b, alternative="greater").min_p_value < 1e-10

    def test_p_value_is_symmetric_in_argument_order(self) -> None:
        a, b = [0.3, 0.5, 0.2, 0.8, 0.45], [0.1, 0.6, 0.4, 0.7, 0.4]
        assert wilcoxon_signed_rank(a, b).p_value == pytest.approx(
            wilcoxon_signed_rank(b, a).p_value
        )
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
