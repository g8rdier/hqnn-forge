"""
hqnn_forge.evaluation.statistics
================================
Paired, non-parametric comparison of two models across CV folds.

With a handful of folds, parametric tests are not justified and the Wilcoxon
signed-rank test is the usual choice.  Its weakness at that size is that the
p-value is *discrete* with a floor: with n = 5 non-zero differences the
smallest two-sided p-value any data can produce is 2 / 2^5 = 0.0625, so
"not significant at 0.05" is guaranteed before a single fold is run.  The
result therefore carries ``min_p_value`` next to ``p_value``, and the
rank-biserial correlation is provided as the effect size to report instead.

Everything is computed in pure Python/NumPy: the exact null distribution of
the signed-rank statistic is built by dynamic programming over the ranks, so
tied absolute differences (average ranks) are handled exactly rather than by
switching to the normal approximation.

References
----------
* Wilcoxon (1945) "Individual comparisons by ranking methods", Biometrics
  Bulletin 1(6), 80–83.
* Kerby (2014) "The simple difference formula: an approach to teaching
  nonparametric correlation", Comprehensive Psychology 3, 11.IT.3.1.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

Alternative = Literal["two-sided", "greater", "less"]

#: Largest number of non-zero differences for which the exact distribution is
#: built; above it the normal approximation (with tie correction) is used.
EXACT_MAX_N: int = 50


class WilcoxonResult(NamedTuple):
    """
    Result of :func:`wilcoxon_signed_rank`.

    Attributes
    ----------
    statistic:
        ``W+``, the sum of the ranks of the positive differences ``a - b``.
    p_value:
        p-value for ``alternative``.
    n:
        Number of non-zero differences the test used.
    min_p_value:
        Smallest p-value attainable for this ``n``, rank pattern and
        ``alternative``.  If it exceeds your significance level, the test
        cannot reject whatever the data.
    method:
        ``"exact"`` or ``"normal"``.
    """

    statistic: float
    p_value: float
    n: int
    min_p_value: float
    method: str


def _differences(scores_a: npt.ArrayLike, scores_b: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    b = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"paired scores must have the same length; got {a.size} and {b.size}.")
    if a.size == 0:
        raise ValueError("no scores given.")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("scores must be finite.")
    return a - b


def _average_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """1-based ranks with ties sharing their mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _signed_ranks(
    diffs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    nonzero = diffs[diffs != 0.0]
    return _average_ranks(np.abs(nonzero)), nonzero > 0


def _exact_null_counts(ranks: npt.NDArray[np.float64]) -> dict[int, int]:
    """
    Number of sign assignments giving each value of 2·W+.

    Average ranks are multiples of 0.5, so doubling makes every rank an
    integer and the distribution can be counted exactly.
    """
    counts: dict[int, int] = {0: 1}
    for r in (int(round(2 * x)) for x in ranks):
        nxt: dict[int, int] = dict(counts)
        for total, c in counts.items():
            nxt[total + r] = nxt.get(total + r, 0) + c
        counts = nxt
    return counts


def _exact_p(counts: dict[int, int], w2: int, alternative: Alternative) -> float:
    total = sum(counts.values())
    upper = sum(c for v, c in counts.items() if v >= w2) / total
    lower = sum(c for v, c in counts.items() if v <= w2) / total
    if alternative == "greater":
        return upper
    if alternative == "less":
        return lower
    return min(1.0, 2.0 * min(upper, lower))


def _normal_p(ranks: npt.NDArray[np.float64], w_plus: float, alternative: Alternative) -> float:
    n = ranks.size
    mean = n * (n + 1) / 4.0
    # Tie correction: variance of W+ under H0 is sum(r_i^2) / 4
    sd = math.sqrt(float(np.sum(ranks**2)) / 4.0)
    if sd == 0.0:
        return 1.0
    z = (w_plus - mean) / sd
    upper = 0.5 * math.erfc(z / math.sqrt(2))
    lower = 0.5 * math.erfc(-z / math.sqrt(2))
    if alternative == "greater":
        return upper
    if alternative == "less":
        return lower
    return min(1.0, 2.0 * min(upper, lower))


def wilcoxon_signed_rank(
    scores_a: npt.ArrayLike,
    scores_b: npt.ArrayLike,
    alternative: Alternative = "two-sided",
) -> WilcoxonResult:
    """
    Wilcoxon signed-rank test on paired per-fold scores.

    Zero differences are dropped before ranking (Wilcoxon's original
    treatment).  Tied absolute differences get average ranks.

    Parameters
    ----------
    scores_a, scores_b:
        Paired metric values, one per fold, same length.
    alternative:
        ``"two-sided"`` (default); ``"greater"`` tests whether ``a`` tends to
        exceed ``b``; ``"less"`` the reverse.

    Returns
    -------
    WilcoxonResult

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty or non-finite, or if every
        difference is zero (the test is undefined; the rank-biserial
        correlation is 0 in that case).

    Examples
    --------
    >>> res = wilcoxon_signed_rank([0.58, 0.61, 0.55, 0.60, 0.57],
    ...                            [0.56, 0.57, 0.54, 0.55, 0.56])
    >>> res.statistic, res.p_value, res.min_p_value
    (15.0, 0.0625, 0.0625)
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"alternative must be 'two-sided', 'greater' or 'less'; got {alternative!r}."
        )
    ranks, positive = _signed_ranks(_differences(scores_a, scores_b))
    n = int(ranks.size)
    if n == 0:
        raise ValueError("every paired difference is zero, so the signed-rank test is undefined.")
    w_plus = float(np.sum(ranks[positive]))

    if n <= EXACT_MAX_N:
        counts = _exact_null_counts(ranks)
        p = _exact_p(counts, int(round(2 * w_plus)), alternative)
        extremes = (min(counts), max(counts))
        if alternative == "greater":
            min_p = _exact_p(counts, extremes[1], alternative)
        elif alternative == "less":
            min_p = _exact_p(counts, extremes[0], alternative)
        else:
            min_p = _exact_p(counts, extremes[1], alternative)
        return WilcoxonResult(w_plus, p, n, min_p, "exact")

    p = _normal_p(ranks, w_plus, alternative)
    total = float(np.sum(ranks))
    extreme = total if alternative != "less" else 0.0
    min_p = _normal_p(ranks, extreme, alternative)
    return WilcoxonResult(w_plus, p, n, min_p, "normal")


def rank_biserial_correlation(scores_a: npt.ArrayLike, scores_b: npt.ArrayLike) -> float:
    """
    Matched-pairs rank-biserial correlation, the effect size for the
    signed-rank test.

    ``r = (W+ − W−) / (W+ + W−)`` over the non-zero differences (Kerby 2014).
    It lies in [-1, 1]: ``+1`` when ``a`` beats ``b`` on every fold, ``-1``
    when it loses on every fold, ``0`` when wins and losses balance by rank.
    If every difference is zero the models are indistinguishable and ``0.0``
    is returned.

    Parameters
    ----------
    scores_a, scores_b:
        Paired metric values, one per fold, same length.
    """
    ranks, positive = _signed_ranks(_differences(scores_a, scores_b))
    total = float(np.sum(ranks))
    if total == 0.0:
        return 0.0
    w_plus = float(np.sum(ranks[positive]))
    return (2.0 * w_plus - total) / total
