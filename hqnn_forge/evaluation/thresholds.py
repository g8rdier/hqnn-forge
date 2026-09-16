"""
hqnn_forge.evaluation.thresholds
================================
Data-driven decision thresholds and score-per-parameter for binary classifiers.

``predict`` takes a ``threshold`` and leaves its choice to the caller.  On
imbalanced data the default of 0.5 is rarely the best operating point, and
the usual remedy -- sweep thresholds on a validation split and keep the one
that maximises a rank-insensitive metric such as MCC -- is what
:func:`find_optimal_threshold` does.

All metrics here take hard labels and are computed from the confusion matrix
with plain tensor arithmetic, so nothing depends on scikit-learn at runtime.
Inputs may be ``torch.Tensor`` or anything ``torch.as_tensor`` accepts.

Conventions
-----------
* A sample is predicted positive when ``probability >= threshold``, matching
  ``BinaryClassifierBase.predict``.
* Metrics that are undefined for a labelling (MCC with a single predicted or
  true class, F1 with no positives anywhere) return ``0.0`` rather than NaN,
  the same convention scikit-learn uses, so a threshold search never picks a
  NaN over a number.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple

import torch
import torch.nn as nn

Metric = Callable[[torch.Tensor, torch.Tensor], float]


# ---------------------------------------------------------------------------
# Confusion-matrix metrics
# ---------------------------------------------------------------------------

def _as_binary(y: object, name: str) -> torch.Tensor:
    t = torch.as_tensor(y).reshape(-1)
    if t.dtype == torch.bool:
        t = t.long()
    if t.numel() and not torch.all((t == 0) | (t == 1)):
        raise ValueError(f"{name} must contain only 0/1 labels; got values {torch.unique(t).tolist()}.")
    return t.long()


def _confusion(y_true: object, y_pred: object) -> tuple[float, float, float, float]:
    """Return (tp, tn, fp, fn) as floats."""
    t = _as_binary(y_true, "y_true")
    p = _as_binary(y_pred, "y_pred")
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_pred differ in length: {t.numel()} vs {p.numel()}.")
    tp = float(((t == 1) & (p == 1)).sum())
    tn = float(((t == 0) & (p == 0)).sum())
    fp = float(((t == 0) & (p == 1)).sum())
    fn = float(((t == 1) & (p == 0)).sum())
    return tp, tn, fp, fn


def matthews_corrcoef(y_true: object, y_pred: object) -> float:
    """
    Matthews correlation coefficient in [-1, 1].

    ``(tp·tn − fp·fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))``.  Returns 0.0 when
    any factor in the denominator is zero (only one class present in
    ``y_true`` or in ``y_pred``), as scikit-learn does.
    """
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0.0:
        return 0.0
    return (tp * tn - fp * fn) / denom ** 0.5


def f1_score(y_true: object, y_pred: object) -> float:
    """F1 of the positive class; 0.0 when there are no positives in either vector."""
    tp, _, fp, fn = _confusion(y_true, y_pred)
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0.0 else 2 * tp / denom


def balanced_accuracy(y_true: object, y_pred: object) -> float:
    """Mean of the recall on each class; 0.5 is chance, 0.0 if a class is absent from y_true."""
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    if tp + fn == 0.0 or tn + fp == 0.0:
        return 0.0
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


#: Metrics selectable by name in :func:`find_optimal_threshold`.
METRICS: Mapping[str, Metric] = {
    "mcc": matthews_corrcoef,
    "f1": f1_score,
    "balanced_accuracy": balanced_accuracy,
}


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------

class ThresholdSearchResult(NamedTuple):
    """Best threshold found and the metric value it achieves."""

    threshold: float
    score: float


def find_optimal_threshold(
    y_true: object,
    y_prob: object,
    metric: str | Metric = "mcc",
) -> ThresholdSearchResult:
    """
    Threshold on ``y_prob`` that maximises ``metric`` against ``y_true``.

    Every distinct labelling the probabilities can produce is scored: the
    candidate thresholds are the sorted unique probabilities (a sample is
    positive when ``prob >= threshold``, so each unique value is one boundary)
    plus one candidate above the largest probability for the all-negative
    labelling.  That is exhaustive, so no grid resolution has to be chosen,
    and it costs one metric evaluation per unique probability.

    Ties are broken towards the candidate closest to 0.5, so a flat optimum
    does not resolve to an extreme threshold that only holds by accident of
    the validation sample.

    Parameters
    ----------
    y_true:
        Binary labels, shape ``(n,)``.
    y_prob:
        Positive-class probabilities, shape ``(n,)``, e.g. from
        ``model.predict_proba(X_val)``.
    metric:
        ``"mcc"`` (default), ``"f1"``, ``"balanced_accuracy"``, or any callable
        ``(y_true, y_pred) -> float`` taking hard labels.  Threshold-free
        metrics such as PR-AUC do not depend on the threshold and cannot be
        searched.

    Returns
    -------
    ThresholdSearchResult
        ``(threshold, score)``.  With a single unique probability the only
        labelling is returned as-is.

    Raises
    ------
    ValueError
        If ``metric`` is an unknown name, if the inputs are empty or differ in
        length, or if ``y_true`` is not binary.

    Examples
    --------
    >>> import torch
    >>> y = torch.tensor([0, 0, 0, 1, 1])
    >>> p = torch.tensor([0.1, 0.2, 0.4, 0.45, 0.9])
    >>> result = find_optimal_threshold(y, p)
    >>> round(result.threshold, 2), result.score
    (0.45, 1.0)
    """
    if isinstance(metric, str):
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRICS)} or pass a callable.")
        scorer = METRICS[metric]
    else:
        scorer = metric

    t = _as_binary(y_true, "y_true")
    p = torch.as_tensor(y_prob, dtype=torch.float64).reshape(-1)
    if t.numel() == 0:
        raise ValueError("y_true is empty; a threshold cannot be chosen from no samples.")
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_prob differ in length: {t.numel()} vs {p.numel()}.")
    if torch.any(p < 0) or torch.any(p > 1):
        raise ValueError(f"y_prob must lie in [0, 1]; got min {p.min():.4g}, max {p.max():.4g}.")

    unique = torch.unique(p)  # sorted ascending
    # The next float above the largest probability labels every sample
    # negative.  Unreachable when some probability is exactly 1.0, since
    # predict uses >= and no threshold in [0, 1] excludes it.
    above_max = torch.nextafter(unique[-1:], torch.ones_like(unique[-1:]) * 2)
    if unique[-1] < 1.0:
        unique = torch.cat([unique, above_max])
    candidates = unique

    best_threshold, best_score = float(candidates[0]), -float("inf")
    for threshold in candidates.tolist():
        score = float(scorer(t, (p >= threshold).long()))
        if score > best_score or (
            score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold, best_score = threshold, score
    return ThresholdSearchResult(best_threshold, best_score)


# ---------------------------------------------------------------------------
# Parameter efficiency
# ---------------------------------------------------------------------------

def parameter_efficiency(model: nn.Module | int, score: float) -> float:
    """
    ``score`` per thousand trainable parameters, e.g. MCC/kParam.

    Parameters
    ----------
    model:
        A module exposing ``count_parameters()`` (the hybrid classifiers), any
        ``nn.Module`` (trainable parameters are counted directly), or an
        integer parameter count.
    score:
        The metric value achieved by that model.

    Raises
    ------
    ValueError
        If the parameter count is not positive.
    """
    if isinstance(model, int):
        n_params = model
    elif hasattr(model, "count_parameters"):
        n_params = int(model.count_parameters())
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params <= 0:
        raise ValueError(f"parameter count must be positive; got {n_params}.")
    return float(score) / (n_params / 1000.0)
