"""
hqnn_forge.evaluation
=====================
Post-hoc evaluation helpers for binary classifiers on imbalanced data.

Exported symbols
----------------
find_optimal_threshold   Decision threshold that maximises a metric on held-out probabilities.
ThresholdSearchResult    (threshold, score) pair returned by find_optimal_threshold.
matthews_corrcoef        MCC from labels, pure torch/NumPy.
f1_score                 F1 for the positive class.
balanced_accuracy        Mean of recall over the two classes.
parameter_efficiency     Score per thousand trainable parameters.
"""

from hqnn_forge.evaluation.thresholds import (
    METRICS,
    ThresholdSearchResult,
    balanced_accuracy,
    f1_score,
    find_optimal_threshold,
    matthews_corrcoef,
    parameter_efficiency,
)

__all__: list[str] = [
    "METRICS",
    "ThresholdSearchResult",
    "balanced_accuracy",
    "f1_score",
    "find_optimal_threshold",
    "matthews_corrcoef",
    "parameter_efficiency",
]
