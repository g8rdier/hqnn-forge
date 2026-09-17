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
wilcoxon_signed_rank     Paired signed-rank test with its attainable p-value floor.
WilcoxonResult           Result of wilcoxon_signed_rank.
rank_biserial_correlation  Effect size for the paired comparison.
plots                    Submodule: confusion matrix, fold boxplot, efficiency
                         frontier (needs matplotlib; import it explicitly:
                         ``from hqnn_forge.evaluation import plots``).
"""

from hqnn_forge.evaluation.statistics import (
    WilcoxonResult,
    rank_biserial_correlation,
    wilcoxon_signed_rank,
)
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
    "WilcoxonResult",
    "balanced_accuracy",
    "f1_score",
    "find_optimal_threshold",
    "matthews_corrcoef",
    "parameter_efficiency",
    "rank_biserial_correlation",
    "wilcoxon_signed_rank",
]
