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
friedman_test            Friedman / Iman–Davenport test of k models over N datasets.
friedman_from_ranks      The same from published average ranks.
FriedmanResult           Result of friedman_test.
average_ranks            Mean rank of each model over the datasets (1 = best).
nemenyi_critical_difference  Nemenyi CD for all-pairs comparison of average ranks.
compare_to_control       Holm-corrected z-tests of every model against a control.
ControlComparison        One row of compare_to_control.
holm_correction          Holm step-down adjusted p-values for any family.
plots                    Submodule: confusion matrix, fold boxplot, efficiency
                         frontier (needs matplotlib; import it explicitly:
                         ``from hqnn_forge.evaluation import plots``).
"""

from hqnn_forge.evaluation.statistics import (
    ControlComparison,
    FriedmanResult,
    WilcoxonResult,
    average_ranks,
    compare_to_control,
    friedman_from_ranks,
    friedman_test,
    holm_correction,
    nemenyi_critical_difference,
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
    "ControlComparison",
    "FriedmanResult",
    "ThresholdSearchResult",
    "WilcoxonResult",
    "average_ranks",
    "balanced_accuracy",
    "compare_to_control",
    "f1_score",
    "find_optimal_threshold",
    "friedman_from_ranks",
    "friedman_test",
    "holm_correction",
    "matthews_corrcoef",
    "nemenyi_critical_difference",
    "parameter_efficiency",
    "rank_biserial_correlation",
    "wilcoxon_signed_rank",
]
