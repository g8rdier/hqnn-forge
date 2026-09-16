"""
tests/test_evaluation_thresholds.py
====================================
Unit tests for hqnn_forge.evaluation: metrics, threshold search, efficiency.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hqnn_forge.evaluation import (
    METRICS,
    ThresholdSearchResult,
    balanced_accuracy,
    f1_score,
    find_optimal_threshold,
    matthews_corrcoef,
    parameter_efficiency,
)

Y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
P = torch.tensor([0.05, 0.10, 0.20, 0.30, 0.35, 0.60, 0.70, 0.80, 0.90, 0.95])


class TestMetrics:
    def test_perfect_and_inverted_mcc(self) -> None:
        assert matthews_corrcoef(Y, Y) == pytest.approx(1.0)
        assert matthews_corrcoef(Y, 1 - Y) == pytest.approx(-1.0)

    def test_hand_computed_confusion(self) -> None:
        y_true = [1, 1, 1, 0, 0, 0, 0, 0]
        y_pred = [1, 1, 0, 1, 0, 0, 0, 0]  # tp=2 fn=1 fp=1 tn=4
        tp, tn, fp, fn = 2, 4, 1, 1
        expected = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        assert matthews_corrcoef(y_true, y_pred) == pytest.approx(expected)
        assert f1_score(y_true, y_pred) == pytest.approx(2 * tp / (2 * tp + fp + fn))
        assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5 * (2 / 3 + 4 / 5))

    def test_single_class_conventions(self) -> None:
        # Undefined cases return 0.0, never NaN, so a search can compare them
        assert matthews_corrcoef([0, 0, 0], [0, 0, 1]) == 0.0
        assert matthews_corrcoef([0, 1, 1], [1, 1, 1]) == 0.0
        assert f1_score([0, 0], [0, 0]) == 0.0
        assert balanced_accuracy([1, 1], [1, 0]) == 0.0

    def test_accepts_numpy_bool_and_float_inputs(self) -> None:
        y = np.array([True, False, True])
        assert matthews_corrcoef(y, np.array([1.0, 0.0, 1.0])) == pytest.approx(1.0)

    def test_rejects_non_binary_and_mismatched(self) -> None:
        with pytest.raises(ValueError, match="only 0/1 labels"):
            matthews_corrcoef([0, 2], [0, 1])
        with pytest.raises(ValueError, match="differ in length"):
            matthews_corrcoef([0, 1], [0, 1, 1])

    def test_matches_scikit_learn(self) -> None:
        sk = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(0)
        for _ in range(20):
            y_true = rng.integers(0, 2, 50)
            y_pred = rng.integers(0, 2, 50)
            assert matthews_corrcoef(y_true, y_pred) == pytest.approx(sk.matthews_corrcoef(y_true, y_pred), abs=1e-12)
            assert f1_score(y_true, y_pred) == pytest.approx(sk.f1_score(y_true, y_pred, zero_division=0), abs=1e-12)
            assert balanced_accuracy(y_true, y_pred) == pytest.approx(sk.balanced_accuracy_score(y_true, y_pred), abs=1e-12)


class TestFindOptimalThreshold:
    def test_returns_named_tuple(self) -> None:
        result = find_optimal_threshold(Y, P)
        assert isinstance(result, ThresholdSearchResult)
        threshold, score = result
        assert result.threshold == threshold and result.score == score

    def test_separable_data_finds_a_perfect_threshold(self) -> None:
        y = torch.tensor([0, 0, 0, 1, 1])
        p = torch.tensor([0.1, 0.2, 0.4, 0.45, 0.9])
        result = find_optimal_threshold(y, p)
        assert result.score == pytest.approx(1.0)
        assert 0.4 < result.threshold <= 0.45

    @pytest.mark.parametrize("metric", sorted(METRICS))
    def test_is_exhaustive_over_unique_probabilities(self, metric: str) -> None:
        """No threshold on a fine grid beats the returned one."""
        result = find_optimal_threshold(Y, P, metric=metric)
        scorer = METRICS[metric]
        assert result.score == pytest.approx(scorer(Y, (P >= result.threshold).long()))
        for t in torch.linspace(0, 1, 1001).tolist():
            assert scorer(Y, (P >= t).long()) <= result.score + 1e-12

    def test_default_metric_is_mcc(self) -> None:
        assert find_optimal_threshold(Y, P) == find_optimal_threshold(Y, P, metric="mcc")

    def test_accepts_callable_metric(self) -> None:
        accuracy = lambda t, p: float((torch.as_tensor(t) == torch.as_tensor(p)).float().mean())  # noqa: E731
        result = find_optimal_threshold(Y, P, metric=accuracy)
        assert result.score == pytest.approx(0.8)

    def test_ties_resolve_towards_half(self) -> None:
        # Every threshold in (0.2, 0.8] gives the same perfect labelling
        y = torch.tensor([0, 1])
        p = torch.tensor([0.2, 0.8])
        assert find_optimal_threshold(y, p).threshold == pytest.approx(0.8)

    def test_single_class_labels_do_not_raise(self) -> None:
        # MCC is 0 for every threshold; the search still returns a result
        result = find_optimal_threshold(torch.zeros(4, dtype=torch.long), P[:4])
        assert result.score == 0.0

    def test_degenerate_constant_probabilities(self) -> None:
        result = find_optimal_threshold(Y, torch.full((10,), 0.3))
        assert result.threshold in (pytest.approx(0.3), pytest.approx(0.3 + 1e-12))
        assert result.score == 0.0

    def test_all_negative_labelling_is_a_candidate(self) -> None:
        # Predicting nothing positive is the best accuracy when positives are rare noise
        y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        p = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        accuracy = lambda t, pr: float((torch.as_tensor(t) == torch.as_tensor(pr)).float().mean())  # noqa: E731
        result = find_optimal_threshold(y, p, metric=accuracy)
        assert result.threshold > p.max().item() and result.score == pytest.approx(0.9)

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            find_optimal_threshold(Y, P, metric="auc")
        with pytest.raises(ValueError, match="empty"):
            find_optimal_threshold(torch.empty(0), torch.empty(0))
        with pytest.raises(ValueError, match="differ in length"):
            find_optimal_threshold(Y, P[:5])
        with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
            find_optimal_threshold(Y, P * 2)

    def test_numpy_inputs(self) -> None:
        result = find_optimal_threshold(Y.numpy(), P.numpy())
        assert result == find_optimal_threshold(Y, P)


class TestParameterEfficiency:
    def test_from_integer_count(self) -> None:
        assert parameter_efficiency(500, 0.4) == pytest.approx(0.8)

    def test_from_plain_module_counts_trainable_only(self) -> None:
        module = torch.nn.Linear(10, 1)  # 11 parameters
        assert parameter_efficiency(module, 0.11) == pytest.approx(10.0)
        module.bias.requires_grad_(False)
        assert parameter_efficiency(module, 0.10) == pytest.approx(10.0)

    def test_uses_count_parameters_when_available(self) -> None:
        from hqnn_forge.models import HybridBinaryClassifier

        model = HybridBinaryClassifier(
            n_input_features=4, n_qubits=4, n_layers=1,
            device_name="default.qubit", diff_method="parameter-shift",
        )
        expected = 0.5 / (model.count_parameters() / 1000)
        assert parameter_efficiency(model, 0.5) == pytest.approx(expected)

    def test_rejects_non_positive_count(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            parameter_efficiency(0, 0.5)
