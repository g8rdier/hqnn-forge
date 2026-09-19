"""
tests/test_multiclass_hybrid_classifier.py
==========================================
hqnn_forge.models.MulticlassHybridClassifier on a synthetic 3-class problem:
output shapes, probability semantics under both strategies, gradient flow to
every class head and to the quantum weights, and a short training run that
has to beat chance.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from hqnn_forge.models import MulticlassHybridClassifier

N_FEATURES = 6
N_QUBITS = 4
N_LAYERS = 1
N_CLASSES = 3
BATCH = 9
CPU = {"device_name": "default.qubit", "diff_method": "backprop"}


def _model(**kwargs: object) -> MulticlassHybridClassifier:
    torch.manual_seed(0)
    options: dict[str, object] = {"n_classes": N_CLASSES, **CPU, **kwargs}
    return MulticlassHybridClassifier(
        n_input_features=N_FEATURES,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        **options,  # type: ignore[arg-type]
    )


def _three_class_dataset(
    n_per_class: int = 12, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Three Gaussian blobs at distinct centres in a 6-D feature space."""
    torch.manual_seed(seed)
    centres = torch.tensor(
        [[2.0, 0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0, 0], [0, 0, 2.0, 0, 0, 0]], dtype=torch.float32
    )
    X = torch.cat([centres[c] + 0.4 * torch.randn(n_per_class, N_FEATURES) for c in range(3)])
    y = torch.repeat_interleave(torch.arange(3), n_per_class)
    perm = torch.randperm(X.shape[0])
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Shapes and semantics
# ---------------------------------------------------------------------------


class TestOutputs:
    @pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
    def test_shapes(self, strategy: str) -> None:
        model = _model(strategy=strategy)
        x = torch.randn(BATCH, N_FEATURES)
        assert model(x).shape == (BATCH, N_CLASSES)
        assert model.predict_proba(x).shape == (BATCH, N_CLASSES)
        assert model.predict(x).shape == (BATCH,)

    @pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
    def test_probabilities_are_a_distribution(self, strategy: str) -> None:
        model = _model(strategy=strategy)
        probs = model.predict_proba(torch.randn(BATCH, N_FEATURES))
        assert probs.min().item() >= 0.0
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-6, rtol=0)

    def test_softmax_probabilities_are_the_softmax_of_the_logits(self) -> None:
        model = _model()
        x = torch.randn(BATCH, N_FEATURES)
        with torch.no_grad():
            expected = torch.softmax(model(x), dim=-1)
        torch.testing.assert_close(model.predict_proba(x), expected)

    def test_one_vs_rest_probabilities_are_normalised_sigmoids(self) -> None:
        model = _model(strategy="one_vs_rest")
        x = torch.randn(BATCH, N_FEATURES)
        with torch.no_grad():
            scores = torch.sigmoid(model(x))
        torch.testing.assert_close(model.predict_proba(x), scores / scores.sum(-1, keepdim=True))

    @pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
    def test_predict_is_the_argmax(self, strategy: str) -> None:
        model = _model(strategy=strategy)
        x = torch.randn(BATCH, N_FEATURES)
        labels = model.predict(x)
        assert labels.dtype == torch.long
        assert torch.equal(labels, model.predict_proba(x).argmax(dim=-1))
        assert labels.min().item() >= 0 and labels.max().item() < N_CLASSES

    def test_one_hot_targets(self) -> None:
        model = _model()
        y = torch.tensor([0, 2, 1])
        expected = torch.tensor([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
        assert torch.equal(model.one_hot(y), expected)

    def test_predict_runs_in_eval_mode_and_restores_it(self) -> None:
        model = _model(dropout_p=0.5)
        model.train()
        x = torch.randn(BATCH, N_FEATURES)
        a = model.predict_proba(x)
        b = model.predict_proba(x)
        assert torch.equal(a, b)  # dropout off inside predict_proba
        assert model.training and model.dropout.training


# ---------------------------------------------------------------------------
# Parameters and gradients
# ---------------------------------------------------------------------------


class TestParametersAndGradients:
    def test_parameter_count(self) -> None:
        model = _model()
        encoder = N_FEATURES * N_QUBITS + N_QUBITS
        quantum = N_LAYERS * N_QUBITS * 3
        heads = N_CLASSES * N_QUBITS + N_CLASSES
        assert model.count_parameters() == encoder + quantum + heads
        assert model.head.weight.shape == (N_CLASSES, N_QUBITS)
        assert f"n_classes={N_CLASSES}" in model.extra_repr()

    def test_quantum_parameters_do_not_grow_with_n_classes(self) -> None:
        three = _model()
        seven = _model(n_classes=7)
        assert (
            three.quantum_layer.qlayer.weights.numel()
            == seven.quantum_layer.qlayer.weights.numel()
        )
        assert seven.count_parameters() - three.count_parameters() == 4 * (N_QUBITS + 1)

    @pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
    def test_gradients_reach_every_class_head_and_the_quantum_weights(self, strategy: str) -> None:
        model = _model(strategy=strategy)
        X, y = _three_class_dataset(n_per_class=3)
        logits = model(X)
        if strategy == "softmax":
            loss = nn.CrossEntropyLoss()(logits, y)
        else:
            loss = nn.BCEWithLogitsLoss()(logits, model.one_hot(y))
        loss.backward()
        head_grad = model.head.weight.grad
        assert head_grad is not None
        # Every class head row receives a non-zero gradient.
        assert torch.all(head_grad.abs().sum(dim=1) > 0)
        assert model.head.bias.grad is not None and torch.all(model.head.bias.grad != 0)
        q_grad = model.quantum_layer.qlayer.weights.grad
        assert q_grad is not None and q_grad.abs().sum().item() > 0.0
        enc_grad = model.classical_encoder[0].weight.grad
        assert enc_grad is not None and enc_grad.abs().sum().item() > 0.0

    def test_quantum_init_matches_documented_sigma(self) -> None:
        torch.manual_seed(0)
        model = MulticlassHybridClassifier(
            n_input_features=16,
            n_qubits=16,
            n_layers=16,
            n_classes=3,
            **CPU,  # type: ignore[arg-type]
        )
        std = model.quantum_layer.qlayer.weights.detach().std().item()
        assert std == pytest.approx(math.pi / math.sqrt(16 * 16), rel=0.2)


# ---------------------------------------------------------------------------
# Training on the synthetic problem
# ---------------------------------------------------------------------------


class TestTraining:
    @pytest.mark.parametrize("strategy", ["softmax", "one_vs_rest"])
    def test_short_training_beats_chance(self, strategy: str) -> None:
        """
        Three well-separated blobs: 25 Adam steps must take the loss down and
        the training accuracy well above 1/3.
        """
        model = _model(strategy=strategy)
        X, y = _three_class_dataset(n_per_class=12)
        loss_fn = nn.CrossEntropyLoss() if strategy == "softmax" else nn.BCEWithLogitsLoss()
        target = y if strategy == "softmax" else model.one_hot(y)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
        losses = []
        for _ in range(25):
            optimiser.zero_grad()
            loss = loss_fn(model(X), target)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        assert losses[-1] < 0.6 * losses[0]
        accuracy = (model.predict(X) == y).float().mean().item()
        assert accuracy > 0.8


# ---------------------------------------------------------------------------
# Options and validation
# ---------------------------------------------------------------------------


class TestOptions:
    def test_iqp_encoding(self) -> None:
        model = _model(encoding_type="iqp")
        assert model(torch.randn(BATCH, N_FEATURES)).shape == (BATCH, N_CLASSES)

    def test_encoder_bypass(self) -> None:
        torch.manual_seed(0)
        model = MulticlassHybridClassifier(
            n_input_features=N_QUBITS,
            n_qubits=N_QUBITS,
            n_layers=1,
            n_classes=3,
            use_classical_encoder=False,
            **CPU,  # type: ignore[arg-type]
        )
        assert isinstance(model.classical_encoder, nn.Identity)
        assert model(torch.rand(BATCH, N_QUBITS) * 2 - 1).shape == (BATCH, 3)

    def test_bypass_requires_matching_widths(self) -> None:
        with pytest.raises(ValueError, match="must equal n_qubits"):
            MulticlassHybridClassifier(
                n_input_features=5,
                n_qubits=4,
                n_classes=3,
                use_classical_encoder=False,
                **CPU,  # type: ignore[arg-type]
            )

    def test_rejects_fewer_than_two_classes(self) -> None:
        with pytest.raises(ValueError, match="n_classes"):
            _model(n_classes=1)

    def test_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            _model(strategy="argmax")

    def test_rejects_unknown_encoding(self) -> None:
        with pytest.raises(ValueError, match="encoding_type"):
            _model(encoding_type="amplitude")
