"""
tests/test_hybrid_classifier.py
================================
Tests specific to hqnn_forge.models.HybridBinaryClassifier.  The inference,
bypass and encoding-type tests shared with every classifier live in
tests/test_binary_classifiers.py.
"""

from __future__ import annotations

from hqnn_forge.models import HybridBinaryClassifier


class TestInitStrategies:
    def test_restricted_strategy(self) -> None:
        model = HybridBinaryClassifier(
            n_input_features=4,
            n_qubits=4,
            n_layers=2,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="parameter-shift",
            init_strategy="restricted",
        )
        assert model is not None

    def test_block_local_strategy(self) -> None:
        model = HybridBinaryClassifier(
            n_input_features=4,
            n_qubits=4,
            n_layers=2,
            use_classical_encoder=False,
            device_name="default.qubit",
            diff_method="parameter-shift",
            init_strategy="block_local",
        )
        assert model is not None
