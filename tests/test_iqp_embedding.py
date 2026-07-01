"""
tests/test_iqp_embedding.py
===========================
Unit tests for hqnn_forge.encoding.iqp_embedding.IQPEncodingLayer.
"""
from __future__ import annotations

import math
import pytest
import torch

from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.initializers.restricted_variance import restricted_normal_init_


class TestIQPEncodingLayer:
    def test_forward_shape_and_bounds(self) -> None:
        batch_size = 4
        n_qubits = 6
        n_layers = 2
        
        layer = IQPEncodingLayer(n_qubits=n_qubits, n_layers=n_layers)
        # Apply restricted initialization
        restricted_normal_init_(layer.qlayer.weights, n_qubits=n_qubits, n_layers=n_layers)
        
        x = torch.randn(batch_size, n_qubits)
        out = layer(x)
        
        assert out.shape == (batch_size, n_qubits)
        # Pauli-Z expectations must be in [-1, 1]
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_mismatched_feature_dim_raises(self) -> None:
        layer = IQPEncodingLayer(n_qubits=4)
        x_wrong = torch.randn(2, 5)
        with pytest.raises(ValueError, match="does not match n_qubits=4"):
            layer(x_wrong)

    def test_gradients_flow(self) -> None:
        layer = IQPEncodingLayer(n_qubits=3, n_layers=1)
        restricted_normal_init_(layer.qlayer.weights, n_qubits=3, n_layers=1)
        
        x = torch.randn(2, 3, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients flow back to the inputs
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0
        
        # Check gradients flow to the weights
        weights = layer.qlayer.weights
        assert weights.grad is not None
        assert weights.grad.abs().sum().item() > 0.0
