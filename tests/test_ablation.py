"""
tests/test_ablation.py
======================
Unit tests for hqnn_forge.utils.disable_quantum_layer.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier
from hqnn_forge.utils import disable_quantum_layer

CPU = dict(device_name="default.qubit", diff_method="backprop")
N_FEATURES, N_QUBITS = 6, 3


def _model(cls: type, **kw: object) -> nn.Module:
    torch.manual_seed(0)
    return cls(n_input_features=N_FEATURES, n_qubits=N_QUBITS, n_layers=2, **CPU, **kw)


MODELS = [pytest.param(HybridBinaryClassifier, id="serial"), pytest.param(ParallelHybridClassifier, id="parallel")]


@pytest.fixture
def x() -> torch.Tensor:
    return torch.randn(5, N_FEATURES, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("cls", MODELS)
class TestAblation:
    def test_output_ignores_quantum_weights(self, cls: type, x: torch.Tensor) -> None:
        model = _model(cls)
        with disable_quantum_layer(model):
            before = model(x).detach()
            with torch.no_grad():
                model.quantum_layer.qlayer.weights.add_(1.0)
            after = model(x).detach()
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_no_gradient_reaches_the_quantum_branch(self, cls: type, x: torch.Tensor) -> None:
        model = _model(cls)
        with disable_quantum_layer(model):
            model(x).sum().backward()
        assert model.quantum_layer.qlayer.weights.grad is None
        for p in model.classical_encoder.parameters():
            assert p.grad is None
        assert model.head.weight.grad is not None

    def test_circuit_is_not_executed(self, cls: type, x: torch.Tensor, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _model(cls)

        def boom(*_: object, **__: object) -> None:
            raise AssertionError("circuit executed")

        monkeypatch.setattr(model.quantum_layer.qlayer, "forward", boom)
        with disable_quantum_layer(model):
            model(x)

    def test_forward_is_restored(self, cls: type, x: torch.Tensor) -> None:
        model = _model(cls)
        model.eval()
        with torch.no_grad():
            expected = model(x)
            with disable_quantum_layer(model):
                assert not torch.allclose(model(x), expected)
            torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        assert "forward" not in vars(model.quantum_layer)

    def test_restored_after_an_exception(self, cls: type) -> None:
        model = _model(cls)
        with pytest.raises(KeyError):
            with disable_quantum_layer(model):
                raise KeyError("inside")
        assert "forward" not in vars(model.quantum_layer)

    def test_predict_proba_works_inside(self, cls: type, x: torch.Tensor) -> None:
        model = _model(cls)
        with disable_quantum_layer(model):
            probs = model.predict_proba(x)
        assert probs.shape == (5,)

    def test_nested_use_raises(self, cls: type) -> None:
        model = _model(cls)
        with disable_quantum_layer(model):
            with pytest.raises(RuntimeError, match="cannot be nested"):
                with disable_quantum_layer(model):
                    pass
        assert "forward" not in vars(model.quantum_layer)


class TestExactReplacement:
    def test_serial_output_is_head_of_constant(self, x: torch.Tensor) -> None:
        model = _model(HybridBinaryClassifier)
        with torch.no_grad(), disable_quantum_layer(model, fill=0.25):
            out = model(x)
            expected = model.head(torch.full((5, N_QUBITS), 0.25))
        torch.testing.assert_close(out, expected)

    def test_parallel_output_keeps_the_classical_branch(self, x: torch.Tensor) -> None:
        model = _model(ParallelHybridClassifier)
        with torch.no_grad(), disable_quantum_layer(model):
            out = model(x)
            fused = torch.cat([model.classical_branch(x), torch.zeros(5, N_QUBITS)], dim=-1)
            expected = model.head(fused)
        torch.testing.assert_close(out, expected)

    def test_iqp_layer_is_supported(self, x: torch.Tensor) -> None:
        model = _model(HybridBinaryClassifier, encoding_type="iqp")
        with torch.no_grad(), disable_quantum_layer(model) as layer:
            assert layer is model.quantum_layer
            torch.testing.assert_close(layer(torch.randn(2, N_QUBITS)), torch.zeros(2, N_QUBITS))

    def test_training_inside_updates_only_live_parameters(self, x: torch.Tensor) -> None:
        model = _model(ParallelHybridClassifier)
        quantum_before = model.quantum_layer.qlayer.weights.detach().clone()
        encoder_before = [p.detach().clone() for p in model.classical_encoder.parameters()]
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        with disable_quantum_layer(model):
            for _ in range(3):
                opt.zero_grad()
                model(x).pow(2).mean().backward()
                opt.step()
        torch.testing.assert_close(model.quantum_layer.qlayer.weights.detach(), quantum_before, rtol=0, atol=0)
        for before, p in zip(encoder_before, model.classical_encoder.parameters()):
            torch.testing.assert_close(p.detach(), before, rtol=0, atol=0)


class TestValidation:
    def test_model_without_quantum_layer(self) -> None:
        with pytest.raises(TypeError, match="expects a model with a quantum_layer.*got Linear"):
            with disable_quantum_layer(nn.Linear(2, 1)):
                pass

    @pytest.mark.parametrize("fill", [-1.5, 1.01])
    def test_fill_out_of_range(self, fill: float) -> None:
        model = _model(HybridBinaryClassifier)
        with pytest.raises(ValueError, match=r"fill must lie in \[-1, 1\]"):
            with disable_quantum_layer(model, fill=fill):
                pass
