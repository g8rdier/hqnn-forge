"""
tests/test_gradient_variance.py
================================
hqnn_forge.diagnostics.gradient_variance: mechanics, and the two physical
signatures it exists to show.

Thresholds are set well inside what was measured over five seeds:
uniform-init total variance falls by 5.3–5.9x from 2 to 6 qubits (asserted:
> 3x), and at 8 qubits with zero inputs restricted init keeps 1.32–1.80x the
uniform variance (asserted: > 1.2x).
"""

from __future__ import annotations

import math

import pennylane as qml
import pytest
import torch

from hqnn_forge.diagnostics import (
    GradientVarianceResult,
    format_sweep,
    gradient_variance,
    gradient_variance_sweep,
)
from hqnn_forge.encoding import QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier

CPU = dict(device_name="default.qubit", diff_method="backprop")


def _layer(n_qubits: int, n_layers: int = 2) -> QuantumEncodingLayer:
    return QuantumEncodingLayer(n_qubits=n_qubits, n_layers=n_layers, **CPU)


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _lightning_available() -> bool:
    try:
        qml.device("lightning.qubit", wires=1)
        return True
    except Exception:  # noqa: BLE001
        return False


def _two_weight_layer() -> torch.nn.Module:
    """A TorchLayer with two trainable arguments, which no library layer has."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, w1, w2):  # type: ignore[no-untyped-def]
        qml.AngleEmbedding(inputs, wires=range(2))
        qml.RX(w1[0], wires=0)
        qml.RY(w2[0], wires=1)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

    class TwoWeightLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_qubits = 2
            self.qlayer = qml.qnn.TorchLayer(circuit, {"w1": (1,), "w2": (1,)})

    return TwoWeightLayer()


def _result(init: str, n_qubits: int, total: float) -> GradientVarianceResult:
    return GradientVarianceResult(
        layer_type="QuantumEncodingLayer",
        n_qubits=n_qubits,
        n_layers=1,
        init=init,
        input_scale=0.0,
        n_samples=2,
        total_variance=total,
        mean_variance=total,
        per_parameter=torch.tensor([total]),
    )


class TestPhysics:
    def test_uniform_init_variance_decays_with_qubits(self) -> None:
        small = gradient_variance(_layer(2), n_samples=100, generator=_gen())
        large = gradient_variance(_layer(6), n_samples=100, generator=_gen())
        assert small.total_variance > 3 * large.total_variance

    def test_restricted_init_keeps_more_variance_near_zero_input(self) -> None:
        layer = _layer(8)
        uniform = gradient_variance(layer, n_samples=100, input_scale=0.0, generator=_gen())
        restricted = gradient_variance(
            layer, n_samples=100, init="restricted", input_scale=0.0, generator=_gen()
        )
        assert restricted.total_variance > 1.2 * uniform.total_variance

    def test_zero_weights_and_zero_input_give_zero_gradient(self) -> None:
        """|0...0> is a stationary point of <Z_0>: every gradient vanishes."""
        result = gradient_variance(
            _layer(3), n_samples=4, init=lambda w: w.zero_(), input_scale=0.0
        )
        assert result.total_variance == pytest.approx(0.0, abs=1e-12)


class TestMechanics:
    def test_result_fields(self) -> None:
        r = gradient_variance(_layer(3, 2), n_samples=5, init="block_local", input_scale=1.0)
        assert isinstance(r, GradientVarianceResult)
        assert (r.layer_type, r.n_qubits, r.n_layers, r.init, r.n_samples) == (
            "QuantumEncodingLayer",
            3,
            2,
            "block_local",
            5,
        )
        assert r.input_scale == 1.0
        assert r.per_parameter.shape == (2, 3, 3)
        assert r.total_variance == pytest.approx(float(r.per_parameter.sum()))
        assert r.mean_variance == pytest.approx(float(r.per_parameter.mean()))
        assert set(r.to_dict()) == {
            "layer_type",
            "n_qubits",
            "n_layers",
            "init",
            "input_scale",
            "n_samples",
            "total_variance",
            "mean_variance",
        }

    def test_matches_a_hand_rolled_estimate(self) -> None:
        layer = _layer(2, 1)
        gen = _gen(3)
        result = gradient_variance(layer, n_samples=6, generator=_gen(3))
        grads = []
        w = layer.qlayer.weights
        original = w.detach().clone()
        for _ in range(6):
            with torch.no_grad():
                w.copy_(torch.rand(w.shape, generator=gen) * 2 * math.pi)
            x = (torch.rand(1, 2, generator=gen) * 2 - 1) * math.pi
            (g,) = torch.autograd.grad(layer(x)[..., 0].sum(), w)
            grads.append(g.double())
        with torch.no_grad():
            w.copy_(original)
        torch.testing.assert_close(result.per_parameter, torch.stack(grads).var(dim=0))

    def test_weights_and_grad_are_restored(self) -> None:
        layer = _layer(3)
        before = layer.qlayer.weights.detach().clone()
        layer.qlayer.weights.grad = torch.ones_like(before)
        gradient_variance(layer, n_samples=3)
        torch.testing.assert_close(layer.qlayer.weights.detach(), before, rtol=0, atol=0)
        torch.testing.assert_close(layer.qlayer.weights.grad, torch.ones_like(before))

    def test_weights_restored_when_cost_raises(self) -> None:
        layer = _layer(2)
        before = layer.qlayer.weights.detach().clone()

        def bad_cost(_: torch.Tensor) -> torch.Tensor:
            raise KeyError("boom")

        with pytest.raises(KeyError):
            gradient_variance(layer, n_samples=3, cost_fn=bad_cost)
        torch.testing.assert_close(layer.qlayer.weights.detach(), before, rtol=0, atol=0)

    def test_reproducible_and_global_rng_untouched(self) -> None:
        layer = _layer(3)
        torch.manual_seed(42)
        state = torch.get_rng_state()
        a = gradient_variance(layer, n_samples=5, init="restricted", generator=_gen(1))
        assert torch.equal(torch.get_rng_state(), state)
        b = gradient_variance(layer, n_samples=5, init="restricted", generator=_gen(1))
        torch.testing.assert_close(a.per_parameter, b.per_parameter, rtol=0, atol=0)

    def test_custom_cost(self) -> None:
        layer = _layer(3)
        z0 = gradient_variance(layer, n_samples=5, generator=_gen())
        z_all = gradient_variance(
            layer, n_samples=5, generator=_gen(), cost_fn=lambda out: out.sum()
        )
        assert not torch.allclose(z0.per_parameter, z_all.per_parameter)

    def test_accepts_model_and_iqp_layer(self) -> None:
        model = HybridBinaryClassifier(n_input_features=5, n_qubits=3, n_layers=1, **CPU)
        r = gradient_variance(model, n_samples=3)
        assert r.layer_type == "QuantumEncodingLayer" and r.n_qubits == 3
        iqp = gradient_variance(IQPEncodingLayer(n_qubits=3, n_layers=1, **CPU), n_samples=3)
        assert iqp.layer_type == "IQPEncodingLayer"

    @pytest.mark.parametrize(
        "kwargs, error, match",
        [
            (dict(n_samples=1), ValueError, "n_samples must be >= 2"),
            (dict(input_scale=-1.0), ValueError, "input_scale must be >= 0"),
            (dict(init="xavier"), ValueError, "unknown init 'xavier'"),
            (dict(cost_fn=lambda out: out), ValueError, "cost_fn must return a scalar"),
            (
                dict(cost_fn=lambda out: out[..., 0].sum().item()),
                ValueError,
                "cost_fn must return a 0-d Tensor",
            ),
        ],
    )
    def test_argument_errors(self, kwargs: dict, error: type, match: str) -> None:
        with pytest.raises(error, match=match):
            gradient_variance(_layer(2), **kwargs)

    def test_unsupported_target(self) -> None:
        with pytest.raises(TypeError, match="got Linear"):
            gradient_variance(torch.nn.Linear(2, 1))

    def test_several_weight_tensors_are_rejected(self) -> None:
        """Measuring one of two weight tensors would understate total_variance."""
        with pytest.raises(NotImplementedError, match="w1, w2"):
            gradient_variance(_two_weight_layer(), n_samples=3)

    def test_equality_is_a_bool_and_the_result_hashes(self) -> None:
        """per_parameter is compare=False, so == does not return a Tensor."""
        a, b = _result("uniform", 3, 0.25), _result("uniform", 3, 0.25)
        assert isinstance(a == b, bool) and a == b
        assert a != _result("uniform", 4, 0.25)
        assert a in [b]
        assert {a: "seen"}[b] == "seen"


class TestDefaultDevice:
    @pytest.mark.skipif(not _lightning_available(), reason="pennylane-lightning not installed")
    @pytest.mark.parametrize("layer_cls", [QuantumEncodingLayer, IQPEncodingLayer])
    def test_estimates_on_the_library_default_device(self, layer_cls: type) -> None:
        """Every other test pins default.qubit/backprop; the default is lightning/adjoint."""
        layer = layer_cls(n_qubits=3, n_layers=2)
        before = layer.qlayer.weights.detach().clone()
        result = gradient_variance(layer, n_samples=5, generator=_gen())
        assert result.total_variance > 0.0
        assert result.per_parameter.shape == before.shape
        torch.testing.assert_close(layer.qlayer.weights.detach(), before, rtol=0, atol=0)


class TestSweep:
    def test_grid_order_and_table(self) -> None:
        results = gradient_variance_sweep(
            lambda q, l: QuantumEncodingLayer(n_qubits=q, n_layers=l, **CPU),
            qubit_counts=(2, 3),
            layer_counts=(1, 2),
            n_samples=3,
            generator=_gen(),
        )
        assert [(r.n_qubits, r.n_layers) for r in results] == [(2, 1), (2, 2), (3, 1), (3, 2)]
        table = format_sweep(results).splitlines()
        assert table[0].split() == [
            "init",
            "qubits",
            "layers",
            "total",
            "var",
            "mean",
            "var",
            "ratio",
        ]
        assert len(table) == 2 + 4
        # First row per (init, layers) has no ratio; the repeat for 3 qubits has one
        assert len(table[2].split()) == 5 and len(table[4].split()) == 6
        expected_ratio = results[2].total_variance / results[0].total_variance
        assert float(table[4].split()[-1]) == pytest.approx(expected_ratio, abs=1e-3)

    def test_a_zero_previous_row_does_not_blank_the_next_ratio(self) -> None:
        """A stationary point measures exactly 0.0; the row after it still has a ratio."""
        rows = format_sweep(
            [_result("uniform", 2, 0.0), _result("uniform", 3, 4e-3), _result("uniform", 4, 2e-3)]
        ).splitlines()
        assert len(rows[2].split()) == 5  # no previous row at all
        assert rows[3].split()[-1] == "n/a"  # previous row was zero
        assert float(rows[4].split()[-1]) == pytest.approx(0.5)
