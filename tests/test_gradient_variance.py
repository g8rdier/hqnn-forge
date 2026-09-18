"""
tests/test_gradient_variance.py
================================
hqnn_forge.diagnostics.gradient_variance: mechanics, and the two physical
signatures it exists to show.

Thresholds are set well inside what was measured over five seeds:
uniform-init total variance falls by 5.3–5.9x from 2 to 6 qubits (asserted:
> 3x).  The initialiser's measured effect is pinned in TestMeasuredInitClaims,
which quotes its own ranges.
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
from hqnn_forge.encoding import DataReuploadingLayer, QuantumEncodingLayer
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


@pytest.fixture(scope="module")
def measured() -> dict[tuple[str, int, float], float]:
    """
    Mean gradient variance per (init, n_qubits, input_scale), 200 draws, seed 0.

    Computed once for TestMeasuredInitClaims: the 8-qubit estimates dominate
    this module's run time, and each is shared by several claims.
    """
    layers = {n: _layer(n) for n in (4, 8)}
    return {
        (init, n, scale): gradient_variance(
            layers[n], n_samples=200, init=init, input_scale=scale, generator=_gen()
        ).mean_variance
        for init in ("uniform", "restricted")
        for n in (4, 8)
        for scale in (0.0, math.pi)
    }


class TestMeasuredInitClaims:
    """
    The statements in hqnn_forge.initializers.restricted_variance's
    "measured" section (#122), pinned with the diagnostic.  Over five seeds
    at these 200 draws: the restricted/uniform ratio at 8 qubits is 0.86–1.23
    with inputs in (-π, π) and 1.69–2.07 with zero input, the zero-input
    ratio grows 1.5–2.0x from 4 to 8 qubits, and from 4 to 8 qubits at
    (-π, π) restricted init loses 8.4–10.6x against uniform's 8.1–10.7x.
    Every threshold below leaves a clear margin to those ranges.
    """

    @staticmethod
    def _ratio(measured: dict, n: int, scale: float) -> float:
        return measured["restricted", n, scale] / measured["uniform", n, scale]

    def test_no_benefit_with_inputs_spread_over_pi(self, measured: dict) -> None:
        """What the classifiers feed the circuit: restricted ≈ uniform."""
        ratio = self._ratio(measured, 8, math.pi)
        assert 0.6 < ratio < 1.5, ratio

    def test_benefit_near_zero_input(self, measured: dict) -> None:
        """Near-zero input: more variance, but by a factor, not an order of magnitude."""
        ratio = self._ratio(measured, 8, 0.0)
        assert 1.4 < ratio < 3.0, ratio

    def test_zero_input_benefit_grows_with_qubits(self, measured: dict) -> None:
        """The zero-input gain is not a constant factor over the measured range."""
        small, large = self._ratio(measured, 4, 0.0), self._ratio(measured, 8, 0.0)
        assert large > 1.25 * small, (small, large)

    def test_restricted_init_decays_with_qubits_like_uniform(self, measured: dict) -> None:
        """At (-π, π) the init does not change the decay: both lose > 5x from 4 to 8 qubits."""
        decay = {
            init: measured[init, 4, math.pi] / measured[init, 8, math.pi]
            for init in ("uniform", "restricted")
        }
        assert decay["uniform"] > 5 and decay["restricted"] > 5, decay
        assert 0.6 < decay["restricted"] / decay["uniform"] < 1.6, decay


class TestPhysics:
    def test_uniform_init_variance_decays_with_qubits(self) -> None:
        small = gradient_variance(_layer(2), n_samples=100, generator=_gen())
        large = gradient_variance(_layer(6), n_samples=100, generator=_gen())
        assert small.total_variance > 3 * large.total_variance

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

    def test_reuploading_layer_is_measured_only_without_input_scaling(self) -> None:
        """Trainable input_scaling is a second tensor next to weights, so it is refused."""
        plain = DataReuploadingLayer(n_qubits=2, n_layers=2, **CPU)
        assert gradient_variance(plain, n_samples=3).per_parameter.shape == (2, 2, 3)
        scaled = DataReuploadingLayer(n_qubits=2, n_layers=2, trainable_input_scaling=True, **CPU)
        with pytest.raises(NotImplementedError, match="input_scaling, weights"):
            gradient_variance(scaled, n_samples=3)

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
