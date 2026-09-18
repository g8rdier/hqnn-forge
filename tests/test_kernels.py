"""
tests/test_kernels.py
=====================
Unit tests for hqnn_forge.kernels.

Two of the encoders have closed-form kernels, which serve as oracles:

* Angle embedding (RX, product state): ``k(x, y) = Π_i cos²((x_i − y_i) / 2)``.
* Amplitude embedding: ``k(x, y) = (x·y)² / (‖x‖² ‖y‖²)``.

The IQP and re-uploading kernels are checked against the overlap circuit
``|⟨0| U†(x) U(y) |0⟩|²`` built directly in PennyLane.
"""

from __future__ import annotations

import math

import pennylane as qml
import pytest
import torch

from hqnn_forge.encoding import (
    AmplitudeEncodingLayer,
    DataReuploadingLayer,
    QuantumEncodingLayer,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.kernels import encoded_states, quantum_kernel_matrix

N_QUBITS = 3
M = 7  # samples


def _angles(n: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(n, N_QUBITS, dtype=torch.float64) * 2 * math.pi - math.pi


def _angle_layer(n_layers: int = 2) -> QuantumEncodingLayer:
    torch.manual_seed(0)
    return QuantumEncodingLayer(
        n_qubits=N_QUBITS, n_layers=n_layers, device_name="default.qubit", diff_method="backprop"
    )


ALL_LAYERS = [
    pytest.param(lambda: _angle_layer(), id="angle"),
    pytest.param(
        lambda: IQPEncodingLayer(
            n_qubits=N_QUBITS, n_layers=1, device_name="default.qubit", diff_method="backprop"
        ),
        id="iqp",
    ),
    pytest.param(
        lambda: AmplitudeEncodingLayer(
            n_qubits=N_QUBITS, n_layers=1, device_name="default.qubit", diff_method="backprop"
        ),
        id="amplitude",
    ),
    pytest.param(
        lambda: DataReuploadingLayer(
            n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="backprop"
        ),
        id="reuploading",
    ),
]


def _inputs_for(layer: torch.nn.Module) -> torch.Tensor:
    if isinstance(layer, AmplitudeEncodingLayer):
        torch.manual_seed(0)
        return torch.randn(M, layer.n_features, dtype=torch.float64)
    return _angles(M)


# ---------------------------------------------------------------------------
# Kernel matrix properties (the issue's acceptance criteria)
# ---------------------------------------------------------------------------


class TestKernelMatrixProperties:
    @pytest.mark.parametrize("build", ALL_LAYERS)
    def test_shape_symmetry_psd_and_unit_diagonal(self, build) -> None:
        layer = build()
        X = _inputs_for(layer)
        K = quantum_kernel_matrix(X, layer)
        assert K.shape == (M, M)
        assert K.dtype == torch.float64
        torch.testing.assert_close(K, K.T, rtol=0, atol=0)
        torch.testing.assert_close(
            K.diagonal(), torch.ones(M, dtype=torch.float64), atol=1e-10, rtol=0
        )
        eigenvalues = torch.linalg.eigvalsh(K)
        assert eigenvalues.min().item() >= -1e-10
        assert K.min().item() >= 0.0
        assert K.max().item() <= 1.0

    @pytest.mark.parametrize("build", ALL_LAYERS)
    def test_rectangular_matrix_matches_square_blocks(self, build) -> None:
        """K(X, Y) is the off-diagonal block of K([X; Y])."""
        layer = build()
        XY = _inputs_for(layer)
        X, Y = XY[:4], XY[4:]
        full = quantum_kernel_matrix(XY, layer)
        rect = quantum_kernel_matrix(X, layer, Y=Y)
        assert rect.shape == (4, M - 4)
        torch.testing.assert_close(rect, full[:4, 4:], atol=1e-12, rtol=0)

    def test_identical_rows_give_kernel_one(self) -> None:
        layer = _angle_layer()
        X = _angles(3)
        X[2] = X[0]
        K = quantum_kernel_matrix(X, layer)
        assert K[0, 2].item() == pytest.approx(1.0, abs=1e-12)

    def test_states_are_normalised(self) -> None:
        for build in (p.values[0] for p in ALL_LAYERS):
            layer = build()
            states = encoded_states(_inputs_for(layer), layer)
            assert states.shape == (M, 2**N_QUBITS)
            norms = torch.linalg.vector_norm(states, dim=1)
            torch.testing.assert_close(
                norms, torch.ones(M, dtype=torch.float64), atol=1e-12, rtol=0
            )


# ---------------------------------------------------------------------------
# Closed-form oracles
# ---------------------------------------------------------------------------


class TestClosedForms:
    def test_angle_kernel_is_product_of_cosines(self) -> None:
        """RX(x_i)|0⟩ is a product state, so the fidelity factorises per qubit."""
        X = _angles(M)
        K = quantum_kernel_matrix(X, _angle_layer())
        diff = X[:, None, :] - X[None, :, :]
        expected = torch.cos(diff / 2).pow(2).prod(dim=-1)
        torch.testing.assert_close(K, expected, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("n_layers", [1, 3])
    def test_angle_kernel_is_independent_of_the_ansatz(self, n_layers: int) -> None:
        """The variational unitary cancels in |⟨Φ(x)|V†V|Φ(y)⟩|²."""
        X = _angles(M)
        layer = _angle_layer(n_layers)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        K = quantum_kernel_matrix(X, layer)
        expected = torch.cos((X[:, None, :] - X[None, :, :]) / 2).pow(2).prod(dim=-1)
        torch.testing.assert_close(K, expected, atol=1e-12, rtol=0)

    def test_amplitude_kernel_is_squared_cosine_similarity(self) -> None:
        layer = AmplitudeEncodingLayer(
            n_qubits=N_QUBITS, n_layers=1, n_features=5, device_name="default.qubit"
        )
        torch.manual_seed(0)
        X = torch.randn(M, 5, dtype=torch.float64)
        K = quantum_kernel_matrix(X, layer)
        unit = X / torch.linalg.vector_norm(X, dim=1, keepdim=True)
        expected = (unit @ unit.T).pow(2)
        torch.testing.assert_close(K, expected, atol=1e-12, rtol=0)

    def test_amplitude_kernel_uses_the_layers_padding(self) -> None:
        """n_features < 2**n_qubits goes through prepare_inputs, not a raw QNode call."""
        layer = AmplitudeEncodingLayer(
            n_qubits=N_QUBITS, n_layers=1, n_features=2, device_name="default.qubit"
        )
        X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        K = quantum_kernel_matrix(X, layer)
        expected = torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]], dtype=torch.float64
        )
        torch.testing.assert_close(K, expected, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Overlap-circuit reference for the entangling encoders
# ---------------------------------------------------------------------------


def _overlap_kernel(layer: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    """|⟨0|U†(x_i)U(x_j)|0⟩|² from the layer's own tape, via qml.adjoint."""
    weights = {k: p.detach().to(torch.float64) for k, p in layer.qlayer.qnode_weights.items()}
    build = qml.workflow.construct_tape(layer.qlayer.qnode, level=0)
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def overlap(x: torch.Tensor, y: torch.Tensor) -> qml.measurements.ProbabilityMP:
        for op in build(y, **weights).operations:
            qml.apply(op)
        for op in reversed(build(x, **weights).operations):
            qml.adjoint(op)
        return qml.probs(wires=range(N_QUBITS))

    K = torch.empty(M, M, dtype=torch.float64)
    for i in range(M):
        for j in range(M):
            K[i, j] = torch.as_tensor(overlap(X[i], X[j]))[0]
    return K


class TestOverlapCircuitReference:
    @pytest.mark.parametrize("build", ALL_LAYERS[1:2] + ALL_LAYERS[3:])
    def test_matches_overlap_circuit(self, build) -> None:
        layer = build()
        X = _angles(M)
        K = quantum_kernel_matrix(X, layer)
        torch.testing.assert_close(K, _overlap_kernel(layer, X), atol=1e-10, rtol=0)

    def test_reuploading_kernel_depends_on_the_weights(self) -> None:
        """Unlike the single-upload encoders, the weights sit between uploads."""
        X = _angles(M)
        layer = DataReuploadingLayer(
            n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="backprop"
        )
        torch.manual_seed(0)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        K1 = quantum_kernel_matrix(X, layer)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        K2 = quantum_kernel_matrix(X, layer)
        assert not torch.allclose(K1, K2, atol=1e-3)


# ---------------------------------------------------------------------------
# sklearn integration and validation
# ---------------------------------------------------------------------------


class TestUsage:
    def test_precomputed_svm_separates_a_toy_problem(self) -> None:
        sklearn = pytest.importorskip("sklearn")
        from sklearn.svm import SVC  # noqa: F401 - imported for the skip above

        torch.manual_seed(0)
        layer = _angle_layer(1)
        # Two clusters of angles, one around -π/2 and one around +π/2.
        X = torch.cat(
            [
                torch.randn(20, N_QUBITS) * 0.3 - math.pi / 2,
                torch.randn(20, N_QUBITS) * 0.3 + math.pi / 2,
            ]
        )
        y = torch.cat([torch.zeros(20), torch.ones(20)]).numpy()
        X_train, X_test = X[::2], X[1::2]
        y_train, y_test = y[::2], y[1::2]
        svm = sklearn.svm.SVC(kernel="precomputed")
        svm.fit(quantum_kernel_matrix(X_train, layer).numpy(), y_train)
        pred = svm.predict(quantum_kernel_matrix(X_test, layer, Y=X_train).numpy())
        assert (pred == y_test).mean() == 1.0

    def test_does_not_touch_gradients_or_weights(self) -> None:
        layer = _angle_layer()
        before = layer.qlayer.weights.detach().clone()
        X = _angles(M).requires_grad_(True)
        K = quantum_kernel_matrix(X, layer)
        assert not K.requires_grad
        assert layer.qlayer.weights.grad is None
        assert torch.equal(layer.qlayer.weights.detach(), before)

    def test_rejects_non_layers(self) -> None:
        with pytest.raises(TypeError, match="encoding layer"):
            quantum_kernel_matrix(_angles(2), torch.nn.Linear(3, 3))

    def test_rejects_wrong_shapes(self) -> None:
        layer = _angle_layer()
        with pytest.raises(ValueError, match="n_samples, n_features"):
            quantum_kernel_matrix(torch.zeros(N_QUBITS), layer)
        with pytest.raises(ValueError, match="no samples"):
            quantum_kernel_matrix(torch.zeros(0, N_QUBITS), layer)
        with pytest.raises(ValueError, match="same number of features"):
            quantum_kernel_matrix(_angles(2), layer, Y=torch.zeros(2, N_QUBITS + 1))
