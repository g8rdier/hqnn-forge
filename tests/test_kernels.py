"""
tests/test_kernels.py
=====================
Unit tests for hqnn_forge.kernels.

Two of the encoders have closed-form kernels, which serve as oracles:

* Angle embedding (RX, product state): ``k(x, y) = Π_i cos²((x_i − y_i) / 2)``.
* Amplitude embedding: ``k(x, y) = (x·y)² / (‖x‖² ‖y‖²)``.

The IQP and re-uploading kernels are checked two ways: against the overlap
circuit ``|⟨0| U†(x) U(y) |0⟩|²`` built from the layer's own tape, and against
states from circuits written out in this file from the documented topology,
which do not go through the replayed tape at all.  ``TestMatchesForward``
also checks that the replayed states reproduce ``layer(x)``: ⟨Z_i⟩ computed
from ``encoded_states`` must equal what the layer's own forward returns.
"""

from __future__ import annotations

import math

import pennylane as qml
import pytest
import torch

from hqnn_forge import kernels
from hqnn_forge.encoding import (
    AmplitudeEncodingLayer,
    DataReuploadingLayer,
    QuantumEncodingLayer,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.kernels import encoded_states, kernel_from_states, quantum_kernel_matrix

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
        # No clamping in the implementation, so these bound the actual values.
        assert K.min().item() >= 0.0
        assert K.max().item() <= 1.0 + 1e-12

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
# The replayed circuit is the layer's circuit
# ---------------------------------------------------------------------------


def _z_expectations(states: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """⟨Z_i⟩ per row of a state batch; wire 0 is the most significant bit."""
    probs = states.abs().pow(2)
    index = torch.arange(2**n_qubits)
    signs = torch.stack(
        [
            1.0 - 2.0 * ((index >> (n_qubits - 1 - i)) & 1).to(torch.float64)
            for i in range(n_qubits)
        ]
    )
    return probs @ signs.T


def _reuploading_with_random_scaling(**kwargs) -> DataReuploadingLayer:
    torch.manual_seed(1)
    layer = DataReuploadingLayer(
        n_qubits=N_QUBITS, n_layers=2, trainable_input_scaling=True, **kwargs
    )
    with torch.no_grad():
        layer.qlayer.input_scaling.uniform_(0.5, 2.0)
    return layer


FORWARD_LAYERS = [
    *ALL_LAYERS,
    pytest.param(
        lambda: _reuploading_with_random_scaling(
            device_name="default.qubit", diff_method="backprop"
        ),
        id="reuploading-scaled",
    ),
    # The library defaults: lightning.qubit with adjoint, whose QNodes are
    # wrapped in a batch-expanding transform the replay must see through.
    pytest.param(lambda: QuantumEncodingLayer(n_qubits=N_QUBITS), id="angle-lightning"),
    pytest.param(lambda: IQPEncodingLayer(n_qubits=N_QUBITS), id="iqp-lightning"),
    pytest.param(
        lambda: AmplitudeEncodingLayer(n_qubits=N_QUBITS, n_features=5), id="amplitude-lightning"
    ),
    pytest.param(lambda: _reuploading_with_random_scaling(), id="reuploading-lightning"),
]


class TestMatchesForward:
    @pytest.mark.parametrize("build", FORWARD_LAYERS)
    def test_states_reproduce_forward(self, build) -> None:
        layer = build()
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        X = _inputs_for(layer)
        with torch.no_grad():
            expected = layer(X.to(torch.float32)).to(torch.float64)
        actual = _z_expectations(encoded_states(X, layer), N_QUBITS)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


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


def _written_out_kernel(states: list[torch.Tensor]) -> torch.Tensor:
    S = torch.stack([torch.as_tensor(s).to(torch.complex128) for s in states])
    return (S @ S.conj().T).abs().pow(2)


def _iqp_reference(X: torch.Tensor) -> torch.Tensor:
    """qml.IQPEmbedding's own decomposition; the ansatz after it cancels."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def state(x: torch.Tensor) -> qml.measurements.StateMP:
        qml.IQPEmbedding(x, wires=range(N_QUBITS))
        return qml.state()

    return _written_out_kernel([state(x) for x in X])


def _reuploading_reference(X: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """RX(x) upload, CNOT ring, Rot on every qubit, repeated per layer."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def state(x: torch.Tensor) -> qml.measurements.StateMP:
        for layer in range(weights.shape[0]):
            for q in range(N_QUBITS):
                qml.RX(x[q], wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
            for q in range(N_QUBITS):
                qml.Rot(*weights[layer, q], wires=q)
        return qml.state()

    return _written_out_kernel([state(x) for x in X])


def _layers(*ids: str) -> list:
    selected = [p for p in ALL_LAYERS if p.id in ids]
    assert [p.id for p in selected] == list(ids)
    return selected


class TestOverlapCircuitReference:
    @pytest.mark.parametrize("build", _layers("iqp", "reuploading"))
    def test_matches_overlap_circuit(self, build) -> None:
        layer = build()
        X = _angles(M)
        K = quantum_kernel_matrix(X, layer)
        torch.testing.assert_close(K, _overlap_kernel(layer, X), atol=1e-10, rtol=0)

    def test_iqp_matches_a_written_out_circuit(self) -> None:
        layer = IQPEncodingLayer(
            n_qubits=N_QUBITS, n_layers=1, device_name="default.qubit", diff_method="backprop"
        )
        X = _angles(M)
        K = quantum_kernel_matrix(X, layer)
        torch.testing.assert_close(K, _iqp_reference(X), atol=1e-10, rtol=0)

    def test_reuploading_matches_a_written_out_circuit(self) -> None:
        layer = DataReuploadingLayer(
            n_qubits=N_QUBITS, n_layers=3, device_name="default.qubit", diff_method="backprop"
        )
        torch.manual_seed(2)
        with torch.no_grad():
            layer.qlayer.weights.uniform_(0, 2 * math.pi)
        weights = layer.qlayer.weights.detach().to(torch.float64)
        X = _angles(M)
        K = quantum_kernel_matrix(X, layer)
        torch.testing.assert_close(K, _reuploading_reference(X, weights), atol=1e-10, rtol=0)

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

    def test_reuploading_last_block_cancels(self) -> None:
        """weights[-1] follows the last upload, so it drops out as V†V does."""
        X = _angles(M)
        torch.manual_seed(0)
        layer = DataReuploadingLayer(
            n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="backprop"
        )
        K1 = quantum_kernel_matrix(X, layer)
        with torch.no_grad():
            layer.qlayer.weights[-1].uniform_(0, 2 * math.pi)
        torch.testing.assert_close(quantum_kernel_matrix(X, layer), K1, atol=1e-12, rtol=0)
        with torch.no_grad():
            layer.qlayer.weights[0].uniform_(0, 2 * math.pi)
        assert not torch.allclose(quantum_kernel_matrix(X, layer), K1, atol=1e-3)


# ---------------------------------------------------------------------------
# sklearn integration and validation
# ---------------------------------------------------------------------------


class TestUsage:
    def test_precomputed_svm_separates_a_toy_problem(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.svm import SVC

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
        svm = SVC(kernel="precomputed")
        svm.fit(quantum_kernel_matrix(X_train, layer).numpy(), y_train)
        pred = svm.predict(quantum_kernel_matrix(X_test, layer, Y=X_train).numpy())
        assert (pred == y_test).mean() == 1.0

    def test_reused_states_match_the_rectangular_matrix(self) -> None:
        layer = _angle_layer()
        X_train, X_test = _angles(5, seed=0), _angles(3, seed=1)
        S_train = encoded_states(X_train, layer)
        torch.testing.assert_close(
            kernel_from_states(S_train), quantum_kernel_matrix(X_train, layer), atol=0, rtol=0
        )
        torch.testing.assert_close(
            kernel_from_states(encoded_states(X_test, layer), S_train),
            quantum_kernel_matrix(X_test, layer, Y=X_train),
            atol=0,
            rtol=0,
        )

    def test_unnormalised_state_is_not_hidden(self) -> None:
        """The diagonal is not clamped, so a bad state shows up as K[i, i] != 1."""
        states = encoded_states(_angles(3), _angle_layer())
        states[1] *= 1.1
        K = kernel_from_states(states)
        assert K[1, 1].item() == pytest.approx(1.1**4, abs=1e-12)
        assert K[0, 0].item() == pytest.approx(1.0, abs=1e-12)

    def test_does_not_touch_gradients_or_weights(self) -> None:
        layer = _angle_layer()
        before = layer.qlayer.weights.detach().clone()
        X = _angles(M).requires_grad_(True)
        K = quantum_kernel_matrix(X, layer)
        assert not K.requires_grad
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
        with pytest.raises(ValueError, match="state dimension"):
            kernel_from_states(torch.zeros(2, 4, dtype=torch.complex128), torch.zeros(2, 8))

    @pytest.mark.parametrize("build", ALL_LAYERS)
    def test_rejects_inputs_of_the_wrong_width_for_the_layer(self, build) -> None:
        """The layer's own width check applies, as it does in forward."""
        layer = build()
        width = _inputs_for(layer).shape[1]
        too_narrow = torch.rand(4, width - 1, dtype=torch.float64)
        with pytest.raises(ValueError, match="does not match"):
            layer(too_narrow)
        with pytest.raises(ValueError, match="does not match"):
            quantum_kernel_matrix(too_narrow, layer)
        with pytest.raises(ValueError, match="does not match"):
            encoded_states(too_narrow, layer)
        with pytest.raises(ValueError, match="does not match"):
            quantum_kernel_matrix(_inputs_for(layer), layer, Y=too_narrow)

    def test_validates_y_before_simulating_x(self, monkeypatch) -> None:
        calls = []
        real = kernels._simulate
        monkeypatch.setattr(kernels, "_simulate", lambda *a: calls.append(1) or real(*a))
        with pytest.raises(ValueError, match="does not match"):
            quantum_kernel_matrix(_angles(2), _angle_layer(), Y=torch.zeros(2, N_QUBITS + 1))
        assert calls == []

    @pytest.mark.parametrize("build", ALL_LAYERS)
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_inputs(self, build, bad: float) -> None:
        layer = build()
        X = _inputs_for(layer)
        X[1, 0] = bad
        with pytest.raises(ValueError, match="NaN or ±inf"):
            quantum_kernel_matrix(X, layer)
        with pytest.raises(ValueError, match="NaN or ±inf"):
            quantum_kernel_matrix(_inputs_for(layer), layer, Y=X)

    @pytest.mark.parametrize("build", ALL_LAYERS)
    def test_forward_rejects_the_same_non_finite_inputs(self, build) -> None:
        layer = build()
        X = _inputs_for(layer)
        X[1, 0] = math.nan
        with pytest.raises(ValueError, match="NaN or ±inf"):
            layer(X)

    def test_refuses_a_foreign_transform_on_the_qnode(self) -> None:
        layer = _angle_layer()
        layer.qlayer.qnode = qml.transforms.cancel_inverses(layer.qlayer.qnode)
        with pytest.raises(RuntimeError, match="cancel_inverses"):
            quantum_kernel_matrix(_angles(3), layer)

    def test_allows_the_encoders_own_broadcast_expand(self) -> None:
        """Non-backprop layers carry broadcast_expand; the kernel is unchanged."""
        X = _angles(M)
        torch.manual_seed(0)
        layer = QuantumEncodingLayer(
            n_qubits=N_QUBITS, n_layers=2, device_name="default.qubit", diff_method="adjoint"
        )
        assert len(layer.qlayer.qnode.compile_pipeline) == 1
        torch.testing.assert_close(
            quantum_kernel_matrix(X, layer),
            quantum_kernel_matrix(X, _angle_layer()),
            atol=1e-12,
            rtol=0,
        )

    def test_refuses_to_run_inside_the_noise_block(self) -> None:
        from hqnn_forge.noise import apply_depolarizing_noise

        layer = _angle_layer()
        X = _angles(3)
        with apply_depolarizing_noise(layer, 0.1):
            with pytest.raises(RuntimeError, match="apply_depolarizing_noise"):
                quantum_kernel_matrix(X, layer)
            with pytest.raises(RuntimeError, match="apply_depolarizing_noise"):
                encoded_states(X, layer)
        # p = 0 replaces nothing, and the layer is usable again after the block.
        with apply_depolarizing_noise(layer, 0.0):
            quantum_kernel_matrix(X, layer)
        quantum_kernel_matrix(X, layer)

    def test_promotes_lower_precision_and_real_states(self) -> None:
        S = encoded_states(_angles(M), _angle_layer())
        expected = kernel_from_states(S)
        S64 = S.to(torch.complex64)
        torch.testing.assert_close(kernel_from_states(S64, S), expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(kernel_from_states(S64), expected, atol=1e-6, rtol=0)
        real = S.real
        assert kernel_from_states(real, S).dtype == torch.float64
        torch.testing.assert_close(
            kernel_from_states(real, real), (real @ real.T).pow(2), atol=1e-12, rtol=0
        )

    def test_simulates_x_and_y_in_one_replay(self, monkeypatch) -> None:
        calls = []
        real = kernels._simulate
        monkeypatch.setattr(kernels, "_simulate", lambda *a: calls.append(1) or real(*a))
        quantum_kernel_matrix(_angles(3, seed=0), _angle_layer(), Y=_angles(4, seed=1))
        assert calls == [1]
