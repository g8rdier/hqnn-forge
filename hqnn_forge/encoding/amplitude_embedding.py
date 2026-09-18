"""
hqnn_forge.encoding.amplitude_embedding
=======================================
Quantum encoding module projecting classical tabular feature vectors into an
n-qubit Hilbert space via amplitude embedding, followed by the same
strongly-entangled variational ansatz the angle and IQP encoders use.

Design Rationale
----------------
* **Amplitude Embedding** writes the (zero-padded, L2-normalised) feature
  vector directly into the probability amplitudes of the state:
  ``|ψ(x)⟩ = Σ_k x_k |k⟩ / ‖x‖``.  ``n_qubits`` qubits therefore take up to
  ``2**n_qubits`` features, where angle and IQP embedding take ``n_qubits``.
  It is the qubit-efficient choice: 8 qubits carry 256 features instead of 8.

* **State-preparation cost** is the price of that efficiency.  An arbitrary
  amplitude vector needs a state-preparation circuit that is exponential in
  ``n_qubits`` on real hardware (O(2^n) gates; Möttönen et al. 2005), and on
  a state-vector simulator the embedding is one O(2^n) vector write instead
  of ``n`` rotations.  Angle and IQP embedding cost O(n) and O(n²) gates
  respectively.  Amplitude embedding is the right tool when the number of
  features, not the gate budget, is the constraint.

* **Normalisation removes one degree of freedom.**  Because only the
  direction of ``x`` survives, ``x`` and ``c·x`` encode the same state for any
  ``c > 0``, and an all-zero vector has no valid state at all (``forward``
  raises for it).  Feature scaling upstream matters less than for angle
  embedding, where the absolute value of each feature is a rotation angle.

* **Differentiation.**  Under ``backprop`` on ``default.qubit`` gradients
  flow to both the weights and the inputs, which is what a classical encoder
  upstream needs.  The ``adjoint``, ``parameter-shift`` and ``finite-diff``
  methods differentiate gate parameters only: they give weight gradients,
  and PennyLane returns **NaN** (not zero, and without raising) for the
  gradient with respect to the prepared state.  ``forward`` therefore raises
  when the inputs require a gradient under any method other than
  ``backprop`` (see :class:`AmplitudeEncodingLayer`).

* **Strongly-Entangling Ansatz** and **Measurement** are identical to
  :mod:`hqnn_forge.encoding.angle_embedding`: L layers of a CNOT ring
  followed by per-qubit ``Rot``, then ⟨Z_i⟩ on every qubit.

References
----------
* Möttönen et al. (2005) "Transformation of quantum states using uniformly
  controlled rotations", Quantum Inf. Comput. 5, 467.
* Schuld & Petruccione (2018) "Supervised Learning with Quantum Computers",
  Springer, §5.2 (amplitude encoding).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pennylane as qml
import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import (
    DeviceName,
    DiffMethod,
    _expand_batch_dimension,
    _resolve_device,
)

logger = logging.getLogger(__name__)

#: Samples whose L2 norm is at or below this cannot be normalised to a state.
ZERO_NORM_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Raw QNode function
# ---------------------------------------------------------------------------


def _make_amplitude_embedding_circuit(
    n_qubits: int,
    n_layers: int,
) -> Callable[[torch.Tensor, torch.Tensor], list[qml.measurements.ExpectationMP]]:
    """
    Factory returning the bare quantum function for the amplitude feature map.

    The returned function has the signature::

        circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[float]

    where ``inputs`` has shape ``(2**n_qubits,)`` (or ``(batch, 2**n_qubits)``
    when broadcasted) and is already zero-padded and L2-normalised, and
    ``weights`` has shape ``(n_layers, n_qubits, 3)``.

    Circuit structure
    -----------------
    1. ``AmplitudeEmbedding(inputs)``: prepares ``Σ_k inputs_k |k⟩``.
       ``normalize=True`` re-normalises inside the template so a float32
       vector that is normalised to 1e-7 passes the template's exact check;
       the operation is a no-op up to rounding on the already-normalised
       input ``forward`` produces.
    2. Per layer: CNOT ring ``CNOT(i → i+1 mod n)``, then ``Rot`` on each qubit.
    3. ``[⟨Z_i⟩ for i in range(n_qubits)]``.
    """

    def circuit(
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> list[qml.measurements.ExpectationMP]:
        # ── 1. Amplitude embedding ───────────────────────────────────────
        qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)

        # ── 2. Strongly entangling layers (same as angle_embedding) ──────
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
            for qubit in range(n_qubits):
                qml.Rot(
                    weights[layer, qubit, 0],
                    weights[layer, qubit, 1],
                    weights[layer, qubit, 2],
                    wires=qubit,
                )

        # ── 3. Measurement ───────────────────────────────────────────────
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ---------------------------------------------------------------------------
# Public QNode factory
# ---------------------------------------------------------------------------


def build_amplitude_qnode(
    n_qubits: int = 8,
    n_layers: int = 2,
    device_name: DeviceName = "lightning.qubit",
    diff_method: DiffMethod = "adjoint",
) -> qml.QNode:
    """
    Build and return a PennyLane QNode for the amplitude feature map.

    The QNode expects ``inputs`` of shape ``(2**n_qubits,)`` or
    ``(batch, 2**n_qubits)``; use :class:`AmplitudeEncodingLayer` for the
    padding and normalisation of raw feature vectors.

    Parameters
    ----------
    n_qubits:
        Number of qubits.  The state has ``2**n_qubits`` amplitudes.
    n_layers:
        Number of entangling + rotation layers in the VQC ansatz.
    device_name, diff_method:
        As for :func:`hqnn_forge.encoding.build_encoding_qnode`.

    Raises
    ------
    ValueError
        If ``n_qubits < 2``.
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be ≥ 2 for the CNOT entangling ring; got {n_qubits}.")

    device = _resolve_device(device_name, n_qubits)
    circuit_fn = _make_amplitude_embedding_circuit(n_qubits, n_layers)

    qnode = qml.QNode(
        func=circuit_fn,
        device=device,
        diff_method=diff_method,
        interface="torch",
    )
    qnode = _expand_batch_dimension(qnode, diff_method)

    logger.info(
        "Amplitude QNode built | device=%s | qubits=%d | layers=%d | diff=%s",
        device.name,
        n_qubits,
        n_layers,
        diff_method,
    )
    return qnode


# ---------------------------------------------------------------------------
# PyTorch nn.Module wrapper
# ---------------------------------------------------------------------------


class AmplitudeEncodingLayer(nn.Module):
    """
    A PyTorch ``nn.Module`` wrapping the amplitude-embedding QNode.

    API-consistent with :class:`~hqnn_forge.encoding.QuantumEncodingLayer` and
    :class:`~hqnn_forge.encoding.iqp_embedding.IQPEncodingLayer`: same
    constructor shape, same ``qlayer.weights`` parameter of shape
    ``(n_layers, n_qubits, 3)``, same ``(batch, n_qubits)`` output of ⟨Z_i⟩.
    The difference is the input width: up to ``2**n_qubits`` features per
    sample instead of exactly ``n_qubits``.

    Input handling in ``forward``
    -----------------------------
    1. The last dimension must equal ``n_features``.
    2. If ``n_features < 2**n_qubits`` the vector is zero-padded on the right
       to ``2**n_qubits`` amplitudes.
    3. The padded vector is L2-normalised.  A sample whose norm is at or
       below ``ZERO_NORM_TOLERANCE`` raises ``ValueError``: the zero vector
       has no direction, so there is no state to prepare.

    Both steps are differentiable, so with ``diff_method="backprop"`` the
    gradient reaches whatever produced the features.

    Cost versus the other encoders
    ------------------------------
    Angle embedding uses ``n`` single-qubit rotations for ``n`` features; IQP
    uses O(n²) gates for ``n`` features; amplitude embedding takes up to
    ``2**n`` features but needs an O(2^n)-gate state-preparation circuit on
    hardware (Möttönen et al. 2005).  On a simulator the difference is one
    vector write against ``n`` rotations, so the wall-clock gap is small; on
    hardware it is exponential.  Choose amplitude embedding when the feature
    count is the constraint, not when the gate budget is.

    Differentiation methods
    -----------------------
    ``"backprop"`` on ``default.qubit`` differentiates through the state
    preparation, so gradients reach both the weights and the inputs.  The
    other methods only handle gate parameters: they give weight gradients,
    but the input gradient PennyLane returns for the prepared state is NaN,
    silently.  ``forward`` raises ``RuntimeError`` instead when ``x``
    requires a gradient under such a method.  Use them for a fixed feature
    map (kernels, frozen inputs), and ``backprop`` when a classical encoder
    sits upstream.

    Parameters
    ----------
    n_qubits:
        Number of qubits.  Default: 8 (256 amplitudes).
    n_layers:
        Entangling + rotation blocks in the VQC ansatz.  Default: 2.
    n_features:
        Width of the input vectors, ``1 ≤ n_features ≤ 2**n_qubits``.
        Default: ``2**n_qubits`` (no padding).
    device_name:
        PennyLane device.  Falls back to ``default.qubit`` if
        ``pennylane-lightning`` is unavailable.
    diff_method:
        Gradient method.  See *Differentiation methods* above.

    Attributes
    ----------
    n_qubits : int
    n_layers : int
    n_features : int
    n_amplitudes : int
        ``2**n_qubits``.
    qlayer : pennylane.qnn.TorchLayer

    Methods
    -------
    prepare_inputs(x)
        The padding and normalisation ``forward`` applies before the QNode.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.encoding import AmplitudeEncodingLayer
    >>> layer = AmplitudeEncodingLayer(
    ...     n_qubits=3, n_layers=1, n_features=6,
    ...     device_name="default.qubit", diff_method="backprop",
    ... )
    >>> x = torch.randn(4, 6)   # 6 features → padded to 8 amplitudes
    >>> layer(x).shape
    torch.Size([4, 3])
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_features: int | None = None,
        device_name: DeviceName = "lightning.qubit",
        diff_method: DiffMethod = "adjoint",
    ) -> None:
        super().__init__()

        n_amplitudes = 2**n_qubits
        if n_features is None:
            n_features = n_amplitudes
        if not 1 <= n_features <= n_amplitudes:
            raise ValueError(
                f"n_features must lie in [1, 2**n_qubits] = [1, {n_amplitudes}]; got {n_features}."
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.n_amplitudes = n_amplitudes
        self.diff_method = diff_method

        qnode = build_amplitude_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device_name=device_name,
            diff_method=diff_method,
        )

        weight_shapes: dict[str, tuple[int, ...]] = {
            "weights": (n_layers, n_qubits, 3),
        }
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    # ------------------------------------------------------------------
    def prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zero-pad ``x`` to ``2**n_qubits`` and L2-normalise each sample.

        This is the classical step ``forward`` applies before the QNode; it is
        public so that tools which replay the circuit (``hqnn_forge.kernels``)
        can feed the QNode the same amplitudes ``forward`` would.
        """
        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"Input feature dimension {x.shape[-1]} does not match "
                f"n_features={self.n_features} (at most 2**n_qubits={self.n_amplitudes} "
                f"features fit in {self.n_qubits} qubits)."
            )
        norms = torch.linalg.vector_norm(x, dim=-1)
        if bool((norms <= ZERO_NORM_TOLERANCE).any()):
            raise ValueError(
                "Amplitude embedding cannot encode an all-zero feature vector: "
                "it has no direction, so there is no state to prepare.  Every "
                f"sample must have an L2 norm above {ZERO_NORM_TOLERANCE}."
            )
        pad = self.n_amplitudes - self.n_features
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        return x / norms.unsqueeze(-1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of feature vectors into Pauli-Z expectation values.

        Parameters
        ----------
        x : torch.Tensor
            Classical input tensor of shape ``(batch_size, n_features)``.
            Any real scale; only the direction is encoded.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, n_qubits)``, each element ∈ [-1, 1].

        Raises
        ------
        ValueError
            If the last dimension of ``x`` is not ``n_features``, or a sample
            is all zeros.
        RuntimeError
            If ``x`` requires a gradient and ``diff_method`` is not
            ``"backprop"``: the gradient through the state preparation would
            be NaN under the other methods, so it is refused up front.
        """
        if x.requires_grad and self.diff_method != "backprop":
            raise RuntimeError(
                f"AmplitudeEncodingLayer cannot differentiate with respect to its "
                f"inputs under diff_method={self.diff_method!r}: PennyLane returns NaN "
                f"for the gradient of a prepared state under gate-parameter methods.  "
                f"Use diff_method='backprop' on default.qubit when the inputs need a "
                f"gradient (for example with a classical encoder upstream), or detach "
                f"the inputs."
            )
        amplitudes = self.prepare_inputs(x)
        # Whole batch in one call; see QuantumEncodingLayer.forward.
        return self.qlayer(amplitudes)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_features={self.n_features}, "
            f"n_params={self.n_layers * self.n_qubits * 3}"
        )
