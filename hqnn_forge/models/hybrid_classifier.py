"""
hqnn_forge.models.hybrid_classifier
======================================
Full Hybrid Quantum-Classical Binary Classifier.

Architecture
------------
::

    Input (batch, n_input_features)
         │
         ▼
    [Classical encoder]  nn.Linear(n_input_features → n_qubits) + Tanh
         │
         ▼
    [Quantum encoding]   QuantumEncodingLayer(n_qubits, n_layers)
         │                  ├─ AngleEmbedding (RX)
         │                  └─ Strongly-entangling VQC (CNOT ring + Rot)
         │                  → ⟨Z_i⟩, shape (batch, n_qubits)
         ▼
    [Classical head]     nn.Linear(n_qubits → 1)
         │
         ▼
    Raw logit (batch, 1)   ← apply sigmoid for probability

Design Notes
------------
* The classical encoder projects arbitrary-width input to ``n_qubits`` dims
  and applies ``tanh`` to soft-clip values into (-1, 1), which ``forward`` then
  scales by π into (-π, π).  If ``PCANormalizer`` with ``scale_to_pi=True`` is
  used upstream, set ``use_classical_encoder=False`` to skip both steps: bypassed
  input reaches the circuit unscaled, so it must already lie in (-π, π).

* The quantum layer is initialised with ``restricted_normal_init_`` immediately
  after construction to avoid barren plateaus.

* The model exposes ``predict_proba(x)`` for inference (applies sigmoid).

Parameters
----------
n_input_features:
    Dimensionality of the raw / PCA-reduced input.
n_qubits:
    Number of qubits.  Must equal the output size of the classical encoder.
n_layers:
    Number of variational layers in the quantum circuit.
use_classical_encoder:
    If ``True`` (default), prepend a ``Linear + Tanh`` to project input to
    ``n_qubits`` dims.  Set ``False`` if input is already n_qubits-dim and
    already in (-π, π); it is then passed to the circuit unscaled.
device_name:
    PennyLane device.
diff_method:
    Gradient computation method.
init_strategy:
    ``"restricted"`` (default) — global restricted-normal init.
    ``"block_local"``           — per-layer decreasing variance.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import DeviceName, DiffMethod, QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.initializers.restricted_variance import (
    restricted_normal_init_,
    block_local_init_,
)
from hqnn_forge.models.base import BinaryClassifierBase


class HybridBinaryClassifier(BinaryClassifierBase):
    """
    Hybrid quantum-classical binary classifier.

    See module docstring for architecture overview.

    Parameters
    ----------
    n_input_features:
        Number of raw (or PCA-reduced) input features.  Default: 8.
    n_qubits:
        Number of qubits in the quantum encoding layer.  Default: 8.
    n_layers:
        VQC ansatz layers.  Default: 2.
    use_classical_encoder:
        Prepend ``Linear(n_input_features → n_qubits) + Tanh``.  Default: True.
        If ``False``, input must already lie in (-π, π); it is not rescaled.
    dropout_p:
        Dropout probability applied after the quantum layer.  Default: 0.0.
    device_name:
        PennyLane device string, type-checked as ``"lightning.qubit"`` or
        ``"default.qubit"``.  Default: ``"lightning.qubit"``.  Any other device name
        still runs — it is handed to ``qml.device``, which falls back to
        ``"default.qubit"`` with a warning if the device cannot be initialised.
    diff_method:
        Gradient method: ``"adjoint"``, ``"parameter-shift"``, ``"backprop"`` or
        ``"finite-diff"``.  Default: ``"adjoint"``.
    init_strategy:
        ``"restricted"`` or ``"block_local"``.  Default: ``"restricted"``.
    encoding_type:
        Type of quantum embedding to use: ``"angle"`` or ``"iqp"``. Default: ``"angle"``.
    noise_level:
        Training-time depolarizing probability for the quantum layer, in
        ``[0, 0.75]``.  Default: ``0.0`` (noiseless).  Applied in train mode
        only; see :mod:`hqnn_forge.noise`.
    noise_position:
        ``"all"`` (default) or ``"end"``; where the channel is inserted.

    Attributes
    ----------
    classical_encoder : nn.Sequential or nn.Identity
    quantum_layer     : QuantumEncodingLayer
    dropout           : nn.Dropout
    head              : nn.Linear

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.models import HybridBinaryClassifier
    >>> model = HybridBinaryClassifier(n_input_features=30, n_qubits=8, n_layers=2)
    >>> x = torch.randn(4, 30)
    >>> logits = model(x)           # shape (4, 1)
    >>> probs  = model.predict_proba(x)  # shape (4,), values in [0, 1]
    """

    def __init__(
        self,
        n_input_features: int = 8,
        n_qubits: int = 8,
        n_layers: int = 2,
        *,
        use_classical_encoder: bool = True,
        dropout_p: float = 0.0,
        device_name: DeviceName = "lightning.qubit",
        diff_method: DiffMethod = "adjoint",
        init_strategy: str = "restricted",
        encoding_type: str = "angle",
        noise_level: float = 0.0,
        noise_position: str = "all",
    ) -> None:
        super().__init__()
        self._config = dict(
            n_input_features=n_input_features,
            n_qubits=n_qubits,
            n_layers=n_layers,
            use_classical_encoder=use_classical_encoder,
            dropout_p=dropout_p,
            device_name=device_name,
            diff_method=diff_method,
            init_strategy=init_strategy,
            encoding_type=encoding_type,
            noise_level=noise_level,
            noise_position=noise_position,
        )

        self.n_input_features = n_input_features
        self.n_qubits         = n_qubits
        self.n_layers         = n_layers
        self.init_strategy    = init_strategy
        self.use_classical_encoder = use_classical_encoder

        # ── Classical encoder ─────────────────────────────────────────────
        if use_classical_encoder:
            self.classical_encoder: nn.Module = nn.Sequential(
                nn.Linear(n_input_features, n_qubits),
                nn.Tanh(),
            )
        else:
            if n_input_features != n_qubits:
                raise ValueError(
                    f"When use_classical_encoder=False, n_input_features "
                    f"({n_input_features}) must equal n_qubits ({n_qubits})."
                )
            self.classical_encoder = nn.Identity()

        # ── Quantum encoding layer ────────────────────────────────────────
        if encoding_type == "angle":
            self.quantum_layer: QuantumEncodingLayer | IQPEncodingLayer = QuantumEncodingLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                device_name=device_name,
                diff_method=diff_method,
                noise_level=noise_level,
                noise_position=noise_position,
            )
        elif encoding_type == "iqp":
            self.quantum_layer = IQPEncodingLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                n_repeats=1,
                device_name=device_name,
                diff_method=diff_method,
                noise_level=noise_level,
                noise_position=noise_position,
            )
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")

        # ── Dropout ───────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

        # ── Classical head ────────────────────────────────────────────────
        self.head = nn.Linear(n_qubits, 1)

        # ── Small-angle restricted-variance initialisation ─────────────────
        self._initialise_weights()

    # ------------------------------------------------------------------
    def _initialise_weights(self) -> None:
        """Apply restricted-variance init to quantum weights; Xavier to classical."""
        # Classical encoder: Xavier uniform (standard for linear + Tanh)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Quantum weights: small-angle restricted-variance init
        weights = self.quantum_layer.qlayer.weights  # shape (n_layers, n_qubits, 3)
        if self.init_strategy == "block_local":
            block_local_init_(weights.data, n_qubits=self.n_qubits)
        else:
            restricted_normal_init_(
                weights.data, n_qubits=self.n_qubits, n_layers=self.n_layers
            )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classical encode → quantum encode → classification head.

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.

        Returns
        -------
        torch.Tensor
            Raw logits, shape ``(batch_size, 1)``.  Apply ``torch.sigmoid``
            for probabilities, or pass directly to ``FocalLoss``.
        """
        # Classical projection + activation
        x = self.classical_encoder(x)        # (B, n_qubits)

        # Tanh output (-1, 1) → (-π, π).  Bypassed input is already in (-π, π);
        # scaling it again would alias angles mod 2π.
        if self.use_classical_encoder:
            x = x * torch.pi

        # Quantum feature map
        x = self.quantum_layer(x)            # (B, n_qubits), values ∈ [-1, 1]

        # Regularisation
        x = self.dropout(x)

        # Classification head
        return self.head(x)                  # (B, 1)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"n_input_features={self.n_input_features}, "
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"total_params={self.count_parameters()}"
        )
