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
  and applies ``tanh`` to soft-clip values into (-1, 1), which are then
  implicitly rescaled by the quantum layer's angle embedding.  If ``PCANormalizer``
  with ``scale_to_pi=True`` is used upstream, you can set ``use_classical_encoder=False``
  to skip this step.

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
    ``n_qubits`` dims.  Set ``False`` if input is already n_qubits-dim.
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

from hqnn_forge.encoding.angle_embedding import QuantumEncodingLayer
from hqnn_forge.initializers.restricted_variance import (
    apply_restricted_init,
    restricted_normal_init_,
    block_local_init_,
)


class HybridBinaryClassifier(nn.Module):
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
    dropout_p:
        Dropout probability applied after the quantum layer.  Default: 0.0.
    device_name:
        PennyLane device string.  Default: ``"lightning.qubit"``.
    diff_method:
        Gradient method.  Default: ``"adjoint"``.
    init_strategy:
        ``"restricted"`` or ``"block_local"``.  Default: ``"restricted"``.

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
        device_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
        init_strategy: str = "restricted",
    ) -> None:
        super().__init__()

        self.n_input_features = n_input_features
        self.n_qubits         = n_qubits
        self.n_layers         = n_layers
        self.init_strategy    = init_strategy

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
        self.quantum_layer = QuantumEncodingLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device_name=device_name,
            diff_method=diff_method,
        )

        # ── Dropout ───────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

        # ── Classical head ────────────────────────────────────────────────
        self.head = nn.Linear(n_qubits, 1)

        # ── Barren-plateau-safe initialisation ────────────────────────────
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

        # Quantum weights: barren-plateau-safe init
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
        x = self.classical_encoder(x)        # (B, n_qubits), values ∈ (-1, 1)

        # Scale into (-π, π) for angle embedding
        x = x * torch.pi                     # (B, n_qubits)

        # Quantum feature map
        x = self.quantum_layer(x)            # (B, n_qubits), values ∈ [-1, 1]

        # Regularisation
        x = self.dropout(x)

        # Classification head
        return self.head(x)                  # (B, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute positive-class probabilities (inference mode, no gradients).

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.

        Returns
        -------
        torch.Tensor
            Probability of class 1, shape ``(batch_size,)``, values ∈ [0, 1].
        """
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary labels.

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.
        threshold:
            Decision threshold.  Default: 0.5.
            For imbalanced datasets consider tuning via ROC/PR curves.

        Returns
        -------
        torch.Tensor
            Binary label tensor of shape ``(batch_size,)``, dtype ``torch.long``.
        """
        return (self.predict_proba(x) >= threshold).long()

    # ------------------------------------------------------------------
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total parameter count (quantum + classical)."""
        params = (
            self.parameters() if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"n_input_features={self.n_input_features}, "
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"total_params={self.count_parameters()}"
        )
