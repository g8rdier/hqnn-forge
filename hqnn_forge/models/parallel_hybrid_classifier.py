"""
hqnn_forge.models.parallel_hybrid_classifier
==============================================
Parallel-topology Hybrid Quantum-Classical Binary Classifier.

Architecture
------------
::

    Input (batch, n_input_features)
         │
         ├─────────────────────────────┬──────────────────────────────┐
         ▼                             ▼
    [Classical branch]            [Quantum branch]
    Linear → ReLU →               [Classical encoder] Linear(→n_qubits) + Tanh
    Linear → ReLU                      │
    → (batch, classical_hidden_dim)    ▼
                                   QuantumEncodingLayer(n_qubits, n_layers)
                                        │
                                        ▼
                                   ⟨Z_i⟩, (batch, n_qubits)
         │                             │
         └──────────────┬──────────────┘
                         ▼
                   [Concatenate]  (batch, classical_hidden_dim + n_qubits)
                         │
                         ▼
                   [Dropout]
                         │
                         ▼
                   [Classical head]  Linear(→ 1)
                         │
                         ▼
                   Raw logit (batch, 1)   ← apply sigmoid for probability

Design Notes
------------
* The two branches process the *same* raw input independently and are fused
  by concatenation before the final classification head.  This is the
  standard architecture for testing whether added classical capacity can
  substitute for, or extend, what the quantum layer contributes — compare
  ``ParallelHybridClassifier.count_parameters()`` against an equivalently
  configured ``HybridBinaryClassifier`` to quantify the trade-off.

* The quantum branch mirrors ``HybridBinaryClassifier`` in *topology* (same
  classical encoder + ``QuantumEncodingLayer`` / ``IQPEncodingLayer`` choice,
  same barren-plateau-safe initialisation scheme).  Note that seeding the two
  architectures identically does **not** give them identical quantum weights:
  this model builds more classical layers before the quantum init runs, so it
  draws from a different RNG state.  To compare the two topologies fairly,
  copy the quantum weights across after construction — see
  ``examples/quick_start.py``.

* The classical branch is a small two-layer MLP (``classical_hidden_dim``
  units) with ReLU activations.

Parameters
----------
n_input_features:
    Dimensionality of the raw / PCA-reduced input.
n_qubits:
    Number of qubits in the quantum branch.
n_layers:
    Number of variational layers in the quantum circuit.
classical_hidden_dim:
    Width of the classical MLP branch.
use_classical_encoder:
    If ``True`` (default), prepend a ``Linear + Tanh`` to project input to
    ``n_qubits`` dims for the quantum branch.  If ``False``, input must already
    lie in (-π, π) (e.g. ``PCANormalizer(scale_to_pi=True)``); the quantum
    branch then passes it to the circuit unscaled.
device_name:
    PennyLane device.
diff_method:
    Gradient computation method.
init_strategy:
    ``"restricted"`` (default) — global restricted-normal init.
    ``"block_local"``           — per-layer decreasing variance.
encoding_type:
    Type of quantum embedding to use: ``"angle"`` or ``"iqp"``. Default: ``"angle"``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.initializers.restricted_variance import (
    restricted_normal_init_,
    block_local_init_,
)
from hqnn_forge.utils.modes import eval_mode


class ParallelHybridClassifier(nn.Module):
    """
    Parallel-topology hybrid quantum-classical binary classifier.

    See module docstring for architecture overview.  API-consistent with
    ``HybridBinaryClassifier`` — same ``encoding_type`` / ``init_strategy``
    options and ``predict_proba`` / ``predict`` / ``count_parameters``
    interface.

    Parameters
    ----------
    n_input_features:
        Number of raw (or PCA-reduced) input features.  Default: 8.
    n_qubits:
        Number of qubits in the quantum branch.  Default: 8.
    n_layers:
        VQC ansatz layers.  Default: 2.
    classical_hidden_dim:
        Width of the classical MLP branch.  Default: 16.
    use_classical_encoder:
        Prepend ``Linear(n_input_features → n_qubits) + Tanh`` to the quantum
        branch.  Default: True.  If ``False``, input must already lie in
        (-π, π); it is not rescaled.
    dropout_p:
        Dropout probability applied to the fused branch outputs.  Default: 0.0.
    device_name:
        PennyLane device string.  Default: ``"lightning.qubit"``.
    diff_method:
        Gradient method.  Default: ``"adjoint"``.
    init_strategy:
        ``"restricted"`` or ``"block_local"``.  Default: ``"restricted"``.
    encoding_type:
        Type of quantum embedding to use: ``"angle"`` or ``"iqp"``. Default: ``"angle"``.

    Attributes
    ----------
    classical_branch  : nn.Sequential
    classical_encoder : nn.Sequential or nn.Identity
    quantum_layer     : QuantumEncodingLayer or IQPEncodingLayer
    dropout           : nn.Dropout
    head              : nn.Linear

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.models import ParallelHybridClassifier
    >>> model = ParallelHybridClassifier(n_input_features=30, n_qubits=8, n_layers=2)
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
        classical_hidden_dim: int = 16,
        use_classical_encoder: bool = True,
        dropout_p: float = 0.0,
        device_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
        init_strategy: str = "restricted",
        encoding_type: str = "angle",
    ) -> None:
        super().__init__()

        self.n_input_features     = n_input_features
        self.n_qubits             = n_qubits
        self.n_layers             = n_layers
        self.classical_hidden_dim = classical_hidden_dim
        self.init_strategy        = init_strategy
        self.use_classical_encoder = use_classical_encoder

        # ── Classical branch (MLP) ────────────────────────────────────────
        self.classical_branch = nn.Sequential(
            nn.Linear(n_input_features, classical_hidden_dim),
            nn.ReLU(),
            nn.Linear(classical_hidden_dim, classical_hidden_dim),
            nn.ReLU(),
        )

        # ── Quantum branch: classical encoder ─────────────────────────────
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

        # ── Quantum branch: quantum encoding layer ────────────────────────
        if encoding_type == "angle":
            self.quantum_layer = QuantumEncodingLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                device_name=device_name,
                diff_method=diff_method,
            )
        elif encoding_type == "iqp":
            self.quantum_layer = IQPEncodingLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                n_repeats=1,
                device_name=device_name,
                diff_method=diff_method,
            )
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")

        # ── Fusion + regularisation ────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

        # ── Classical head ────────────────────────────────────────────────
        self.head = nn.Linear(classical_hidden_dim + n_qubits, 1)

        # ── Barren-plateau-safe initialisation ────────────────────────────
        self._initialise_weights()

    # ------------------------------------------------------------------
    def _initialise_weights(self) -> None:
        """
        Initialise each block with the scheme derived for its non-linearity:
        He for the ReLU branch, Xavier for the Tanh encoder and linear head,
        restricted-variance for the quantum weights.
        """
        # Classical MLP branch: He/Kaiming — derived for ReLU, which Xavier
        # under-scales by sqrt(2) per layer.
        for module in self.classical_branch.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Tanh encoder and linear head: Xavier uniform.
        for module in (*self.classical_encoder.modules(), self.head):
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
        Forward pass: classical branch and quantum branch process ``x``
        independently, are concatenated, and fed through the classification
        head.

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
        # Classical branch
        classical_out = self.classical_branch(x)   # (B, classical_hidden_dim)

        # Quantum branch: classical projection + activation
        q = self.classical_encoder(x)               # (B, n_qubits)
        # Tanh output (-1, 1) → (-π, π).  Bypassed input is already in (-π, π);
        # scaling it again would alias angles mod 2π.
        if self.use_classical_encoder:
            q = q * torch.pi
        quantum_out = self.quantum_layer(q)          # (B, n_qubits), values ∈ [-1, 1]

        # Fuse branches
        fused = torch.cat([classical_out, quantum_out], dim=-1)
        fused = self.dropout(fused)

        # Classification head
        return self.head(fused)                     # (B, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute positive-class probabilities (inference mode, no gradients).

        Runs in eval mode whatever mode the model is in, so dropout is off and
        repeated calls on the same input agree.  Every submodule's ``training``
        flag is restored afterwards, so calling this mid-training leaves the
        model exactly as it was.

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.

        Returns
        -------
        torch.Tensor
            Probability of class 1, shape ``(batch_size,)``, values ∈ [0, 1].
        """
        # no_grad alone leaves nn.Dropout active: it checks self.training, not
        # grad mode.
        with eval_mode(self):
            logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary labels.  Runs in eval mode, like ``predict_proba``.

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
        """Return total parameter count (quantum + classical, both branches)."""
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
            f"classical_hidden_dim={self.classical_hidden_dim}, "
            f"total_params={self.count_parameters()}"
        )
