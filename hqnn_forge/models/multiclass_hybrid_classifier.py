"""
hqnn_forge.models.multiclass_hybrid_classifier
==============================================
Hybrid quantum-classical classifier for ``n_classes ≥ 2`` classes.

Architecture
------------
::

    Input (batch, n_input_features)
         │
         ▼
    [Classical encoder]  nn.Linear(n_input_features → n_qubits) + Tanh, · π
         │
         ▼
    [Quantum encoding]   QuantumEncodingLayer / IQPEncodingLayer (n_qubits, n_layers)
         │                  → ⟨Z_i⟩, shape (batch, n_qubits)
         ▼
    [Dropout]
         │
         ▼
    [Class heads]        nn.Linear(n_qubits → n_classes)
         │
         ▼
    Raw logits (batch, n_classes)

Design Notes
------------
* The quantum layer is shared; the class heads are the rows of one
  ``nn.Linear(n_qubits, n_classes)``, each a binary head reading the same
  ``n_qubits`` expectation values.  This is the "ensemble of binary heads
  sharing the quantum layer" option: the quantum parameter count does not
  grow with ``n_classes``, only the head does (``n_qubits + 1`` per class).

* ``strategy`` selects how the ``n_classes`` logits are turned into
  probabilities and, by implication, which loss to train with:

  - ``"softmax"`` (default): ``predict_proba`` is the softmax over classes.
    Train with ``nn.CrossEntropyLoss`` on integer labels.
  - ``"one_vs_rest"``: each head is an independent binary classifier
    (class ``c`` against the rest); ``predict_proba`` applies a sigmoid to
    each logit and normalises the ``n_classes`` scores to sum to one, as
    ``sklearn.multiclass.OneVsRestClassifier`` does.  Train with
    ``nn.BCEWithLogitsLoss`` on one-hot targets (see :meth:`one_hot`).

  ``forward`` returns raw logits in both cases, and ``predict`` is the
  argmax in both cases; only ``predict_proba`` differs.

* With ``n_classes=2`` the softmax model is a two-logit form of the binary
  classifier.  :class:`~hqnn_forge.models.HybridBinaryClassifier` with its
  single logit is the smaller model for that case and works with
  :class:`~hqnn_forge.utils.FocalLoss`; this class exists for ``n_classes >
  2``.

* Initialisation, encoder bypass and encoding options mirror
  :class:`~hqnn_forge.models.HybridBinaryClassifier`.

Parameters
----------
n_input_features:
    Dimensionality of the raw / PCA-reduced input.
n_qubits:
    Number of qubits.
n_layers:
    Number of variational layers in the quantum circuit.
n_classes:
    Number of classes, ``≥ 2``.
strategy:
    ``"softmax"`` or ``"one_vs_rest"``; see above.
use_classical_encoder, dropout_p, device_name, diff_method, init_strategy,
encoding_type:
    As for :class:`~hqnn_forge.models.HybridBinaryClassifier`.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.initializers.restricted_variance import (
    block_local_init_,
    restricted_normal_init_,
)
from hqnn_forge.utils.modes import eval_mode

MulticlassStrategy = Literal["softmax", "one_vs_rest"]


class MulticlassHybridClassifier(nn.Module):
    """
    Hybrid quantum-classical multiclass classifier.

    See the module docstring for the architecture and the two strategies.

    Parameters
    ----------
    n_input_features:
        Number of raw (or PCA-reduced) input features.  Default: 8.
    n_qubits:
        Number of qubits in the quantum encoding layer.  Default: 8.
    n_layers:
        VQC ansatz layers.  Default: 2.
    n_classes:
        Number of classes.  Default: 3.  Must be ``≥ 2``.
    strategy:
        ``"softmax"`` (default) or ``"one_vs_rest"``.
    use_classical_encoder:
        Prepend ``Linear(n_input_features → n_qubits) + Tanh``.  Default: True.
        If ``False``, input must already lie in (-π, π); it is not rescaled.
    dropout_p:
        Dropout probability applied after the quantum layer.  Default: 0.0.
    device_name:
        PennyLane device string.  Default: ``"lightning.qubit"``.
    diff_method:
        Gradient method.  Default: ``"adjoint"``.
    init_strategy:
        ``"restricted"`` or ``"block_local"``.  Default: ``"restricted"``.
    encoding_type:
        ``"angle"`` or ``"iqp"``.  Default: ``"angle"``.

    Attributes
    ----------
    classical_encoder : nn.Sequential or nn.Identity
    quantum_layer     : QuantumEncodingLayer or IQPEncodingLayer
    dropout           : nn.Dropout or nn.Identity
    head              : nn.Linear
        ``weight[c]`` and ``bias[c]`` are class ``c``'s head.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.models import MulticlassHybridClassifier
    >>> model = MulticlassHybridClassifier(n_input_features=6, n_qubits=4, n_layers=1, n_classes=3)
    >>> x = torch.randn(5, 6)
    >>> model(x).shape                 # logits
    torch.Size([5, 3])
    >>> model.predict_proba(x).shape   # rows sum to 1
    torch.Size([5, 3])
    >>> model.predict(x).shape         # argmax labels in {0, 1, 2}
    torch.Size([5])
    """

    def __init__(
        self,
        n_input_features: int = 8,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_classes: int = 3,
        *,
        strategy: MulticlassStrategy = "softmax",
        use_classical_encoder: bool = True,
        dropout_p: float = 0.0,
        device_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
        init_strategy: str = "restricted",
        encoding_type: str = "angle",
    ) -> None:
        super().__init__()

        if n_classes < 2:
            raise ValueError(f"n_classes must be ≥ 2; got {n_classes}.")
        if strategy not in ("softmax", "one_vs_rest"):
            raise ValueError(f"strategy must be 'softmax' or 'one_vs_rest'; got {strategy!r}.")

        self.n_input_features = n_input_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.strategy = strategy
        self.init_strategy = init_strategy
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

        # ── Quantum encoding layer (shared by every class head) ───────────
        self.quantum_layer: QuantumEncodingLayer | IQPEncodingLayer
        if encoding_type == "angle":
            self.quantum_layer = QuantumEncodingLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                device_name=device_name,  # type: ignore[arg-type]
                diff_method=diff_method,  # type: ignore[arg-type]
            )
        elif encoding_type == "iqp":
            self.quantum_layer = IQPEncodingLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                n_repeats=1,
                device_name=device_name,  # type: ignore[arg-type]
                diff_method=diff_method,  # type: ignore[arg-type]
            )
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")

        # ── Dropout ───────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

        # ── Class heads: one row per class ────────────────────────────────
        self.head = nn.Linear(n_qubits, n_classes)

        # ── Barren-plateau-safe initialisation ────────────────────────────
        self._initialise_weights()

    # ------------------------------------------------------------------
    def _initialise_weights(self) -> None:
        """Restricted-variance init on the quantum weights; Xavier on the linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        weights = self.quantum_layer.qlayer.weights  # (n_layers, n_qubits, 3)
        if self.init_strategy == "block_local":
            block_local_init_(weights.data, n_qubits=self.n_qubits)
        else:
            restricted_normal_init_(weights.data, n_qubits=self.n_qubits, n_layers=self.n_layers)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classical encode → quantum encode → class heads.

        Parameters
        ----------
        x:
            Input tensor, shape ``(batch_size, n_input_features)``.

        Returns
        -------
        torch.Tensor
            Raw logits, shape ``(batch_size, n_classes)``.  Feed to
            ``nn.CrossEntropyLoss`` (``strategy="softmax"``) or to
            ``nn.BCEWithLogitsLoss`` with :meth:`one_hot` targets
            (``strategy="one_vs_rest"``).
        """
        x = self.classical_encoder(x)  # (B, n_qubits)
        # Tanh output (-1, 1) → (-π, π); bypassed input is already in (-π, π).
        if self.use_classical_encoder:
            x = x * torch.pi
        x = self.quantum_layer(x)  # (B, n_qubits), values ∈ [-1, 1]
        x = self.dropout(x)
        return self.head(x)  # (B, n_classes)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-class probabilities, shape ``(batch_size, n_classes)``, rows
        summing to one.  Softmax of the logits under ``"softmax"``; sigmoid
        of each logit, normalised across classes, under ``"one_vs_rest"``.

        Runs in eval mode whatever mode the model is in (dropout off) and
        restores every submodule's ``training`` flag afterwards.
        """
        with eval_mode(self):
            logits = self.forward(x)
        if self.strategy == "softmax":
            return torch.softmax(logits, dim=-1)
        scores = torch.sigmoid(logits)
        return scores / scores.sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicted class labels, shape ``(batch_size,)``, dtype ``torch.long``:
        the argmax over classes.  The same under both strategies, since the
        normalisation in ``predict_proba`` is monotone per row.
        """
        with eval_mode(self):
            logits = self.forward(x)
        return logits.argmax(dim=-1)

    # ------------------------------------------------------------------
    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        """
        Integer labels ``(batch_size,)`` → float one-hot ``(batch_size,
        n_classes)``, the target format of ``nn.BCEWithLogitsLoss`` for the
        one-vs-rest strategy.
        """
        return nn.functional.one_hot(y.long(), num_classes=self.n_classes).to(torch.float32)

    # ------------------------------------------------------------------
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total parameter count (quantum + classical)."""
        params = (
            self.parameters()
            if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"n_input_features={self.n_input_features}, "
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_classes={self.n_classes}, "
            f"strategy={self.strategy!r}, "
            f"total_params={self.count_parameters()}"
        )
