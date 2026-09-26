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

* Initialisation (``"restricted"``, ``"block_local"``, ``"normal"`` with
  ``init_std``), encoder bypass and ``encoding_type`` behave as in
  :class:`~hqnn_forge.models.HybridBinaryClassifier`.  The circuit options
  added there since (``embedding_rotation``, ``entangler``, ``readout``,
  ``encoder_activation``) are not supported here yet: this model always uses
  the RX embedding, CNOT ring, all-qubit readout and a tanh encoder.

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
encoding_type, init_std:
    As for :class:`~hqnn_forge.models.HybridBinaryClassifier`.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from hqnn_forge.encoding.angle_embedding import DeviceName, DiffMethod, QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.initializers.restricted_variance import (
    block_local_init_,
    restricted_normal_init_,
)
from hqnn_forge.utils.modes import eval_mode

MulticlassStrategy = Literal["softmax", "one_vs_rest"]

_DEFAULT_INIT_STD = 0.1


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
        Dropout probability applied after the quantum layer, in ``[0, 1)``.
        Default: 0.0.
    device_name:
        PennyLane device string.  Default: ``"lightning.qubit"``.
    diff_method:
        Gradient method.  Default: ``"adjoint"``.
    init_strategy:
        ``"restricted"``, ``"block_local"`` or ``"normal"``
        (``N(0, init_std²)``).  Default: ``"restricted"``.
    encoding_type:
        ``"angle"`` or ``"iqp"``.  Default: ``"angle"``.
    init_std:
        Standard deviation for ``init_strategy="normal"``.  Default: 0.1.
        Any other value with another ``init_strategy`` raises, since it would
        be recorded in the config and ignored.

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
        device_name: DeviceName = "lightning.qubit",
        diff_method: DiffMethod = "adjoint",
        init_strategy: str = "restricted",
        encoding_type: str = "angle",
        init_std: float = _DEFAULT_INIT_STD,
    ) -> None:
        super().__init__()
        self._config: dict[str, Any] = dict(
            n_input_features=n_input_features,
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_classes=n_classes,
            strategy=strategy,
            use_classical_encoder=use_classical_encoder,
            dropout_p=dropout_p,
            device_name=device_name,
            diff_method=diff_method,
            init_strategy=init_strategy,
            encoding_type=encoding_type,
            init_std=init_std,
        )

        if n_classes < 2:
            raise ValueError(f"n_classes must be ≥ 2; got {n_classes}.")
        if strategy not in ("softmax", "one_vs_rest"):
            raise ValueError(f"strategy must be 'softmax' or 'one_vs_rest'; got {strategy!r}.")
        if init_strategy not in ("restricted", "block_local", "normal"):
            raise ValueError(
                f"init_strategy must be 'restricted', 'block_local' or 'normal'; "
                f"got {init_strategy!r}."
            )
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1); got {dropout_p}.")
        if init_std <= 0.0:
            raise ValueError(f"init_std must be > 0; got {init_std}.")
        if init_strategy != "normal" and init_std != _DEFAULT_INIT_STD:
            raise ValueError(
                f"init_std applies to init_strategy='normal' only; "
                f"'{init_strategy}' derives its own sigma from the circuit size, so "
                f"init_std={init_std} would be recorded in the config and ignored."
            )

        self.n_input_features = n_input_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.strategy = strategy
        self.init_strategy = init_strategy
        self.init_std = init_std
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

        # ── Dropout ───────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

        # ── Class heads: one row per class ────────────────────────────────
        self.head = nn.Linear(n_qubits, n_classes)

        # ── Small-angle restricted-variance initialisation ─────────────────
        self._initialise_weights()

    # ------------------------------------------------------------------
    def _initialise_weights(self) -> None:
        """Restricted-variance (or chosen) init on the quantum weights; Xavier on the linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        weights = self.quantum_layer.qlayer.weights  # (n_layers, n_qubits, 3)
        if self.init_strategy == "block_local":
            block_local_init_(weights.data, n_qubits=self.n_qubits)
        elif self.init_strategy == "normal":
            with torch.no_grad():
                weights.normal_(mean=0.0, std=self.init_std)
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
        The latter is computed as ``softmax(logsigmoid(logits))``, which is
        the same quantity but stays finite when every sigmoid in a row
        underflows (all logits below about -88 in float32), where the direct
        ratio would be ``0 / 0``.

        Runs in eval mode whatever mode the model is in (dropout off) and
        restores every submodule's ``training`` flag afterwards.
        """
        with eval_mode(self):
            logits = self.forward(x)
        if self.strategy == "softmax":
            return torch.softmax(logits, dim=-1)
        return torch.softmax(nn.functional.logsigmoid(logits), dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicted class labels, shape ``(batch_size,)``, dtype ``torch.long``:
        the argmax of the logits, under both strategies.

        This is the argmax of :meth:`predict_proba` in exact arithmetic, since
        both normalisations are monotone per row, but not always in floating
        point: under ``"one_vs_rest"`` large positive logits saturate the
        sigmoid, so e.g. logits ``[17, 20, 30]`` give float32 probabilities in
        which classes 1 and 2 tie, and ``predict_proba(x).argmax(-1)`` returns
        class 1 while ``predict`` returns class 2.  Use this method, not
        ``predict_proba(x).argmax(-1)``, for labels.
        """
        with eval_mode(self):
            logits = self.forward(x)
        return logits.argmax(dim=-1)

    # ------------------------------------------------------------------
    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        """
        Integer labels ``(batch_size,)`` → one-hot ``(batch_size, n_classes)``
        in the dtype of the class heads, the target format of
        ``nn.BCEWithLogitsLoss`` for the one-vs-rest strategy.  Matching the
        heads' dtype keeps the loss of a ``model.double()`` in float64.

        Float labels are accepted only if integer-valued; anything else (e.g.
        smoothed targets) raises instead of being truncated.
        """
        if y.is_floating_point() and not torch.equal(y, y.round()):
            raise ValueError("one_hot expects integer class labels; got non-integer values.")
        one_hot = nn.functional.one_hot(y.long(), num_classes=self.n_classes)
        return one_hot.to(self.head.weight.dtype)

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
    def get_config(self) -> dict[str, Any]:
        """
        Constructor arguments of this model, as a fresh dict.

        ``type(model)(**model.get_config())`` builds a model with the same
        architecture (weights are re-initialised; load a ``state_dict`` for
        those).  Used by ``hqnn_forge.utils.checkpoint``.
        """
        return dict(self._config)

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
