"""
hqnn_forge.sklearn
==================
A scikit-learn estimator around the hybrid classifiers.

``HybridClassifierEstimator`` implements ``fit`` / ``predict`` /
``predict_proba`` / ``get_params`` / ``set_params``, so a hybrid model can be
dropped into ``cross_val_score``, ``GridSearchCV`` and ``Pipeline`` like any
other classifier.  Training is delegated to
:func:`hqnn_forge.training.train_model`.

scikit-learn is an optional dependency; importing this module without it
raises an ``ImportError`` saying how to install it.

Example
-------
::

    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from hqnn_forge.sklearn import HybridClassifierEstimator

    clf = make_pipeline(
        StandardScaler(),
        HybridClassifierEstimator(n_qubits=4, n_layers=2, max_epochs=20),
    )
    scores = cross_val_score(clf, X, y, cv=5, scoring="matthews_corrcoef")
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch

try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.utils.multiclass import unique_labels
    from sklearn.utils.validation import check_is_fitted, validate_data
except ImportError as exc:  # pragma: no cover - exercised only without scikit-learn
    raise ImportError(
        "hqnn_forge.sklearn needs scikit-learn: pip install scikit-learn "
        '(or pip install "hqnn-forge[examples]").'
    ) from exc

from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier
from hqnn_forge.training import TrainingHistory, train_model
from hqnn_forge.utils import FocalLoss

ModelName = Literal["serial", "parallel"]
LossName = Literal["focal", "bce"]


class HybridClassifierEstimator(ClassifierMixin, BaseEstimator):
    """
    scikit-learn classifier training a hybrid quantum-classical model.

    Parameters
    ----------
    model:
        ``"serial"`` (``HybridBinaryClassifier``) or ``"parallel"``
        (``ParallelHybridClassifier``).
    n_qubits, n_layers, encoding_type, init_strategy, use_classical_encoder,
    dropout_p, device_name, diff_method:
        Passed to the model constructor.  ``n_input_features`` is taken from
        the data in ``fit``.
    classical_hidden_dim:
        MLP width; used by the parallel model only.
    loss:
        ``"focal"`` (``FocalLoss()``, the library default for imbalanced data)
        or ``"bce"`` (``BCEWithLogitsLoss``).
    lr:
        Adam learning rate.
    max_epochs, batch_size, patience, monitor:
        Passed to ``train_model``.  Early stopping needs a validation split.
    validation_fraction:
        Share of the training data held out (stratified) for early stopping
        and threshold selection.  ``0`` trains on everything.
    threshold:
        ``"optimal"`` uses the validation-optimal threshold found by
        ``train_model`` (0.5 without a validation split); a float fixes it.
    random_state:
        Seeds weight initialisation, the validation split and batch order.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The two labels; ``classes_[1]`` is the positive class.
    n_features_in_ : int
    model_ : torch.nn.Module
        The trained model.
    history_ : TrainingHistory
    threshold_ : float
        Decision threshold used by ``predict``.
    """

    def __init__(
        self,
        model: ModelName = "serial",
        n_qubits: int = 8,
        n_layers: int = 2,
        encoding_type: str = "angle",
        init_strategy: str = "restricted",
        use_classical_encoder: bool = True,
        classical_hidden_dim: int = 16,
        dropout_p: float = 0.0,
        device_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
        loss: LossName = "focal",
        lr: float = 0.01,
        max_epochs: int = 20,
        batch_size: int = 64,
        validation_fraction: float = 0.0,
        patience: int | None = None,
        monitor: str = "mcc",
        threshold: float | Literal["optimal"] = "optimal",
        random_state: int | None = None,
    ) -> None:
        self.model = model
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        self.init_strategy = init_strategy
        self.use_classical_encoder = use_classical_encoder
        self.classical_hidden_dim = classical_hidden_dim
        self.dropout_p = dropout_p
        self.device_name = device_name
        self.diff_method = diff_method
        self.loss = loss
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.monitor = monitor
        self.threshold = threshold
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _build(self, n_features: int) -> HybridBinaryClassifier | ParallelHybridClassifier:
        common: dict[str, Any] = dict(
            n_input_features=n_features,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            use_classical_encoder=self.use_classical_encoder,
            dropout_p=self.dropout_p,
            device_name=self.device_name,
            diff_method=self.diff_method,
            init_strategy=self.init_strategy,
            encoding_type=self.encoding_type,
        )
        if self.model == "serial":
            return HybridBinaryClassifier(**common)
        if self.model == "parallel":
            return ParallelHybridClassifier(classical_hidden_dim=self.classical_hidden_dim, **common)
        raise ValueError(f"model must be 'serial' or 'parallel'; got {self.model!r}.")

    def _loss(self) -> torch.nn.Module:
        if self.loss == "focal":
            return FocalLoss()
        if self.loss == "bce":
            return torch.nn.BCEWithLogitsLoss()
        raise ValueError(f"loss must be 'focal' or 'bce'; got {self.loss!r}.")

    @staticmethod
    def _stratified_holdout(
        y: npt.NDArray[np.int64], fraction: float, rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        val_parts, train_parts = [], []
        for cls in (0, 1):
            idx = rng.permutation(np.flatnonzero(y == cls))
            n_val = int(round(fraction * idx.size))
            if n_val == 0 or n_val == idx.size:
                raise ValueError(
                    f"validation_fraction={fraction} leaves no training or no validation "
                    f"samples of class {cls} ({idx.size} available)."
                )
            val_parts.append(idx[:n_val])
            train_parts.append(idx[n_val:])
        return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))

    # ------------------------------------------------------------------
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> HybridClassifierEstimator:
        """Build the model for ``X``'s width and train it on ``(X, y)``."""
        X_arr, y_arr = validate_data(self, X, y, dtype=np.float32)
        self.classes_ = unique_labels(y_arr)
        if self.classes_.size != 2:
            raise ValueError(
                f"HybridClassifierEstimator is a binary classifier; got {self.classes_.size} "
                f"classes: {self.classes_.tolist()}."
            )
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(f"validation_fraction must lie in [0, 1); got {self.validation_fraction}.")
        if not (self.threshold == "optimal" or isinstance(self.threshold, (int, float))):
            raise ValueError(f"threshold must be 'optimal' or a number; got {self.threshold!r}.")
        y01 = (y_arr == self.classes_[1]).astype(np.int64)

        seed = self.random_state
        rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.model_ = self._build(X_arr.shape[1])
        loss_fn = self._loss()

        X_t = torch.from_numpy(X_arr)
        if self.validation_fraction > 0:
            tr, va = self._stratified_holdout(y01, self.validation_fraction, rng)
            val: tuple[torch.Tensor, torch.Tensor] | tuple[None, None] = (
                X_t[va], torch.from_numpy(y01[va]).float()
            )
        else:
            tr = np.arange(y01.size)
            val = (None, None)

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self.history_: TrainingHistory = train_model(
            self.model_,
            loss_fn,
            torch.optim.Adam(self.model_.parameters(), lr=self.lr),
            X_t[tr],
            torch.from_numpy(y01[tr]).float(),
            val[0],
            val[1],
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            monitor=self.monitor,
            patience=self.patience,
            generator=generator,
        )
        if self.threshold == "optimal":
            best = self.history_.best_threshold
            self.threshold_ = float(best) if best is not None else 0.5
        else:
            self.threshold_ = float(self.threshold)
        self.model_.eval()
        return self

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities, shape ``(n_samples, 2)``, columns in ``classes_`` order."""
        check_is_fitted(self, "model_")
        X_arr = validate_data(self, X, dtype=np.float32, reset=False)
        positive = self.model_.predict_proba(torch.from_numpy(X_arr)).numpy().astype(np.float64)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Labels from ``classes_``, thresholding the positive probability at ``threshold_``."""
        positive = self.predict_proba(X)[:, 1]
        return self.classes_[(positive >= self.threshold_).astype(np.intp)]

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.non_deterministic = self.random_state is None
        return tags
