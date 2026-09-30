"""
hqnn_forge.sklearn
==================
A scikit-learn estimator around the hybrid classifiers.

``HybridClassifierEstimator`` implements ``fit`` / ``predict`` /
``predict_proba`` / ``get_params`` / ``set_params``, so a hybrid model can be
dropped into ``cross_val_score``, ``GridSearchCV`` and ``Pipeline`` like any
other classifier.  Training is delegated to
:func:`hqnn_forge.training.train_model`.

scikit-learn (>= 1.6) is an optional dependency, declared by the ``sklearn``
extra; importing this module without it raises an ``ImportError`` saying how to
install it.

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

import numbers
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
        'hqnn_forge.sklearn needs scikit-learn >= 1.6: pip install "scikit-learn>=1.6" '
        '(or pip install "hqnn-forge[sklearn]").  validate_data and __sklearn_tags__ '
        "were added in 1.6, so an older install fails this import too."
    ) from exc

from hqnn_forge.models import (
    HybridBinaryClassifier,
    MulticlassHybridClassifier,
    ParallelHybridClassifier,
)
from hqnn_forge.training import TrainingHistory, train_model
from hqnn_forge.utils import FocalLoss, SoftmaxFocalLoss
from hqnn_forge.utils.rng import as_seed, seeded_rng

ModelName = Literal["serial", "parallel"]
LossName = Literal["focal", "bce"]
StrategyName = Literal["softmax", "one_vs_rest"]


class _OneHotLoss(torch.nn.Module):
    """A per-class binary loss on ``(N, K)`` logits and integer labels, via one-hot targets."""

    def __init__(self, loss: torch.nn.Module, n_classes: int) -> None:
        super().__init__()
        self.loss = loss
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        target = torch.nn.functional.one_hot(y.long(), self.n_classes).to(logits.dtype)
        return self.loss(logits, target)  # type: ignore[no-any-return]


class HybridClassifierEstimator(ClassifierMixin, BaseEstimator):
    """
    scikit-learn classifier training a hybrid quantum-classical model.

    Binary or multiclass, decided by ``y`` in ``fit``, like any scikit-learn
    classifier: two classes train a ``HybridBinaryClassifier`` (or the parallel
    model), three or more a ``MulticlassHybridClassifier`` with the same
    circuit options.

    Parameters
    ----------
    model:
        ``"serial"`` (``HybridBinaryClassifier``) or ``"parallel"``
        (``ParallelHybridClassifier``).  With more than two classes only
        ``"serial"`` exists (``MulticlassHybridClassifier``); ``"parallel"``
        raises in ``fit``.
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
        Passed to ``train_model``.  Early stopping needs a validation split:
        with ``validation_fraction=0`` the run always lasts ``max_epochs``,
        whatever ``patience`` says.  ``patience=None`` disables early stopping
        even when there is a split.
    validation_fraction:
        Share of the training data held out (stratified) for early stopping
        and threshold selection.  ``0`` trains on everything.
    strategy:
        More than two classes only: ``"softmax"`` (default; trained with
        cross-entropy, or its focal version) or ``"one_vs_rest"`` (one binary
        head per class, trained with BCE, or the focal loss, on one-hot
        targets).  See :class:`~hqnn_forge.models.MulticlassHybridClassifier`.
    threshold:
        Binary only.  ``"optimal"`` uses the validation-optimal threshold found by
        ``train_model``; a float in ``[0, 1]`` fixes it.  ``train_model``
        searches a threshold for the metric monitors only, so ``"optimal"``
        falls back to 0.5 both without a validation split and under
        ``monitor="val_loss"``.
    random_state:
        Seeds weight initialisation, dropout, the validation split and batch
        order.  The initial weights are the model's ``init_seed=random_state``
        draws; dropout and batch order use seeds spawned from it, so they are
        independent of the init.  A seeded ``fit`` is reproducible and leaves
        the global torch RNG exactly as it was; ``None`` draws everything from
        the global RNG.  The draws of ``noise_method="trajectories"`` share
        the dropout stream, so a seeded noisy ``fit`` is reproducible too.
    noise_level, noise_position, noise_method, noise_trajectories:
        Noise-aware training, passed to the model: depolarizing noise of
        probability ``noise_level`` (in ``[0, 0.75]``) applied to the circuit
        in train mode only, so ``fit`` trains through the noisy circuit and
        ``predict`` / ``predict_proba`` are noiseless.  ``noise_position`` is
        ``"all"`` or ``"end"``; ``noise_method`` is ``"density"`` (exact,
        practical up to about 6 qubits) or ``"trajectories"`` (sampled, at
        pure-state cost), with ``noise_trajectories`` draws per sample.  The
        defaults (no noise) train exactly as without these parameters.  Like
        every other parameter they are validated in ``fit``, so they can be
        tuned with ``GridSearchCV``.  See :mod:`hqnn_forge.noise`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The labels, sorted; with two, ``classes_[1]`` is the positive class.
    n_features_in_ : int
    model_ : torch.nn.Module
        The trained model.
    history_ : TrainingHistory
    threshold_ : float or None
        Decision threshold used by ``predict``; ``None`` with more than two
        classes, where ``predict`` is the argmax.
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
        patience: int | None = 10,
        monitor: str = "mcc",
        threshold: float | Literal["optimal"] = "optimal",
        random_state: int | None = None,
        noise_level: float = 0.0,
        noise_position: str = "all",
        noise_method: str = "density",
        noise_trajectories: int = 1,
        strategy: StrategyName = "softmax",
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
        self.noise_level = noise_level
        self.noise_position = noise_position
        self.noise_method = noise_method
        self.noise_trajectories = noise_trajectories
        self.strategy = strategy

    # ------------------------------------------------------------------
    def _build(
        self, n_features: int, n_classes: int, init_seed: int | None = None
    ) -> HybridBinaryClassifier | ParallelHybridClassifier | MulticlassHybridClassifier:
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
            noise_level=self.noise_level,
            noise_position=self.noise_position,
            noise_method=self.noise_method,
            noise_trajectories=self.noise_trajectories,
            init_seed=init_seed,
        )
        if self.model not in ("serial", "parallel"):
            raise ValueError(f"model must be 'serial' or 'parallel'; got {self.model!r}.")
        if n_classes > 2:
            if self.model != "serial":
                raise ValueError(
                    f"model='parallel' is a binary topology; {n_classes} classes need "
                    f"model='serial' (MulticlassHybridClassifier)."
                )
            return MulticlassHybridClassifier(
                n_classes=n_classes, strategy=self.strategy, **common
            )
        if self.model == "serial":
            return HybridBinaryClassifier(**common)
        return ParallelHybridClassifier(classical_hidden_dim=self.classical_hidden_dim, **common)

    def _loss(self, n_classes: int) -> torch.nn.Module:
        if self.loss not in ("focal", "bce"):
            raise ValueError(f"loss must be 'focal' or 'bce'; got {self.loss!r}.")
        if n_classes == 2:
            return FocalLoss() if self.loss == "focal" else torch.nn.BCEWithLogitsLoss()
        if self.strategy == "softmax":
            return SoftmaxFocalLoss() if self.loss == "focal" else torch.nn.CrossEntropyLoss()
        per_class = FocalLoss() if self.loss == "focal" else torch.nn.BCEWithLogitsLoss()
        return _OneHotLoss(per_class, n_classes)

    @staticmethod
    def _stratified_holdout(
        y: npt.NDArray[np.int64], fraction: float, rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        val_parts, train_parts = [], []
        for cls in np.unique(y):
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

    @staticmethod
    def _check_threshold(threshold: float | Literal["optimal"]) -> None:
        if threshold == "optimal":
            return
        # bool is a subclass of int, and threshold=True would silently mean 1.0.
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
            # ValueError, as for any other threshold outside 'optimal' or [0, 1]
            raise ValueError(  # noqa: TRY004
                f"threshold must be 'optimal' or a real number; got {threshold!r}."
            )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                f"threshold must lie in [0, 1], the range of a probability; got {threshold!r}."
            )

    # ------------------------------------------------------------------
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> HybridClassifierEstimator:
        """Build the model for ``X``'s width and train it on ``(X, y)``."""
        X_arr, y_arr = validate_data(self, X, y, dtype=np.float32)
        classes = unique_labels(y_arr)
        if classes.size < 2:
            # "1 class" is one of the phrasings scikit-learn's conformance
            # checks (check_fit2d_1sample) match on.
            raise ValueError(
                f"HybridClassifierEstimator needs at least two classes; got 1 class: "
                f"{classes.tolist()}."
            )
        if self.strategy not in ("softmax", "one_vs_rest"):
            raise ValueError(
                f"strategy must be 'softmax' or 'one_vs_rest'; got {self.strategy!r}."
            )
        n_classes = int(classes.size)
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must lie in [0, 1); got {self.validation_fraction}."
            )
        self._check_threshold(self.threshold)
        if n_classes > 2 and self.threshold != "optimal":
            raise ValueError(
                f"threshold={self.threshold!r} applies to two classes; with {n_classes}, "
                f"predict is the argmax."
            )
        # Class indices in classes_ order; for two classes, 1 is the positive one.
        y01 = np.searchsorted(classes, y_arr).astype(np.int64)

        # A NumPy integer, as scikit-learn tools pass, is taken as the int it is.
        seed = as_seed(self.random_state, "random_state")
        rng = np.random.default_rng(seed)
        # Three independent torch streams, none of them the caller's (#175):
        # the model seeds its initial weights from random_state itself, and
        # training -- the dropout masks and the batch order -- runs on seeds
        # spawned from it, so neither replays the numbers the init drew.  The
        # global RNG is restored afterwards, so a seeded fit leaves it exactly
        # where it was.
        model = self._build(X_arr.shape[1], n_classes, init_seed=seed)
        loss_fn = self._loss(n_classes)
        # BCE-style losses take float targets, cross-entropy class indices.
        as_target = (lambda t: t.float()) if n_classes == 2 else (lambda t: t.long())
        if seed is None:
            dropout_seed: int | None = None
            generator = None
        else:
            dropout_seed, batch_seed = (
                int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(2)
            )
            generator = torch.Generator().manual_seed(batch_seed)

        X_t = torch.from_numpy(X_arr)
        if self.validation_fraction > 0:
            tr, va = self._stratified_holdout(y01, self.validation_fraction, rng)
            val: tuple[torch.Tensor, torch.Tensor] | tuple[None, None] = (
                X_t[va],
                as_target(torch.from_numpy(y01[va])),
            )
        else:
            tr = np.arange(y01.size)
            val = (None, None)

        with seeded_rng(dropout_seed):
            history = train_model(
                model,
                loss_fn,
                torch.optim.Adam(model.parameters(), lr=self.lr),
                X_t[tr],
                as_target(torch.from_numpy(y01[tr])),
                val[0],
                val[1],
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                monitor=self.monitor,
                patience=self.patience,
                generator=generator,
            )
        threshold: float | None
        if n_classes > 2:
            threshold = None
        elif self.threshold == "optimal":
            best = history.best_threshold
            threshold = float(best) if best is not None else 0.5
        else:
            threshold = float(self.threshold)
        model.eval()

        # Fitted attributes are published only once training has succeeded, so a
        # failed refit leaves the estimator on its previous fit rather than on an
        # untrained model that ``check_is_fitted`` would wave through.
        self.classes_ = classes
        self.history_: TrainingHistory = history
        self.threshold_ = threshold
        self.model_ = model
        return self

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities, shape ``(n_samples, n_classes)``, columns in ``classes_`` order."""
        check_is_fitted(self, "model_")
        X_arr = validate_data(self, X, dtype=np.float32, reset=False)
        proba = self.model_.predict_proba(torch.from_numpy(X_arr)).numpy().astype(np.float64)
        if proba.ndim == 2:  # multiclass: already one column per class
            return proba
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """
        Labels from ``classes_``: the positive probability thresholded at
        ``threshold_`` for two classes, the argmax of the logits for more.
        """
        check_is_fitted(self, "model_")
        if self.threshold_ is None:
            X_arr = validate_data(self, X, dtype=np.float32, reset=False)
            return self.classes_[self.model_.predict(torch.from_numpy(X_arr)).numpy()]
        positive = self.predict_proba(X)[:, 1]
        return self.classes_[(positive >= self.threshold_).astype(np.intp)]

    # ------------------------------------------------------------------
    # Pickling: the fitted model holds a PennyLane QNode built around a local
    # function, which pickle cannot serialise.  The model is stored as its
    # class, constructor arguments and weights instead, and rebuilt on load.
    def __getstate__(self) -> dict[str, Any]:
        # BaseEstimator returns the live __dict__ on Python 3.11+; copy it so
        # pickling does not strip model_ from the estimator itself.
        state = dict(super().__getstate__())
        model = state.pop("model_", None)
        if model is not None:
            state["_model_state"] = {
                "class_name": type(model).__name__,
                "config": model.get_config(),
                "state_dict": model.state_dict(),
            }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = dict(state)
        saved = state.pop("_model_state", None)
        super().__setstate__(state)
        if saved is not None:
            classes = {
                cls.__name__: cls
                for cls in (
                    HybridBinaryClassifier,
                    ParallelHybridClassifier,
                    MulticlassHybridClassifier,
                )
            }
            # Construction initialises weights from the global torch RNG before
            # load_state_dict overwrites them; fork it so unpickling leaves the
            # caller's random stream untouched.
            with torch.random.fork_rng(devices=[]):
                model = classes[saved["class_name"]](**saved["config"])
            model.load_state_dict(saved["state_dict"])
            model.eval()
            self.model_ = model

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = True
        tags.non_deterministic = self.random_state is None
        return tags
