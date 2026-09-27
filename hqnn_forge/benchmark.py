"""
hqnn_forge.benchmark
====================
Train a hybrid model and its matched classical control on identical folds of
several datasets, and report one row per dataset and model.

This answers the library's question -- does a small quantum layer earn its
parameters? -- the same way every time, instead of by a comparison assembled
by hand.

What both models share
----------------------
For each dataset, the outer folds come from :func:`stratified_kfold`, and in
every fold the hybrid model and its control (:func:`classical_baseline`, the
same parameter count) get exactly the same:

* **Scaling.**  Features are standardised with the mean and standard deviation
  of the fold's training part only, then applied to every row of the fold.
* **Validation split.**  The training part is split once more, stratified: one
  ``validation_folds``-th is held out for early stopping and threshold tuning.
  It holds only real rows.
* **Oversampling.**  With ``oversample=True``, SMOTE runs on the remaining
  training rows only, after scaling, since its neighbour search measures
  distances.  Test and validation rows never contribute a synthetic sample.
* **Training.**  The same loss, optimiser, learning rate, batch size, epoch
  budget and early stopping, the same batch order (one generator seed per
  fold), and the same initialisation seed.  Each model is built and trained
  inside :func:`torch.random.fork_rng`, so the run neither depends on nor
  disturbs the caller's global RNG.
* **Threshold.**  The threshold that maximises MCC on the validation split at
  the best epoch (``TrainingHistory.best_threshold``), applied unchanged to
  the fold's test rows.  The test rows are used for nothing else.

Reported per dataset and model
------------------------------
``mcc_mean``/``mcc_std`` over the test folds, ``n_parameters`` and MCC per
1,000 parameters (:func:`parameter_efficiency` of the mean), the wall-clock
training time summed over folds (simulating the circuit is part of an honest
efficiency comparison), and the paired Wilcoxon signed-rank test of hybrid
against control over the per-fold MCCs, with its effect size.  The test
columns are the same on both rows of a dataset.

``wilcoxon_min_p`` is the smallest p-value the test could have produced with
this many folds: with 5 folds, two-sided, it is 0.0625, so no outcome reaches
0.05.  A difference then needs more folds (``n_splits``) to be declared, not
a lower threshold.
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from hqnn_forge.evaluation import (
    matthews_corrcoef,
    parameter_efficiency,
    rank_biserial_correlation,
    wilcoxon_signed_rank,
)
from hqnn_forge.models import BinaryClassifierBase, HybridBinaryClassifier
from hqnn_forge.preprocessing import oversample_fold, stratified_kfold
from hqnn_forge.training import train_model
from hqnn_forge.utils import FocalLoss, classical_baseline

HybridBuilder = Callable[[int], nn.Module]
LossBuilder = Callable[[], nn.Module]

#: The two models of every dataset, in report order.
MODELS: tuple[str, str] = ("hybrid", "control")

#: Columns of :attr:`BenchmarkResult.records` and of the CSV, in order.
COLUMNS: tuple[str, ...] = (
    "dataset",
    "model",
    "architecture",
    "n_samples",
    "n_positives",
    "n_folds",
    "n_parameters",
    "mcc_mean",
    "mcc_std",
    "mcc_per_kparam",
    "fold_mcc",
    "train_seconds",
    "wilcoxon_p",
    "wilcoxon_min_p",
    "rank_biserial",
)


def default_hybrid(n_input_features: int) -> nn.Module:
    """``HybridBinaryClassifier(n_input_features, n_qubits=8, n_layers=2)``."""
    return HybridBinaryClassifier(n_input_features, 8, 2)


@dataclass(frozen=True)
class FoldResult:
    """
    One model on one outer fold.  The index arrays point into the dataset's
    rows, so the two models of a fold can be checked to have seen the same
    data.
    """

    dataset: str
    model: str
    fold: int
    train_idx: npt.NDArray[np.intp]
    """Rows the model was trained on (before SMOTE added synthetic ones)."""
    val_idx: npt.NDArray[np.intp]
    """Rows used for early stopping and threshold tuning."""
    test_idx: npt.NDArray[np.intp]
    """Rows the reported score comes from."""
    n_synthetic: int
    init_seed: int
    batch_seed: int
    threshold: float
    mcc: float
    train_seconds: float
    epochs: int


@dataclass(frozen=True)
class BenchmarkResult:
    """
    Attributes
    ----------
    records:
        One dict per dataset and model with the keys in :data:`COLUMNS`;
        ``fold_mcc`` is a tuple of the per-fold scores.
    folds:
        Every :class:`FoldResult`, in dataset, fold, model order.
    """

    records: list[dict[str, Any]]
    folds: list[FoldResult]


def _standardise(
    X: npt.NDArray[np.float64], rows: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """``X`` scaled by the mean and std of ``X[rows]``; constant columns only centred."""
    mean = X[rows].mean(axis=0)
    std = X[rows].std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def _seeds(root: np.random.SeedSequence, n: int) -> list[int]:
    return [int(s.generate_state(1)[0]) for s in root.spawn(n)]


def _fit_and_score(
    model: BinaryClassifierBase,
    loss: LossBuilder,
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int64],
    X_val: npt.NDArray[np.float64],
    y_val: npt.NDArray[np.int64],
    X_test: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.int64],
    *,
    lr: float,
    max_epochs: int,
    batch_size: int,
    patience: int | None,
    batch_seed: int,
) -> tuple[float, float, float, int]:
    """Train, then score the test rows at the validation threshold: (mcc, threshold, s, epochs)."""
    as_tensor = torch.from_numpy
    start = time.perf_counter()
    history = train_model(
        model,
        loss(),
        torch.optim.Adam(model.parameters(), lr=lr),
        as_tensor(X_train.astype(np.float32)),
        as_tensor(y_train.astype(np.float32)),
        as_tensor(X_val.astype(np.float32)),
        as_tensor(y_val.astype(np.float32)),
        max_epochs=max_epochs,
        batch_size=batch_size,
        monitor="mcc",
        patience=patience,
        generator=torch.Generator().manual_seed(batch_seed),
    )
    seconds = time.perf_counter() - start
    threshold = history.best_threshold if history.best_threshold is not None else 0.5
    prob = model.predict_proba(as_tensor(X_test.astype(np.float32)))
    mcc = matthews_corrcoef(y_test, (prob >= threshold).long())
    return float(mcc), float(threshold), seconds, history.n_epochs


def run_benchmark(
    datasets: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]],
    hybrid: HybridBuilder = default_hybrid,
    *,
    n_splits: int = 5,
    validation_folds: int = 5,
    oversample: bool = True,
    loss: LossBuilder = FocalLoss,
    lr: float = 0.01,
    max_epochs: int = 100,
    batch_size: int = 256,
    patience: int | None = 10,
    random_state: int = 0,
    smote_kwargs: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    """
    Compare ``hybrid`` with its matched classical control on every dataset.

    See the module docstring for what the two models share in each fold.

    Parameters
    ----------
    datasets:
        Name → ``(X, y)``, with ``y`` binary 0/1 and 1 the rare class, e.g.
        ``{"credit-card": load_credit_card_fraud()[:2]}``.
    hybrid:
        ``hybrid(n_input_features)`` returns a fresh, untrained
        ``HybridBinaryClassifier`` or ``ParallelHybridClassifier``; the control
        is ``classical_baseline`` of it.  Called once per fold.
    n_splits:
        Outer folds per dataset.  At least 2; see the module docstring for
        the smallest p-value a given number of folds allows.
    validation_folds:
        One ``validation_folds``-th of each training part is held out for
        validation.  Default 5, i.e. 20 %.
    oversample:
        SMOTE the training rows (``smote_kwargs`` are passed on).
    loss:
        Called once per model and fold for a fresh loss, default
        :class:`FocalLoss`.
    lr, max_epochs, batch_size, patience:
        Adam learning rate and the :func:`train_model` settings, the same for
        both models.
    random_state:
        Root seed.  Every fold split, SMOTE draw, initialisation and batch
        order derives from it, so a run is repeatable; the seeds used are in
        :attr:`BenchmarkResult.folds`.

    Returns
    -------
    BenchmarkResult
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2; got {n_splits}.")
    if validation_folds < 2:
        raise ValueError(f"validation_folds must be >= 2; got {validation_folds}.")
    if not datasets:
        raise ValueError("datasets is empty.")
    smote_options = dict(smote_kwargs or {})

    records: list[dict[str, Any]] = []
    folds: list[FoldResult] = []
    for d, (name, (X_raw, y_raw)) in enumerate(datasets.items()):
        X = np.asarray(X_raw, dtype=np.float64)
        y = np.asarray(y_raw)
        if X.ndim != 2 or y.shape != (X.shape[0],):
            raise ValueError(
                f"{name}: X must be (n_samples, n_features) and y (n_samples,); "
                f"got {X.shape} and {y.shape}."
            )
        if not np.isin(y, (0, 1)).all():
            raise ValueError(f"{name}: y must be binary 0/1.")
        y = y.astype(np.int64)
        if not np.isfinite(X).all():
            raise ValueError(f"{name}: X contains NaN or infinite values.")

        root = np.random.SeedSequence([random_state, d])
        split_seed, *fold_roots = root.spawn(n_splits + 1)
        outer = stratified_kfold(y, n_splits, random_state=int(split_seed.generate_state(1)[0]))
        scores: dict[str, list[float]] = {m: [] for m in MODELS}
        seconds: dict[str, float] = {m: 0.0 for m in MODELS}
        n_parameters: dict[str, int] = {}
        architecture: dict[str, str] = {}

        for k, ((train_part, test_idx), fold_root) in enumerate(zip(outer, fold_roots)):
            inner_seed, smote_seed, init_seed, batch_seed = _seeds(fold_root, 4)
            inner_tr, inner_va = stratified_kfold(
                y[train_part], validation_folds, random_state=inner_seed
            )[0]
            train_idx, val_idx = train_part[inner_tr], train_part[inner_va]
            X_fold = _standardise(X, train_part)
            if oversample:
                fold = oversample_fold(
                    X_fold, y, train_idx, val_idx, random_state=smote_seed, **smote_options
                )
                X_train, y_train, n_synthetic = fold.X_train, fold.y_train, len(fold.sources)
            else:
                X_train, y_train, n_synthetic = X_fold[train_idx], y[train_idx], 0

            for model_name in MODELS:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(init_seed)
                    built = hybrid(X.shape[1])
                    model = classical_baseline(built) if model_name == "control" else built
                    if not isinstance(model, BinaryClassifierBase):
                        raise TypeError(
                            f"hybrid must return a hybrid classifier; got {type(model).__name__}."
                        )
                    mcc, threshold, secs, epochs = _fit_and_score(
                        model,
                        loss,
                        X_train,
                        np.asarray(y_train, dtype=np.int64),
                        X_fold[val_idx],
                        y[val_idx],
                        X_fold[test_idx],
                        y[test_idx],
                        lr=lr,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                        patience=patience,
                        batch_seed=batch_seed,
                    )
                n_parameters[model_name] = model.count_parameters()
                architecture[model_name] = type(model).__name__
                scores[model_name].append(mcc)
                seconds[model_name] += secs
                folds.append(
                    FoldResult(
                        name,
                        model_name,
                        k,
                        np.sort(train_idx),
                        np.sort(val_idx),
                        np.sort(test_idx),
                        n_synthetic,
                        init_seed,
                        batch_seed,
                        threshold,
                        mcc,
                        secs,
                        epochs,
                    )
                )

        hybrid_scores, control_scores = scores["hybrid"], scores["control"]
        try:
            test = wilcoxon_signed_rank(hybrid_scores, control_scores)
            p, min_p = test.p_value, test.min_p_value
        except ValueError:  # every fold tied: the test is undefined
            p = min_p = math.nan
        effect = rank_biserial_correlation(hybrid_scores, control_scores)

        for model_name in MODELS:
            fold_scores = np.asarray(scores[model_name])
            mean = float(fold_scores.mean())
            records.append(
                {
                    "dataset": name,
                    "model": model_name,
                    "architecture": architecture[model_name],
                    "n_samples": int(y.size),
                    "n_positives": int(y.sum()),
                    "n_folds": n_splits,
                    "n_parameters": n_parameters[model_name],
                    "mcc_mean": mean,
                    "mcc_std": float(fold_scores.std(ddof=1)),
                    "mcc_per_kparam": parameter_efficiency(n_parameters[model_name], mean),
                    "fold_mcc": tuple(float(s) for s in fold_scores),
                    "train_seconds": seconds[model_name],
                    "wilcoxon_p": p,
                    "wilcoxon_min_p": min_p,
                    "rank_biserial": effect,
                }
            )
    return BenchmarkResult(records, folds)


def write_csv(records: Sequence[Mapping[str, Any]], path: str | os.PathLike[str]) -> None:
    """
    Write ``records`` to ``path`` with the columns in :data:`COLUMNS`;
    ``fold_mcc`` is written as its scores joined by ``;``.
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for record in records:
            row = {column: record[column] for column in COLUMNS}
            row["fold_mcc"] = ";".join(repr(s) for s in record["fold_mcc"])
            writer.writerow(row)
