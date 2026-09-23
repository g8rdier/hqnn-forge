"""
hqnn_forge.training.trainer
===========================
Mini-batch training with per-epoch validation and early stopping.

The loop is deliberately small and makes no assumption about the model beyond
``model(x) -> logits`` of shape ``(batch,)`` or ``(batch, 1)``, so it works for
the hybrid classifiers and for any classical baseline alike.  Loss and
optimiser are passed in.

Monitoring
----------
``monitor`` selects what early stopping watches on the validation split:

* ``"val_loss"`` -- the mean validation loss, lower is better.
* ``"mcc"``, ``"f1"``, ``"balanced_accuracy"`` -- the metric at the threshold
  that maximises it on the validation probabilities
  (:func:`hqnn_forge.evaluation.find_optimal_threshold`), higher is better.
  This is the default (``"mcc"``) because on imbalanced data a fixed 0.5
  threshold makes the monitored value mostly a function of calibration.

The threshold found at the best epoch is recorded, so the operating point that
produced the best score travels with the history instead of being re-derived.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from hqnn_forge.evaluation import METRICS, find_optimal_threshold
from hqnn_forge.utils.modes import eval_mode

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class EpochRecord:
    """Statistics for one epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None = None
    val_score: float | None = None
    val_threshold: float | None = None


@dataclass
class TrainingHistory:
    """
    What ``train_model`` did.

    Attributes
    ----------
    epochs:
        One ``EpochRecord`` per completed epoch, in order.
    monitor:
        The monitored quantity.
    best_epoch:
        1-based epoch with the best monitored value, or ``None`` without
        validation data.
    best_value:
        The monitored value at ``best_epoch``.
    best_threshold:
        Decision threshold at ``best_epoch`` (metric monitors only).
    stopped_early:
        ``True`` if patience ran out before ``max_epochs``.
    restored_best:
        ``True`` if the model's weights were rolled back to ``best_epoch``.
    """

    monitor: str
    epochs: list[EpochRecord] = field(default_factory=list)
    best_epoch: int | None = None
    best_value: float | None = None
    best_threshold: float | None = None
    stopped_early: bool = False
    restored_best: bool = False

    @property
    def train_loss(self) -> list[float]:
        return [e.train_loss for e in self.epochs]

    @property
    def n_epochs(self) -> int:
        return len(self.epochs)


def _logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if out.ndim == 2 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    if out.ndim != 1:
        raise ValueError(
            f"model output must have shape (batch,) or (batch, 1); got {tuple(out.shape)}."
        )
    return out


def _check_pair(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X_{name} and y_{name} differ in length: {x.shape[0]} vs {y.shape[0]}.")
    if x.shape[0] == 0:
        raise ValueError(f"X_{name} is empty.")


def train_model(
    model: nn.Module,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    *,
    max_epochs: int = 100,
    batch_size: int = 256,
    monitor: str = "mcc",
    patience: int | None = 10,
    min_delta: float = 0.0,
    restore_best: bool = True,
    generator: torch.Generator | None = None,
    on_epoch_end: Callable[[EpochRecord], None] | None = None,
) -> TrainingHistory:
    """
    Train ``model`` with mini-batches, validating and early-stopping per epoch.

    Parameters
    ----------
    model:
        Any module mapping ``(batch, n_features)`` to logits of shape
        ``(batch,)`` or ``(batch, 1)``.
    loss_fn:
        ``loss_fn(logits, targets) -> scalar``, e.g. ``FocalLoss()`` or
        ``nn.BCEWithLogitsLoss()``.  Targets are passed as float.  Mean
        reduction is assumed for the reported ``train_loss``, which averages
        the batch losses weighted by batch size; with ``reduction="sum"``
        training is unaffected but ``train_loss`` is comparable neither
        across batch sizes nor with the full-batch ``val_loss``.
    optimizer:
        Optimiser already bound to the parameters to train.
    X_train, y_train:
        Training split.
    X_val, y_val:
        Validation split.  Without it the loop runs ``max_epochs`` epochs and
        ``monitor``, ``patience`` and ``restore_best`` have no effect.
    max_epochs:
        Upper bound on the number of epochs.  Default: 100.
    batch_size:
        Mini-batch size.  The last batch may be smaller.  Default: 256.
    monitor:
        ``"mcc"`` (default), ``"f1"``, ``"balanced_accuracy"`` or ``"val_loss"``.
    patience:
        Stop after this many consecutive epochs without an improvement larger
        than ``min_delta``.  ``None`` disables early stopping.  Default: 10.
    min_delta:
        Minimum change that counts as an improvement.  Default: 0.0.
    restore_best:
        Load the weights of the best epoch before returning.  Default: True.
    generator:
        Generator for the per-epoch shuffle, for reproducible batch order.
    on_epoch_end:
        Called with each ``EpochRecord``, e.g. for logging.

    Returns
    -------
    TrainingHistory

    Notes
    -----
    Validation runs in eval mode through ``eval_mode``, so dropout is off and
    every submodule's mode is restored afterwards.  The model is left in train
    mode on return, as it was during training.
    """
    if monitor != "val_loss" and monitor not in METRICS:
        raise ValueError(
            f"unknown monitor {monitor!r}; choose 'val_loss' or one of {sorted(METRICS)}."
        )
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be >= 1; got {max_epochs}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}.")
    if patience is not None and patience < 1:
        raise ValueError(f"patience must be >= 1 or None; got {patience}.")
    if (X_val is None) != (y_val is None):
        raise ValueError("pass both X_val and y_val, or neither.")

    X_train = torch.as_tensor(X_train)
    y_train = torch.as_tensor(y_train).reshape(-1).float()
    _check_pair(X_train, y_train, "train")
    val: tuple[torch.Tensor, torch.Tensor] | None = None
    if X_val is not None and y_val is not None:
        val = (torch.as_tensor(X_val), torch.as_tensor(y_val).reshape(-1).float())
        _check_pair(*val, "val")
        # A single-class split scores the same degenerate value at every
        # threshold, so epoch 1 wins, patience expires and restore_best hands
        # back the initial weights -- silently, on data the model does learn.
        if monitor != "val_loss" and torch.unique(val[1]).numel() < 2:
            raise ValueError(
                f"y_val contains a single class, so the {monitor!r} monitor cannot rank "
                f"epochs on it.  Pass a validation split holding both classes (e.g. a "
                f"stratified one), or monitor='val_loss'."
            )
    has_val = val is not None

    lower_is_better = monitor == "val_loss"
    history = TrainingHistory(monitor=monitor)
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    n = X_train.shape[0]

    for epoch in range(1, max_epochs + 1):
        # ── train ────────────────────────────────────────────────────────
        model.train()
        perm = torch.randperm(n, generator=generator)
        total, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(_logits(model, X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * idx.numel()
            seen += idx.numel()
        record = EpochRecord(epoch=epoch, train_loss=total / seen)

        # ── validate ─────────────────────────────────────────────────────
        if val is not None:
            x_v, y_v = val
            with torch.no_grad(), eval_mode(model):
                val_logits = _logits(model, x_v)
                val_loss = float(loss_fn(val_logits, y_v))
            if lower_is_better:
                value, threshold = val_loss, None
            else:
                val_prob = torch.sigmoid(val_logits)
                if torch.any(torch.isnan(val_prob)):
                    # find_optimal_threshold rejects NaN probabilities rather
                    # than label them negative.  Diverging must not take the
                    # history and the best-epoch snapshot down with it, so
                    # score the epoch NaN, as the val_loss path already does:
                    # it never improves, and patience ends the run.
                    value, threshold = math.nan, None
                else:
                    search = find_optimal_threshold(y_v.long(), val_prob, metric=monitor)
                    value, threshold = search.score, search.threshold
            record = EpochRecord(
                epoch=epoch,
                train_loss=record.train_loss,
                val_loss=val_loss,
                val_score=None if lower_is_better else value,
                val_threshold=threshold,
            )

            if history.best_value is None or math.isnan(history.best_value):
                improved = not math.isnan(value)
            elif lower_is_better:
                improved = value < history.best_value - min_delta
            else:
                improved = value > history.best_value + min_delta

            if improved:
                history.best_epoch, history.best_value = epoch, value
                history.best_threshold = threshold
                epochs_without_improvement = 0
                if restore_best:
                    best_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

        history.epochs.append(record)
        if on_epoch_end is not None:
            on_epoch_end(record)

        if has_val and patience is not None and epochs_without_improvement >= patience:
            history.stopped_early = epoch < max_epochs
            break

    if restore_best and best_state is not None and history.best_epoch != history.n_epochs:
        model.load_state_dict(best_state)
        history.restored_best = True
    model.train()
    return history
