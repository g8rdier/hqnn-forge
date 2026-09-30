"""
hqnn_forge.evaluation.calibration
=================================
Calibration of a binary classifier's probabilities, and post-hoc fixes (#319).

For risk scoring -- fraud, credit, churn -- the predicted probability is used as
a probability, so it has to mean what it says: of the samples scored 0.2, about
20 % should be positive.  Ranking (AUC) and thresholded performance (MCC) do
not see this, and training choices move it: focal loss, the library default,
is known to under-state confidence (Mukhoti et al. 2020), and resampling for
imbalance shifts the base rate the model learns.

Measures
--------
``brier_score``
    Mean squared error ``mean((p − y)²)`` of the probabilities: calibration
    and sharpness together; 0 is perfect, 0.25 is a constant 0.5.
``expected_calibration_error``
    ``Σ_b (n_b / n) · |mean(y)_b − mean(p)_b|`` over probability bins: the
    average gap between confidence and frequency (Naeini et al. 2015; Guo et
    al. 2017), here on the positive-class probability.  ``strategy="uniform"``
    uses ``n_bins`` equal-width bins of [0, 1]; ``"quantile"`` equal-count
    bins, which on imbalanced data keeps the bins near 0 from swallowing
    every sample.
``reliability_curve``
    Per non-empty bin, the mean probability and the observed frequency, the
    points of a reliability diagram (``plots.plot_reliability_diagram``).

Post-hoc calibration
--------------------
Both are fitted on a validation split by minimising the negative
log-likelihood of the labels, and applied to logits:

``TemperatureScaler``
    ``σ(z / T)``: one parameter.  Monotone, so the ranking -- AUC, and the
    MCC at a correspondingly moved threshold -- is unchanged; only
    confidence is rescaled.  ``T > 1`` softens over-confident predictions.
``PlattScaler``
    ``σ(a·z + b)`` (Platt 1999): also shifts the base rate, e.g. after
    training on oversampled data.  Monotone for ``a > 0``.

References
----------
* Platt (1999) "Probabilistic outputs for support vector machines".
* Naeini, Cooper & Hauskrecht (2015) "Obtaining well calibrated
  probabilities using Bayesian binning", AAAI.
* Guo, Pleiss, Sun & Weinberger (2017) "On calibration of modern neural
  networks", ICML.
* Mukhoti et al. (2020) "Calibrating deep neural networks using focal loss",
  NeurIPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

__all__ = [
    "PlattScaler",
    "TemperatureScaler",
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
]

BinStrategy = Literal["uniform", "quantile"]


def _pair(y_true: object, prob: object) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.as_tensor(y_true).reshape(-1).to(torch.float64)
    # contiguous: a column of a 2-D array (predict_proba(X)[:, 1]) is a strided
    # view, which torch.searchsorted in the binning copies with a warning.
    p = torch.as_tensor(prob).reshape(-1).to(torch.float64).contiguous()
    if y.shape != p.shape:
        raise ValueError(f"y_true and prob differ in length: {y.numel()} vs {p.numel()}.")
    if y.numel() == 0:
        raise ValueError("y_true is empty.")
    if not torch.isin(y, torch.tensor([0.0, 1.0], dtype=torch.float64)).all():
        raise ValueError("y_true must hold binary labels 0 and 1.")
    if torch.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise ValueError("prob must hold probabilities in [0, 1].")
    return y, p


def brier_score(y_true: object, prob: object) -> float:
    """``mean((prob − y_true)²)``."""
    y, p = _pair(y_true, prob)
    return float(((p - y) ** 2).mean())


def _bins(p: torch.Tensor, n_bins: int, strategy: BinStrategy) -> torch.Tensor:
    """Bin index of every probability; the top edge belongs to the last bin."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}.")
    if strategy == "uniform":
        edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    elif strategy == "quantile":
        edges = torch.quantile(p, torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64))
    else:
        raise ValueError(f"strategy must be 'uniform' or 'quantile'; got {strategy!r}.")
    # Right-closed at the top, as sklearn.calibration.calibration_curve does.
    return torch.searchsorted(edges[1:-1], p, right=True)


def reliability_curve(
    y_true: object, prob: object, n_bins: int = 10, strategy: BinStrategy = "uniform"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ``(mean probability, observed frequency, count)`` per non-empty bin.

    With the default uniform bins this is ``sklearn.calibration.calibration_curve``
    (which returns the frequency first and the mean probability second).
    """
    y, p = _pair(y_true, prob)
    idx = _bins(p, n_bins, strategy)
    counts = torch.bincount(idx, minlength=n_bins).to(torch.float64)
    sum_p = torch.bincount(idx, weights=p, minlength=n_bins)
    sum_y = torch.bincount(idx, weights=y, minlength=n_bins)
    keep = counts > 0
    return sum_p[keep] / counts[keep], sum_y[keep] / counts[keep], counts[keep]


def expected_calibration_error(
    y_true: object, prob: object, n_bins: int = 10, strategy: BinStrategy = "uniform"
) -> float:
    """``Σ_b (n_b / n) · |frequency_b − mean probability_b|`` over the bins."""
    confidence, frequency, counts = reliability_curve(y_true, prob, n_bins, strategy)
    return float((counts / counts.sum() * (frequency - confidence).abs()).sum())


def _logit_pair(logits: object, y_true: object) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.as_tensor(logits).detach().reshape(-1).to(torch.float64)
    y, _ = _pair(y_true, torch.sigmoid(z))
    if torch.unique(y).numel() < 2:
        raise ValueError("fitting a calibration needs both classes in y_true.")
    return z, y


@dataclass(frozen=True)
class TemperatureScaler:
    """``σ(z / temperature)``; see the module docstring."""

    temperature: float

    @classmethod
    def fit(cls, logits: object, y_true: object) -> TemperatureScaler:
        """The temperature minimising the validation NLL (L-BFGS on ``log T``, so ``T > 0``)."""
        z, y = _logit_pair(logits, y_true)
        log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.LBFGS(
            [log_t],
            lr=1.0,
            max_iter=500,
            tolerance_grad=1e-10,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
            loss.backward()
            return loss

        optimiser.step(closure)
        return cls(float(log_t.detach().exp()))

    def __call__(self, logits: object) -> torch.Tensor:
        """Calibrated positive-class probabilities, ``float64``."""
        z = torch.as_tensor(logits).detach().to(torch.float64)
        return torch.sigmoid(z / self.temperature)


@dataclass(frozen=True)
class PlattScaler:
    """``σ(a·z + b)``; see the module docstring."""

    a: float
    b: float

    @classmethod
    def fit(cls, logits: object, y_true: object) -> PlattScaler:
        """``(a, b)`` minimising the validation NLL: logistic regression on the logit."""
        z, y = _logit_pair(logits, y_true)
        ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.LBFGS(
            [ab],
            lr=1.0,
            max_iter=500,
            tolerance_grad=1e-10,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            loss = F.binary_cross_entropy_with_logits(ab[0] * z + ab[1], y)
            loss.backward()
            return loss

        optimiser.step(closure)
        a, b = ab.detach().tolist()
        return cls(float(a), float(b))

    def __call__(self, logits: object) -> torch.Tensor:
        """Calibrated positive-class probabilities, ``float64``."""
        z = torch.as_tensor(logits).detach().to(torch.float64)
        return torch.sigmoid(self.a * z + self.b)
