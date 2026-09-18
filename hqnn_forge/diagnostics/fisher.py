"""
hqnn_forge.diagnostics.fisher
=============================
Fisher information spectrum and effective dimension of a model's quantum
parameters.

The case for HQNN parameter efficiency rests on quantum circuits having a
more evenly spread Fisher information spectrum, hence a higher *effective
dimension* (Abbas et al. 2021), than classical networks at equal parameter
count.  These helpers compute both for a model built from this library, so
the claim can be checked on the architecture at hand instead of asserted
from the literature.

Fisher information
------------------
For a statistical model ``p(y | x; θ)`` the Fisher information matrix is::

    F(θ) = E_{x} E_{y ~ p(·|x;θ)} [ ∇_θ log p(y|x;θ) ∇_θ log p(y|x;θ)ᵀ ]

The expectation over ``x`` is taken over the rows of ``data_sample``; the
expectation over ``y`` is taken in closed form, which needs a likelihood:

* A **hybrid classifier** (``forward`` returns one logit ``z`` per sample)
  is the Bernoulli model ``p(y=1|x) = σ(z)``.  Then
  ``E_y[∇ log p ∇ log pᵀ] = σ(z)(1 − σ(z)) ∇z ∇zᵀ``, exactly, so no labels
  are needed and nothing is sampled.
* An **encoding layer** (``forward`` returns ``n_qubits`` expectation values)
  has no likelihood of its own.  It is treated as a Gaussian observation
  model with unit variance around its outputs, for which the Fisher matrix
  is ``E_x[ Jᵀ J ]`` with ``J = ∂ outputs / ∂ θ``: the Gauss-Newton matrix.
  This is the natural information matrix of a regression-style readout and
  what "the Fisher spectrum of the circuit" means in practice.

Only the quantum layer's weights are differentiated, so the spectrum is that
of the circuit's parameters, whatever classical layers sit around it.

Effective dimension
-------------------
Abbas et al. define, for ``d`` parameters, ``n`` data points and
``γ ∈ (0, 1]``::

    κ        = γ n / (2π log n)
    F̂(θ)     = d · F(θ) / E_θ[ tr F(θ) ]              (trace normalised to d)
    d_{γ,n}  = 2 log( E_θ[ sqrt(det(I_d + κ F̂(θ))) ] ) / log κ

where ``E_θ`` is over the parameter distribution the model is initialised
from.  :func:`effective_dimension` draws that expectation with the same
``init`` choices as :func:`~hqnn_forge.diagnostics.gradient_variance`
(``"uniform"`` over [0, 2π) is the reference used in the paper).  The
normalised ``d_{γ,n} / d`` is the number that is comparable across
architectures with different ``d``: it is close to 1 when the parameters are
all used in independent directions and small when most of them are
redundant.  Note that for finite ``n`` the estimate can exceed ``d``
slightly (``F̂ = I`` gives ``d · log(1 + κ) / log κ``); this is a property of
the definition, not a bug.

References
----------
* Abbas et al. (2021) "The power of quantum neural networks", Nature
  Computational Science 1, 403.
* Berezniuk et al. (2020) "A scale-dependent notion of effective dimension
  for generalization", arXiv:2001.10872.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from hqnn_forge.diagnostics.gradients import InitFn, InitName, _make_init, _resolve_weights


@dataclass(frozen=True)
class FisherSpectrum:
    """
    Fisher information of a model's quantum weights at their current values.

    Attributes
    ----------
    layer_type:
        Class name of the model or layer that was measured.
    likelihood:
        ``"bernoulli"`` for a classifier, ``"gaussian"`` for an encoding layer.
    n_params:
        Number of quantum weights, ``d``.
    n_data:
        Rows of ``data_sample`` the expectation over ``x`` was taken over.
    matrix:
        The ``(d, d)`` Fisher matrix, ``float64``.
    eigenvalues:
        Its eigenvalues in descending order, ``float64``, length ``d``.
    """

    layer_type: str
    likelihood: str
    n_params: int
    n_data: int
    matrix: torch.Tensor
    eigenvalues: torch.Tensor

    @property
    def trace(self) -> float:
        return float(self.eigenvalues.sum())

    @property
    def rank(self) -> int:
        """Eigenvalues above ``1e-10 · max``; 0 if the matrix is zero."""
        top = float(self.eigenvalues.max()) if self.n_params else 0.0
        if top <= 0.0:
            return 0
        return int((self.eigenvalues > 1e-10 * top).sum())

    @property
    def normalized_eigenvalues(self) -> torch.Tensor:
        """Eigenvalues scaled to sum to ``n_params`` (zeros if the trace is 0)."""
        if self.trace <= 0.0:
            return torch.zeros_like(self.eigenvalues)
        return self.eigenvalues * (self.n_params / self.trace)

    def to_dict(self) -> dict[str, Any]:
        """Scalar fields plus the eigenvalues as a list, for logging."""
        return {
            "layer_type": self.layer_type,
            "likelihood": self.likelihood,
            "n_params": self.n_params,
            "n_data": self.n_data,
            "trace": self.trace,
            "rank": self.rank,
            "eigenvalues": self.eigenvalues.tolist(),
        }


@dataclass(frozen=True)
class EffectiveDimensionResult:
    """
    Effective dimension of a model's quantum weights.

    Attributes
    ----------
    layer_type, init, n_params, n_data, gamma:
        What was measured and with which constants.
    n_theta_samples:
        Parameter draws the expectation over ``θ`` was taken over.
    effective_dimension:
        ``d_{γ,n}`` of Abbas et al.
    normalized_effective_dimension:
        ``d_{γ,n} / n_params``, the figure to compare across architectures.
    mean_normalized_spectrum:
        Mean over draws of the trace-normalised eigenvalues, descending.
    """

    layer_type: str
    init: str
    n_params: int
    n_data: int
    gamma: float
    n_theta_samples: int
    effective_dimension: float
    normalized_effective_dimension: float
    mean_normalized_spectrum: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        """Scalar fields only, for logging."""
        return {
            "layer_type": self.layer_type,
            "init": self.init,
            "n_params": self.n_params,
            "n_data": self.n_data,
            "gamma": self.gamma,
            "n_theta_samples": self.n_theta_samples,
            "effective_dimension": self.effective_dimension,
            "normalized_effective_dimension": self.normalized_effective_dimension,
        }


# ---------------------------------------------------------------------------
# Fisher matrix
# ---------------------------------------------------------------------------


def _check_data(data_sample: torch.Tensor) -> torch.Tensor:
    if not isinstance(data_sample, torch.Tensor) or data_sample.ndim != 2:
        raise ValueError(
            "data_sample must be a 2-D tensor of shape (n_samples, n_features); "
            f"got {type(data_sample).__name__}"
            + (
                f" with shape {tuple(data_sample.shape)}"
                if isinstance(data_sample, torch.Tensor)
                else ""
            )
            + "."
        )
    if data_sample.shape[0] == 0:
        raise ValueError("data_sample has no rows.")
    return data_sample.detach()


def _per_sample_jacobian(
    model: nn.Module, weights: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ``(outputs, J)`` for one sample: outputs of shape ``(k,)`` and the
    Jacobian ``J`` of shape ``(k, d)`` with respect to the flattened weights.
    """
    out = model(x.unsqueeze(0)).reshape(-1)
    rows = []
    for i in range(out.shape[0]):
        (grad,) = torch.autograd.grad(out[i], weights, retain_graph=i < out.shape[0] - 1)
        rows.append(grad.reshape(-1).to(torch.float64))
    return out.detach().to(torch.float64), torch.stack(rows)


def fisher_information_matrix(model: nn.Module, data_sample: torch.Tensor) -> FisherSpectrum:
    """
    Fisher information matrix of the quantum weights at their current values.

    Parameters
    ----------
    model:
        A hybrid classifier (one logit per sample → Bernoulli likelihood) or
        an encoding layer (``n_qubits`` outputs → unit-variance Gaussian
        likelihood, i.e. the Gauss-Newton matrix).  See the module docstring.
    data_sample:
        Inputs of shape ``(n_samples, n_features)`` the expectation over ``x``
        is taken over.  For a classifier these go through its classical
        encoder; for a layer they are fed to the circuit directly.

    Returns
    -------
    FisherSpectrum

    Notes
    -----
    Cost is one forward pass and ``k`` backward passes per row, ``k`` being
    the number of outputs (1 for a classifier), plus one ``(d, d)``
    eigendecomposition.  The model is evaluated as it is (train or eval mode
    untouched, dropout included if active); put it in eval mode first for a
    deterministic answer.
    """
    _, weights, _, _ = _resolve_weights(model)
    X = _check_data(data_sample)
    d = weights.numel()
    fisher = torch.zeros(d, d, dtype=torch.float64)
    likelihood = ""
    for i in range(X.shape[0]):
        out, jac = _per_sample_jacobian(model, weights, X[i])
        if i == 0:
            # A classifier gives one logit; a layer gives n_qubits outputs.
            likelihood = "bernoulli" if out.shape[0] == 1 else "gaussian"
        if likelihood == "bernoulli":
            p = torch.sigmoid(out[0])
            fisher += (p * (1 - p)) * (jac.T @ jac)
        else:
            fisher += jac.T @ jac
    fisher /= X.shape[0]
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues = torch.linalg.eigvalsh(fisher).flip(0).clamp_min(0.0)
    return FisherSpectrum(
        layer_type=type(model).__name__,
        likelihood=likelihood,
        n_params=d,
        n_data=X.shape[0],
        matrix=fisher,
        eigenvalues=eigenvalues,
    )


def fisher_information_spectrum(model: nn.Module, data_sample: torch.Tensor) -> torch.Tensor:
    """
    Eigenvalues of :func:`fisher_information_matrix`, descending, ``float64``.

    The convenience form the issue asks for; use
    :func:`fisher_information_matrix` when the matrix or the metadata is
    wanted too.
    """
    return fisher_information_matrix(model, data_sample).eigenvalues


# ---------------------------------------------------------------------------
# Effective dimension
# ---------------------------------------------------------------------------


def effective_dimension_from_spectra(
    spectra: Sequence[torch.Tensor] | torch.Tensor,
    n_data: int,
    gamma: float = 1.0,
) -> float:
    """
    ``d_{γ,n}`` from Fisher eigenvalues at several parameter draws.

    Parameters
    ----------
    spectra:
        ``(n_theta_samples, d)`` tensor, or a sequence of length-``d``
        eigenvalue tensors, one per parameter draw.  Eigenvalues of the raw
        Fisher matrices; the trace normalisation is done here.
    n_data:
        ``n`` in the definition.  Must exceed 1 so that ``log n > 0``.
    gamma:
        ``γ ∈ (0, 1]``.

    Returns
    -------
    float
        The effective dimension; 0 if every spectrum is zero.

    Notes
    -----
    ``sqrt(det(I + κ F̂))`` is evaluated as ``exp(½ Σ_i log(1 + κ λ̂_i))`` and
    the average over draws with a log-sum-exp, so large ``κ`` does not
    overflow.
    """
    if n_data <= 1:
        raise ValueError(f"n_data must be > 1 (log n_data must be positive); got {n_data}.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1]; got {gamma}.")
    lam = torch.as_tensor(
        torch.stack(list(spectra)) if not isinstance(spectra, torch.Tensor) else spectra,
        dtype=torch.float64,
    )
    if lam.ndim != 2 or lam.shape[0] == 0:
        raise ValueError(
            f"spectra must have shape (n_theta_samples, d) with at least one draw; got {tuple(lam.shape)}."
        )
    lam = lam.clamp_min(0.0)
    n_draws, d = lam.shape
    mean_trace = lam.sum(dim=1).mean()
    if mean_trace <= 0.0:
        return 0.0
    normalised = lam * (d / mean_trace)  # F̂: E_θ[tr F̂] = d
    kappa = gamma * n_data / (2 * math.pi * math.log(n_data))
    half_log_det = 0.5 * torch.log1p(kappa * normalised).sum(dim=1)  # log sqrt(det(I + κ F̂))
    log_mean = torch.logsumexp(half_log_det, dim=0) - math.log(n_draws)
    return float(2 * log_mean / math.log(kappa))


def effective_dimension(
    model: nn.Module,
    data_sample: torch.Tensor,
    *,
    n_data: int | None = None,
    gamma: float = 1.0,
    n_theta_samples: int = 20,
    init: InitName | InitFn = "uniform",
    generator: torch.Generator | None = None,
) -> EffectiveDimensionResult:
    """
    Effective dimension (Abbas et al. 2021) of a model's quantum weights.

    The Fisher matrix is computed at ``n_theta_samples`` random draws of the
    quantum weights from ``init``, each over the rows of ``data_sample``, and
    the draws are combined with :func:`effective_dimension_from_spectra`.
    The weights are restored afterwards.

    Parameters
    ----------
    model:
        As for :func:`fisher_information_matrix`.
    data_sample:
        Inputs the expectation over ``x`` is taken over.
    n_data:
        ``n`` in the definition: the size of the data set the model is meant
        for, which sets the resolution ``κ``.  Default: the number of rows
        in ``data_sample``.  Use the same value when comparing architectures.
    gamma:
        ``γ ∈ (0, 1]``.  Default: 1.
    n_theta_samples:
        Parameter draws.  Default: 20.
    init:
        ``"uniform"`` over [0, 2π) (default, as in the paper),
        ``"restricted"``, ``"block_local"``, or a callable that fills the
        weight tensor in place.
    generator:
        Source of randomness for the draws; the global RNG is left untouched.

    Returns
    -------
    EffectiveDimensionResult
    """
    if n_theta_samples < 1:
        raise ValueError(f"n_theta_samples must be >= 1; got {n_theta_samples}.")
    X = _check_data(data_sample)
    n = X.shape[0] if n_data is None else n_data
    _, weights, n_qubits, n_layers = _resolve_weights(model)
    gen = generator if generator is not None else torch.Generator().manual_seed(0)
    init_name, init_fn = _make_init(init, n_qubits, n_layers, gen)

    original = weights.detach().clone()
    spectra = []
    try:
        for _ in range(n_theta_samples):
            with torch.no_grad():
                init_fn(weights)
            spectra.append(fisher_information_matrix(model, X).eigenvalues)
    finally:
        with torch.no_grad():
            weights.copy_(original)

    lam = torch.stack(spectra)
    d_eff = effective_dimension_from_spectra(lam, n, gamma)
    d = weights.numel()
    mean_trace = lam.sum(dim=1).mean()
    mean_spectrum = (
        (lam * (d / mean_trace)).mean(dim=0)
        if mean_trace > 0
        else torch.zeros(d, dtype=torch.float64)
    )
    return EffectiveDimensionResult(
        layer_type=type(model).__name__,
        init=init_name,
        n_params=d,
        n_data=n,
        gamma=gamma,
        n_theta_samples=n_theta_samples,
        effective_dimension=d_eff,
        normalized_effective_dimension=d_eff / d if d else 0.0,
        mean_normalized_spectrum=mean_spectrum,
    )
