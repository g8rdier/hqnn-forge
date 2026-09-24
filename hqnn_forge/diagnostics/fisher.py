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

* A **hybrid classifier** (a model with a ``quantum_layer``, whose
  ``forward`` returns one logit ``z`` per sample) is the Bernoulli model ``p(y=1|x) = σ(z)``.  Then
  ``E_y[∇ log p ∇ log pᵀ] = σ(z)(1 − σ(z)) ∇z ∇zᵀ``, exactly, so no labels
  are needed and nothing is sampled.
* An **encoding layer** (``forward`` returns its expectation values, one
  or ``n_qubits`` of them depending on ``readout``) has no likelihood of its
  own.  It is treated as a Gaussian observation
  model with unit variance around its outputs, for which the Fisher matrix
  is ``E_x[ Jᵀ J ]`` with ``J = ∂ outputs / ∂ θ``: the Gauss-Newton matrix.
  This is the natural information matrix of a regression-style readout and
  what "the Fisher spectrum of the circuit" means in practice.

The likelihood follows from which of the two was passed, not from the output
width: a one-qubit layer, or one with ``readout="first"``, still returns an
expectation value, not a logit.

Only the quantum layer's weights are differentiated, so the spectrum is that
of the circuit's parameters, whatever classical layers sit around it.  They
are differentiated even when frozen (``requires_grad=False``); the flag is
restored afterwards.

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
redundant.

The formula divides by ``log κ``, so it only means something once ``κ`` is
comfortably above 1: at ``κ ≤ 1`` it changes sign or divides by zero, and
just above 1 it explodes (``F̂ = I`` gives ``d · log(1 + κ) / log κ``, which
is ``26 d`` at ``κ = 1.02``).  Both functions therefore require
``κ ≥ e``, i.e. ``log κ ≥ 1``, where that identity-spectrum value is at most
``d · log(1 + e) ≈ 1.31 d``; with ``γ = 1`` this means ``n ≥ 74``.  ``n`` is
the size of the data set the model is meant for, not of ``data_sample``, so
pass it explicitly when estimating from a small sample.  Even then the
estimate can exceed ``d`` slightly; this is a property of the definition,
not a bug.

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
from dataclasses import dataclass, field
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
    # Tensor fields are left out of the generated __eq__, which would otherwise
    # call bool() on an element-wise tensor comparison and raise.
    matrix: torch.Tensor = field(compare=False)
    eigenvalues: torch.Tensor = field(compare=False)

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
        return _normalise_spectra(self.eigenvalues.unsqueeze(0))[0]

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
    mean_normalized_spectrum: torch.Tensor = field(compare=False)

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
        if out[i].requires_grad:
            # An output that does not reach the weights (an ablated quantum
            # layer) contributes a zero row rather than an autograd error.
            (grad,) = torch.autograd.grad(
                out[i],
                weights,
                retain_graph=i < out.shape[0] - 1,
                allow_unused=True,
                materialize_grads=True,
            )
        else:
            grad = torch.zeros_like(weights)
        rows.append(grad.reshape(-1).to(torch.float64))
    return out.detach().to(torch.float64), torch.stack(rows)


def fisher_information_matrix(model: nn.Module, data_sample: torch.Tensor) -> FisherSpectrum:
    """
    Fisher information matrix of the quantum weights at their current values.

    Parameters
    ----------
    model:
        A hybrid classifier (one logit per sample → Bernoulli likelihood) or
        an encoding layer (its expectation values → unit-variance Gaussian
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
    layer, weights, _, _ = _resolve_weights(model, caller="fisher_information_matrix")
    X = _check_data(data_sample)
    # Decided by what was passed, not by the output width: a one-qubit or
    # readout="first" layer also returns a single value, which is not a logit.
    likelihood = "gaussian" if layer is model else "bernoulli"
    d = weights.numel()
    fisher = torch.zeros(d, d, dtype=torch.float64)
    was_frozen = not weights.requires_grad
    weights.requires_grad_(True)
    try:
        for i in range(X.shape[0]):
            out, jac = _per_sample_jacobian(model, weights, X[i])
            if likelihood == "bernoulli":
                if out.shape[0] != 1:
                    raise ValueError(
                        f"fisher_information_matrix expects a classifier to return one logit "
                        f"per sample; {type(model).__name__} returned {out.shape[0]}."
                    )
                p = torch.sigmoid(out[0])
                fisher += (p * (1 - p)) * (jac.T @ jac)
            else:
                fisher += jac.T @ jac
    finally:
        if was_frozen:
            weights.requires_grad_(False)
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


def _kappa(n_data: int, gamma: float) -> float:
    return gamma * n_data / (2 * math.pi * math.log(n_data))


def _min_n_data(gamma: float) -> int:
    """Smallest ``n`` with ``κ(n) ≥ e``; ``κ`` increases with ``n`` for ``n > e``."""
    hi = 3
    while _kappa(hi, gamma) < math.e:
        hi *= 2
    lo = hi // 2
    while lo < hi:
        mid = (lo + hi) // 2
        if _kappa(mid, gamma) < math.e:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _check_kappa(n_data: int, gamma: float) -> float:
    """Validate ``n_data`` and ``gamma`` and return ``κ``; see the module docstring."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1]; got {gamma}.")
    if n_data <= 1:
        raise ValueError(f"n_data must be > 1 (log n_data must be positive); got {n_data}.")
    kappa = _kappa(n_data, gamma)
    if kappa < math.e:
        raise ValueError(
            f"n_data={n_data} with gamma={gamma} gives κ = {kappa:.3g}; the effective "
            f"dimension needs κ ≥ e, i.e. n_data ≥ {_min_n_data(gamma)} at this gamma. "
            "n_data is the size of the data set the model is meant for; pass it "
            "explicitly when data_sample is smaller."
        )
    return kappa


def _normalise_spectra(lam: torch.Tensor) -> torch.Tensor:
    """
    ``F̂`` eigenvalues: ``(n_theta_samples, d)`` spectra scaled by ``d / E_θ[tr F]``
    so that their mean trace is ``d``; zeros if every spectrum is zero.
    """
    lam = lam.to(torch.float64).clamp_min(0.0)
    mean_trace = lam.sum(dim=1).mean()
    if mean_trace <= 0.0:
        return torch.zeros_like(lam)
    return lam * (lam.shape[1] / mean_trace)


def _effective_dimension(normalised: torch.Tensor, kappa: float) -> float:
    """``d_{γ,n}`` from :func:`_normalise_spectra` output."""
    half_log_det = 0.5 * torch.log1p(kappa * normalised).sum(dim=1)  # log sqrt(det(I + κ F̂))
    log_mean = torch.logsumexp(half_log_det, dim=0) - math.log(normalised.shape[0])
    return float(2 * log_mean / math.log(kappa))


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
        ``n`` in the definition.  Must be large enough that
        ``κ = γ n / (2π log n) ≥ e`` (``n ≥ 74`` at ``γ = 1``); see the module
        docstring.
    gamma:
        ``γ ∈ (0, 1]``.

    Returns
    -------
    float
        The effective dimension, ``≥ 0``; 0 if every spectrum is zero.

    Notes
    -----
    ``sqrt(det(I + κ F̂))`` is evaluated as ``exp(½ Σ_i log(1 + κ λ̂_i))`` and
    the average over draws with a log-sum-exp, so large ``κ`` does not
    overflow.
    """
    kappa = _check_kappa(n_data, gamma)
    lam = torch.as_tensor(
        torch.stack(list(spectra)) if not isinstance(spectra, torch.Tensor) else spectra,
        dtype=torch.float64,
    )
    if lam.ndim != 2 or lam.shape[0] == 0:
        raise ValueError(
            f"spectra must have shape (n_theta_samples, d) with at least one draw; got {tuple(lam.shape)}."
        )
    return _effective_dimension(_normalise_spectra(lam), kappa)


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
        Must give ``κ ≥ e`` (``n ≥ 74`` at ``γ = 1``); this is checked before
        any Fisher matrix is computed.
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
    kappa = _check_kappa(n, gamma)
    _, weights, n_qubits, n_layers = _resolve_weights(model, caller="effective_dimension")
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

    normalised = _normalise_spectra(torch.stack(spectra))
    d_eff = _effective_dimension(normalised, kappa)
    d = weights.numel()
    return EffectiveDimensionResult(
        layer_type=type(model).__name__,
        init=init_name,
        n_params=d,
        n_data=n,
        gamma=gamma,
        n_theta_samples=n_theta_samples,
        effective_dimension=d_eff,
        normalized_effective_dimension=d_eff / d if d else 0.0,
        mean_normalized_spectrum=normalised.mean(dim=0),
    )
