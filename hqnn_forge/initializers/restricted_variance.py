"""
hqnn_forge.initializers.restricted_variance
=============================================
Barren-plateau-aware weight initialisation for variational quantum circuits.

Theory
------
In deep variational quantum circuits initialised with uniform random weights
(i.e. a 2-design), the gradient variance decays *exponentially* in the number
of qubits n and layers L:

    Var[∂L/∂θ] ∝ 2^{-n}   (global cost, McClean et al. 2018)

This makes training effectively impossible beyond ~10 qubits with naive init.

**Block-local restricted-variance initialisation** (Cerezo et al. 2021) mitigates
this by keeping individual rotation angles small — drawn from N(0, σ²) with σ
chosen to preserve O(1) gradient variance at initialisation:

    σ = π / sqrt(n_qubits * n_layers)

This ensures that at t=0 the circuit is "close to the identity" so that local
cost-function gradients remain polynomial.

Functions
---------
restricted_normal_init_     In-place; fills a tensor with restricted-normal values.
block_local_init_           Fills each block (layer slice) independently.

References
----------
* McClean et al. (2018) "Barren plateaus in quantum neural network training
  landscapes", Nature Communications 9, 4812.
* Cerezo et al. (2021) "Cost function dependent barren plateaus in shallow
  parametrized quantum circuits", Nature Communications 12, 1791.
* Grant et al. (2019) "An initialization strategy for addressing barren
  plateaus in parametrized quantum circuits", Quantum 3, 214.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# In-place initialiser: single call, shared σ across all parameters
# ---------------------------------------------------------------------------

def restricted_normal_init_(
    tensor: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    scale: float = math.pi,
) -> torch.Tensor:
    """
    Fill *tensor* **in-place** with values drawn from N(0, σ²) where

        σ = scale / sqrt(n_qubits * n_layers)

    This keeps the initial circuit "close to the identity" and preserves
    O(1) gradient variance for local cost functions.

    Parameters
    ----------
    tensor:
        The weight tensor to initialise.  Typically the ``weights`` parameter
        of a ``QuantumEncodingLayer``, shape ``(n_layers, n_qubits, 3)``.
    n_qubits:
        Number of qubits in the circuit.
    n_layers:
        Number of variational layers.
    scale:
        Numerator of the standard deviation formula.  Default: π.
        Adjust downward (e.g. π/2) for deeper circuits if gradients still vanish.

    Returns
    -------
    torch.Tensor
        The initialised tensor (modified in-place and returned for chaining).

    Raises
    ------
    ValueError
        If ``n_qubits`` or ``n_layers`` is less than 1.

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.initializers import restricted_normal_init_
    >>> w = torch.empty(2, 8, 3)  # (n_layers=2, n_qubits=8, 3 Euler angles)
    >>> restricted_normal_init_(w, n_qubits=8, n_layers=2)
    >>> w.std().item()  # ≈ π / sqrt(16) ≈ 0.785
    """
    if n_qubits < 1 or n_layers < 1:
        raise ValueError(
            f"n_qubits and n_layers must be ≥ 1; "
            f"got n_qubits={n_qubits}, n_layers={n_layers}."
        )

    std = scale / math.sqrt(n_qubits * n_layers)
    with torch.no_grad():
        tensor.normal_(mean=0.0, std=std)
    return tensor


# ---------------------------------------------------------------------------
# Block-local variant: each layer initialised with its own restricted σ
# ---------------------------------------------------------------------------

def block_local_init_(
    tensor: torch.Tensor,
    n_qubits: int,
    scale: float = math.pi,
) -> torch.Tensor:
    """
    Fill *tensor* **in-place** with per-block restricted-normal values.

    For each layer ℓ the standard deviation is computed using *only that
    layer's* depth contribution:

        σ_ℓ = scale / sqrt(n_qubits * (ℓ + 1))

    This is the "layer-wise" variant recommended by Grant et al. (2019) for
    deeper circuits where a global σ may be too aggressive.

    Parameters
    ----------
    tensor:
        Weight tensor of shape ``(n_layers, n_qubits, 3)`` or any shape
        where ``dim 0`` indexes layers.
    n_qubits:
        Number of qubits.
    scale:
        Numerator for std computation.  Default: π.

    Returns
    -------
    torch.Tensor
        The initialised tensor (in-place).

    Examples
    --------
    >>> import torch
    >>> from hqnn_forge.initializers import block_local_init_
    >>> w = torch.empty(4, 8, 3)  # 4-layer circuit
    >>> block_local_init_(w, n_qubits=8)
    >>> # Layer 0 has the largest variance; layer 3 the smallest.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be ≥ 1; got {n_qubits}.")

    n_layers: int = tensor.shape[0]
    with torch.no_grad():
        for layer_idx in range(n_layers):
            std = scale / math.sqrt(n_qubits * (layer_idx + 1))
            tensor[layer_idx].normal_(mean=0.0, std=std)
    return tensor


# ---------------------------------------------------------------------------
# Convenience: apply to all VQC parameters in an nn.Module
# ---------------------------------------------------------------------------

def apply_restricted_init(
    module: nn.Module,
    n_qubits: int,
    n_layers: int,
    *,
    block_local: bool = False,
    scale: float = math.pi,
) -> None:
    """
    Apply barren-plateau-safe initialisation to **all parameters** in *module*
    whose shape starts with ``(n_layers, ...)``.

    This is a convenience wrapper; for fine-grained control call
    :func:`restricted_normal_init_` or :func:`block_local_init_` directly.

    Parameters
    ----------
    module:
        The PyTorch module (e.g. a ``QuantumEncodingLayer`` or the full
        ``HybridBinaryClassifier``) whose quantum weight parameters should be
        initialised.
    n_qubits:
        Number of qubits.
    n_layers:
        Number of variational layers.
    block_local:
        If ``True``, use :func:`block_local_init_` (per-layer σ).
        If ``False`` (default), use :func:`restricted_normal_init_` (global σ).
    scale:
        Standard deviation numerator.  Default: π.
    """
    for name, param in module.named_parameters():
        if param.ndim >= 1 and param.shape[0] == n_layers:
            if block_local:
                block_local_init_(param.data, n_qubits=n_qubits, scale=scale)
            else:
                restricted_normal_init_(
                    param.data, n_qubits=n_qubits, n_layers=n_layers, scale=scale
                )
            # Do not update running stats / non-trainable buffers
