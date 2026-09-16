"""
hqnn_forge.utils.ablation
=========================
Switch off a hybrid model's quantum layer to measure what it contributes.

The question "is the circuit carrying signal, or would the classical layers
around it do as well on their own?" is answered by comparing a model with its
quantum layer against the same model with that layer replaced by a constant.
:func:`disable_quantum_layer` does the replacement for the duration of a
``with`` block, without retraining and without touching the weights.

The replacement output is ``fill`` (default 0.0) for every qubit.  0 is the
midpoint of the ⟨Z⟩ range [-1, 1], i.e. the readout of a maximally
uninformative qubit, so the head sees "no quantum information" rather than an
out-of-distribution value.  The circuit is not executed at all inside the
block, so an ablated evaluation is also much faster.

Example
-------
::

    from hqnn_forge.evaluation import find_optimal_threshold
    from hqnn_forge.utils import disable_quantum_layer

    full = find_optimal_threshold(y_val, model.predict_proba(X_val)).score
    with disable_quantum_layer(model):
        ablated = find_optimal_threshold(y_val, model.predict_proba(X_val)).score
    print(f"MCC with circuit {full:.3f}, without {ablated:.3f}")

For a fair ablation study, also train a model *from scratch* with the layer
disabled: a head trained alongside the circuit has adapted to its outputs,
so ablating after training measures dependence, not the best achievable
classical performance.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def disable_quantum_layer(model: nn.Module, fill: float = 0.0) -> Iterator[nn.Module]:
    """
    Replace ``model.quantum_layer``'s output with the constant ``fill``.

    Inside the block the layer returns a tensor of shape
    ``(*batch_dims, n_qubits)`` filled with ``fill``, in the input's dtype and
    device, without running the circuit.  The output therefore carries no
    gradient to the quantum weights or to anything upstream of the layer.  The
    original ``forward`` is restored on exit, including when the block raises.

    Training inside the block is allowed and trains only the parameters that
    still influence the loss.

    Parameters
    ----------
    model:
        A hybrid classifier with a ``quantum_layer`` attribute whose layer has
        ``n_qubits``.
    fill:
        Constant output per qubit.  Must lie in [-1, 1], the range of a Pauli-Z
        expectation.  Default: 0.0.

    Yields
    ------
    nn.Module
        The disabled quantum layer.

    Raises
    ------
    TypeError
        If ``model`` has no suitable ``quantum_layer``.
    ValueError
        If ``fill`` is outside [-1, 1].
    RuntimeError
        If the layer is already disabled (nested use on the same model).
    """
    layer = getattr(model, "quantum_layer", None)
    n_qubits = getattr(layer, "n_qubits", None)
    if not isinstance(layer, nn.Module) or not isinstance(n_qubits, int):
        raise TypeError(
            f"disable_quantum_layer expects a model with a quantum_layer attribute; "
            f"got {type(model).__name__}."
        )
    if not -1.0 <= fill <= 1.0:
        raise ValueError(f"fill must lie in [-1, 1], the range of <Z>; got {fill}.")
    if "forward" in vars(layer):
        raise RuntimeError(
            f"{type(layer).__name__}.forward is already overridden on this instance; "
            f"disable_quantum_layer cannot be nested."
        )

    def constant_forward(x: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (*x.shape[:-1], n_qubits), fill, dtype=x.dtype, device=x.device
        )

    # nn.Module.__call__ dispatches to self.forward, so an instance attribute
    # shadows the class method for this layer only.
    layer.forward = constant_forward  # type: ignore[method-assign]
    try:
        yield layer
    finally:
        del layer.forward
