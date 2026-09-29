"""
hqnn_forge._resolve
===================
Find the encoding layer inside the object a public function was given.

Every diagnostic or wrapper that takes "a model or a layer" -- circuit_summary,
gradient_variance, the Fisher diagnostics, apply_depolarizing_noise, the
quantum kernels -- needs the same walk: unwrap a hybrid classifier's
``quantum_layer``, then check for a ``qlayer`` TorchLayer and an integer
``n_qubits``.  Each used to carry its own copy, and the copies drifted (#182):
one skipped the ``nn.Module`` check, and each hardcoded a function name into
its error message, which then named the wrong function when a second caller
reused it.  Callers keep only what is specific to them on top of the result.
"""

from __future__ import annotations

import pennylane as qml
import torch.nn as nn


def resolve_encoding_layer(
    target: object, caller: str, *, allow_model: bool = True
) -> tuple[nn.Module, qml.qnn.TorchLayer, int]:
    """
    ``(layer, qlayer, n_qubits)`` for the encoding layer ``target`` is or holds.

    Parameters
    ----------
    target:
        An encoding layer (an ``nn.Module`` with a ``qlayer`` TorchLayer and an
        integer ``n_qubits``) or, with ``allow_model``, a hybrid classifier
        holding one as ``quantum_layer``.
    caller:
        The public function's name, for the error message.
    allow_model:
        Unwrap ``quantum_layer``.  Off for a caller defined by the encoder
        alone, which refuses a classifier with a message saying why.

    Raises
    ------
    TypeError
        If no encoding layer is found.
    """
    if not allow_model and hasattr(target, "quantum_layer"):
        raise TypeError(
            f"{caller} expects an encoding layer, not a hybrid classifier "
            f"({type(target).__name__}): the classifier's classical encoder runs before "
            f"its quantum layer, so the result for its quantum_layer on raw inputs would "
            f"describe a different feature map.  Pass model.quantum_layer, with inputs "
            f"already encoded, if that is what you mean."
        )
    layer = getattr(target, "quantum_layer", target) if allow_model else target
    qlayer = getattr(layer, "qlayer", None)
    n_qubits = getattr(layer, "n_qubits", None)
    if (
        not isinstance(layer, nn.Module)
        or not isinstance(qlayer, qml.qnn.TorchLayer)
        or isinstance(n_qubits, bool)
        or not isinstance(n_qubits, int)
    ):
        expected = (
            "an encoding layer or a hybrid classifier with a quantum_layer attribute"
            if allow_model
            else "an encoding layer"
        )
        raise TypeError(
            f"{caller} expects {expected} (QuantumEncodingLayer, IQPEncodingLayer, "
            f"AmplitudeEncodingLayer, DataReuploadingLayer); got {type(target).__name__}."
        )
    return layer, qlayer, n_qubits
