"""
hqnn_forge.circuits
===================
Reusable, device-agnostic VQC ansatz primitives.

These functions are *pure quantum functions* — they are intended to be called
**inside** a QNode context (i.e. called within a function decorated with
``@qml.qnode``).  They apply in-place gate sequences and return nothing.

Exported symbols
----------------
strongly_entangling_layer   CNOT ring + per-qubit Rot(φ, θ, ω) block.
hardware_efficient_layer    CZ ladder + per-qubit RY(θ) block (lower CNOT depth).
"""

from hqnn_forge.circuits.strongly_entangling import strongly_entangling_layer
from hqnn_forge.circuits.hardware_efficient import hardware_efficient_layer

__all__: list[str] = [
    "strongly_entangling_layer",
    "hardware_efficient_layer",
]
