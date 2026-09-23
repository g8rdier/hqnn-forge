"""
hqnn_forge.diagnostics
======================
Tools for inspecting the quantum side of a hybrid model without reading the
circuit source: what the circuit costs on hardware, and later, how it trains.

Exported symbols
----------------
CircuitSummary     Frozen record: qubits, depth, gate counts, trainable parameters.
circuit_summary    Build a CircuitSummary from an encoding layer or a hybrid classifier.
draw_circuit       Text drawing of the same circuit, for logs and notebooks.
LOGICAL_GATE_SET   The gate names circuits are decomposed to before counting.
gradient_variance        Variance of the cost gradient over random weight draws.
gradient_variance_sweep  The same over a grid of qubit and layer counts.
GradientVarianceResult   Result of gradient_variance.
format_sweep             Text table of a sweep.
"""

from hqnn_forge.diagnostics.circuit import (
    LOGICAL_GATE_SET,
    CircuitSummary,
    circuit_summary,
    draw_circuit,
)
from hqnn_forge.diagnostics.gradients import (
    GradientVarianceResult,
    format_sweep,
    gradient_variance,
    gradient_variance_sweep,
)

__all__: list[str] = [
    "LOGICAL_GATE_SET",
    "CircuitSummary",
    "GradientVarianceResult",
    "circuit_summary",
    "draw_circuit",
    "format_sweep",
    "gradient_variance",
    "gradient_variance_sweep",
]
