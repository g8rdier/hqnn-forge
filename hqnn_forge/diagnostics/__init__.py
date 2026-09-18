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
count_inert_parameters   Trainable gate parameters that can never reach a measurement.
"""

from hqnn_forge.diagnostics.circuit import (
    LOGICAL_GATE_SET,
    CircuitSummary,
    circuit_summary,
    count_inert_parameters,
    draw_circuit,
)

__all__: list[str] = [
    "LOGICAL_GATE_SET",
    "CircuitSummary",
    "circuit_summary",
    "count_inert_parameters",
    "draw_circuit",
]
