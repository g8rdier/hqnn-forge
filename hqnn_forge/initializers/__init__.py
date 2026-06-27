"""
hqnn_forge.initializers
=======================
Barren-plateau-aware weight initialisation strategies for variational quantum circuits.

Exported symbols
----------------
restricted_normal_init_     In-place initialiser; variance scaled as σ² ∝ 1/(n_qubits * n_layers).
block_local_init_           Applies restricted_normal_init_ independently per circuit block/layer.
"""

from hqnn_forge.initializers.restricted_variance import (
    block_local_init_,
    restricted_normal_init_,
)

__all__: list[str] = [
    "restricted_normal_init_",
    "block_local_init_",
]
