# hqnn-forge

> **Parameter-efficient Hybrid Quantum Neural Networks for imbalanced tabular classification.**

`hqnn-forge` is a research-grade Python library that fuses **PennyLane** quantum circuits with **PyTorch** classical layers into end-to-end differentiable hybrid architectures optimised for NISQ-era hardware and binary fraud-detection workloads.

---

## Key Features

| Feature | Detail |
|---|---|
| **Barren-plateau-aware init** | Small-angle Gaussian initialisation: global σ = π/√(n·L), or a per-layer schedule (this library's own heuristics — see the module docstring for what they do and do not guarantee) |
| **Adjoint differentiation** | Exact gradients via `lightning.qubit` — no finite-difference approximation |
| **Custom angle encoding** | 8-qubit angle-embedding feature map with strongly-entangled VQC ansatz |
| **Imbalance-robust losses** | Focal Loss & inverse-frequency weighted BCE |
| **Pure-NumPy pre-processing** | PCA + standardisation without scikit-learn runtime dependency |

---

## Installation

```bash
pip install -e ".[lightning,dev]"
```

Adjoint differentiation needs `pennylane-lightning`, which the `lightning` extra above
installs. To add it to an existing install:

```bash
pip install -e ".[lightning]"
```

---

## Quick Start

```python
import torch
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.utils  import FocalLoss

model = HybridBinaryClassifier(n_input_features=8, n_qubits=8, n_layers=2)
loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

x = torch.randn(16, 8)          # batch of 16 samples, 8 PCA features
y = torch.randint(0, 2, (16,)).float()

logits = model(x)
loss   = loss_fn(logits.squeeze(), y)
loss.backward()
```

See `examples/quick_start.py` for a full training loop on a synthetic imbalanced dataset.

---

## Folder Structure

```
hqnn_forge/
├── encoding/        Quantum feature maps (angle embedding, IQP placeholder)
├── circuits/        Reusable VQC ansatz primitives
├── initializers/    Barren-plateau-aware weight initialisation
├── preprocessing/   Classical PCA + normalisation (no sklearn runtime dep)
├── models/          Full hybrid architectures
└── utils/           Imbalance-robust losses and helpers
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the issue/branch/PR workflow, commit conventions,
and versioning policy this project follows.

---

## References

- Cerezo et al. (2021) — *Cost function dependent barren plateaus in shallow parametrized quantum circuits*
- McClean et al. (2018) — *Barren plateaus in quantum neural network training landscapes*
- Zhang et al. (2022) — *Escaping from the barren plateau via Gaussian initializations in deep variational quantum circuits*
- Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- Bergholm et al. (2022) — *PennyLane: Automatic differentiation of hybrid quantum-classical computations*
