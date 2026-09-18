# hqnn-forge

> **Parameter-efficient Hybrid Quantum Neural Networks for imbalanced tabular classification.**

`hqnn-forge` is a research-grade Python library that fuses **PennyLane** quantum circuits with **PyTorch** classical layers into end-to-end differentiable hybrid architectures optimised for NISQ-era hardware and binary fraud-detection workloads.

---

## Key Features

| Feature | Detail |
|---|---|
| **Barren-plateau-safe init** | Block-local restricted-variance initialisation (Cerezo et al. 2021) |
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

### Device backends

Every encoding layer and classifier takes a `device_name`. If the requested backend is not
installed or finds no usable hardware, the library falls back one step at a time, with a
`RuntimeWarning` at each step, along `requested → lightning.qubit → default.qubit`.

| `device_name` | What it is | Prerequisites |
|---|---|---|
| `default.qubit` | PennyLane's reference state-vector simulator (Python/NumPy) | None; always available |
| `lightning.qubit` | C++ state-vector simulator, CPU; adjoint differentiation | `pip install -e ".[lightning]"` (`pennylane-lightning`) |
| `lightning.gpu` | State-vector simulator on NVIDIA GPUs via cuQuantum (cuStateVec) | `pip install pennylane-lightning-gpu`; Linux, an NVIDIA GPU with compute capability ≥ 7.0, a CUDA 12 driver. The wheel pulls in `custatevec-cu12` |
| `lightning.kokkos` | State-vector simulator on Kokkos; OpenMP-parallel CPU on the PyPI wheel, CUDA or HIP GPUs when built from source | `pip install pennylane-lightning-kokkos` for the CPU build; see the [PennyLane-Lightning docs](https://docs.pennylane.ai/projects/lightning/) for a GPU build |

The GPU backends pay off at larger qubit counts or batch sizes; at the 8 qubits the library
targets, `lightning.qubit` is usually the fastest option. Both accelerated devices support the
same `diff_method="adjoint"` as `lightning.qubit`.

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

- Cerezo et al. (2021) — *Barren plateaus in quantum neural network training landscapes*
- McClean et al. (2018) — *Barren plateaus in quantum neural network training landscapes*
- Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- Bergholm et al. (2022) — *PennyLane: Automatic differentiation of hybrid quantum-classical computations*
