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
| **Two hybrid topologies** | Serial `HybridBinaryClassifier` and parallel `ParallelHybridClassifier` (classical MLP branch ‖ quantum branch), with angle or IQP encoding |

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

## Architecture

Two hybrid topologies share the same building blocks. Both return a raw logit of shape
`(batch, 1)`: apply `torch.sigmoid` for a probability, or pass it straight to `FocalLoss`.

### `HybridBinaryClassifier` (serial)

```
Input (batch, n_input_features)
     │
     ▼
Classical encoder   Linear(n_input_features → n_qubits) + Tanh, scaled by π into (-π, π)
     │
     ▼
Quantum layer       AngleEmbedding RX(x_i) on qubit i   (or IQP embedding)
     │              n_layers × [ CNOT ring → per-qubit Rot(φ, θ, ω) ]
     │              → ⟨Z_i⟩ for every qubit, shape (batch, n_qubits)
     ▼
Classical head      Linear(n_qubits → 1)
     │
     ▼
Raw logit (batch, 1)
```

### `ParallelHybridClassifier` (parallel)

```
Input (batch, n_input_features)
     ├───────────────────────────────────┐
     ▼                                   ▼
Classical branch                    Classical encoder   Linear(→ n_qubits) + Tanh, × π
Linear → ReLU → Linear → ReLU            │
→ (batch, classical_hidden_dim)          ▼
     │                              Quantum layer       same circuit as the serial model
     │                                   │              → ⟨Z_i⟩, shape (batch, n_qubits)
     └────────────────┬──────────────────┘
                      ▼
                Concatenate   (batch, classical_hidden_dim + n_qubits)
                      │
                      ▼
                Classical head   Linear(→ 1)
                      │
                      ▼
                Raw logit (batch, 1)
```

The parallel model asks whether added classical capacity can substitute for, or extend, what
the quantum layer contributes: compare `count_parameters()` across the two at equal `n_qubits`
and `n_layers`.

Options shared by both models:

- `encoding_type="angle"` (default) or `"iqp"` (Havlíček-style feature map with pairwise
  `x_i x_j` phases).
- `init_strategy="restricted"` (one σ for the whole circuit) or `"block_local"` (σ narrowing
  with layer depth); see `hqnn_forge.initializers`.
- `use_classical_encoder=False` to feed features already scaled into (-π, π), for example from
  `PCANormalizer(scale_to_pi=True)`, straight into the circuit. `n_input_features` must then
  equal `n_qubits`.
- `dropout_p` on the features entering the head, and `predict_proba` / `predict`, which always
  run in eval mode.

---

## Folder Structure

```
hqnn_forge/
├── encoding/        Quantum feature maps (angle embedding, IQP embedding)
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
