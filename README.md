# hqnn-forge

[![Tests](https://github.com/g8rdier/hqnn-forge/actions/workflows/tests.yml/badge.svg)](https://github.com/g8rdier/hqnn-forge/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/g8rdier/hqnn-forge)](LICENSE)

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
     │              (the defaults; see the options below for the axis,
     │               the entangler and the ⟨Z_0⟩-only readout)
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
- `init_strategy="restricted"` (one σ for the whole circuit), `"block_local"` (σ narrowing
  with layer depth) or `"normal"` (plain `N(0, init_std²)`, `init_std=0.1` by default); see
  `hqnn_forge.initializers`.
- `embedding_rotation="X"` (default), `"Y"` or `"Z"`: the Pauli axis of the angle embedding
  (angle encoding only).
- `entangler="ring"` (default: CNOT ring then per-qubit `Rot`) or `"strongly_entangling"`
  (`qml.StronglyEntanglingLayers`: `Rot` first, then a CNOT ring whose range grows with the
  layer index).
- `readout="all"` (default: ⟨Z_i⟩ on every qubit) or `"first"` (⟨Z_0⟩ only, so the head reads
  a single number).
- `encoder_activation="tanh"` (default: `tanh(·)·π`, in (-π, π)) or `"sigmoid"`
  (`π·sigmoid(·)`, in (0, π)).
- `published_shnn()` on either class builds the configuration published in the thesis and in
  `hqnn-fraud-detection-benchmark`: 8 qubits, 2 layers, RY embedding, strongly-entangling
  ansatz, ⟨Z_0⟩ readout, sigmoid encoder, `N(0, 0.1²)` init — 122 trainable parameters for the
  serial model. Keyword arguments override it.
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
├── diagnostics/     Circuit depth, gate and parameter counts
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
- Grant et al. (2019) — *An initialization strategy for addressing barren plateaus in parametrized quantum circuits*
- Schuld et al. (2020) — *Circuit-centric quantum classifiers*
- Sim et al. (2019) — *Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms*
- Jones & Gacon (2020) — *Efficient calculation of gradients in classical simulations of variational quantum algorithms*
- Kandala et al. (2017) — *Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets*
- Havlíček et al. (2019) — *Supervised learning with quantum-enhanced feature spaces*
- Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- King & Zeng (2001) — *Logistic Regression in Rare Events Data*
- Bergholm et al. (2022) — *PennyLane: Automatic differentiation of hybrid quantum-classical computations*
