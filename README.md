<h1><img src="assets/social-preview.png" alt="hqnn-forge" width="640"></h1>

[![Tests](https://github.com/g8rdier/hqnn-forge/actions/workflows/tests.yml/badge.svg)](https://github.com/g8rdier/hqnn-forge/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/g8rdier/hqnn-forge)](LICENSE)

> **Test whether a small quantum layer earns its parameters on imbalanced binary tabular data.**

`hqnn-forge` is a research library for hybrid quantum-classical classifiers: **PennyLane**
circuits inside **PyTorch** models, trained end to end. It is built around one question: on
an imbalanced binary classification problem, does a small quantum layer add enough per
parameter to justify it? The library brings the parts needed to answer it on your own data:
hybrid models, a scikit-learn estimator, imbalance-robust losses, stratified CV with SMOTE,
threshold search, MCC per thousand parameters, a paired Wilcoxon test, ablation of the quantum
layer, and circuit diagnostics.

**Scope.** Binary classification on imbalanced tabular data, or data made tabular by a
pretrained embedding (see [Non-tabular data](#non-tabular-data-precomputed-embeddings)). The
estimator, losses, thresholds and metrics are built for binary targets;
`MulticlassHybridClassifier` covers multiclass targets at the model level only. End-to-end
image, text or time-series pipelines are out of scope.

---

## Key Features

| Feature | Detail |
|---|---|
| **Small-angle init** | Gaussian initialisation: global σ = π/√(n·L), or a per-layer schedule σ_ℓ = π/√(n·(L+ℓ)) that starts at the global σ and narrows by up to √2 towards the last layer (this library's own heuristics, in the spirit of Zhang et al. 2022). Measured with `hqnn_forge.diagnostics.gradient_variance` on a 2-layer circuit with a ⟨Z_0⟩ cost: no gain over uniform init for inputs spread over (−π, π), which is what both classifiers feed the circuit, and a gain growing from 1.1x to 1.75x between 4 and 8 qubits only near zero input. Over (−π, π) the variance falls ~3x per two qubits under either init — see the module docstring |
| **Adjoint differentiation** | Exact gradients via `lightning.qubit` — no finite-difference approximation |
| **Custom angle encoding** | Angle-embedding feature map (8 qubits by default) with a CNOT-ring VQC ansatz; strongly-entangling and brickwork entanglers are options |
| **Imbalance-robust losses** | Focal Loss & inverse-frequency weighted BCE |
| **Pure-NumPy pre-processing** | PCA + standardisation without scikit-learn runtime dependency |
| **Three hybrid topologies** | Serial `HybridBinaryClassifier`, parallel `ParallelHybridClassifier` (classical MLP branch ‖ quantum branch) and multiclass `MulticlassHybridClassifier` (softmax or one-vs-rest heads on a shared quantum layer), with angle or IQP encoding |
| **Data-driven decision threshold** | `find_optimal_threshold` picks the threshold that maximises MCC, F1 or balanced accuracy on validation probabilities, instead of the default 0.5 that is rarely the right operating point on imbalanced data |
| **Quantum ablation** | `disable_quantum_layer` replaces a trained model's quantum-layer output with a constant for the duration of a `with` block, so re-scoring a `ParallelHybridClassifier` shows how much the circuit adds to its classical branch (in the serial `HybridBinaryClassifier` the circuit is the only path, so the ablated model is a constant predictor) |
| **Checkpoints** | `save_checkpoint` / `load_checkpoint` store a classifier's class, constructor arguments and weights, and rebuild the model from that file; a file written by a different `hqnn_forge` version is refused unless `allow_version_mismatch=True` |

---

## Installation

To use the package, install it from a clone of the repository with pip:

```bash
pip install -e ".[lightning]"
```

The extras add optional parts; combine them as needed, e.g. `".[lightning,sklearn]"`:

| Extra | Installs | Needed for |
|---|---|---|
| `lightning` | `pennylane-lightning` | the `lightning.qubit` backend and adjoint differentiation, the library defaults |
| `sklearn` | `scikit-learn` | the scikit-learn estimator in `hqnn_forge.sklearn` |
| `examples` | `scikit-learn`, `matplotlib` | the scripts in `examples/` and the plots in `hqnn_forge.evaluation` |
| `dev` | test and lint tools | development; see [Development Setup](#development-setup) |

pip installs the newest versions that `pyproject.toml` allows. To work on the project in the
environment CI tests against, use the uv setup under [Development Setup](#development-setup).

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

## Quick Start: your own data

`HybridClassifierEstimator` trains a hybrid model on any binary `X, y` through the usual
scikit-learn `fit` / `predict` / `predict_proba`, so it also works in `Pipeline`,
`cross_val_score` and `GridSearchCV`. It needs the `sklearn` extra:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hqnn_forge.evaluation import matthews_corrcoef, parameter_efficiency
from hqnn_forge.sklearn import HybridClassifierEstimator

# Any imbalanced binary X, y; here 2,000 rows, 20 features, 5% positives
X, y = make_classification(n_samples=2000, n_features=20, weights=[0.95], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

clf = make_pipeline(
    StandardScaler(),
    HybridClassifierEstimator(
        n_qubits=4,
        n_layers=2,
        max_epochs=20,
        batch_size=8,
        validation_fraction=0.2,
        random_state=0,
    ),
)
clf.fit(X_train, y_train)  # about a minute on a laptop CPU

mcc = matthews_corrcoef(y_test, clf.predict(X_test))
print(f"test MCC {mcc:.3f}")
print(f"MCC per 1,000 parameters {parameter_efficiency(clf[-1].model_, mcc):.2f}")
```

The model's classical encoder (`Linear` + `tanh`, scaled by π) maps any number of features
onto the qubits, so the input width is free. With `validation_fraction` set, a stratified
share of the training data drives early stopping and picks the decision threshold that
`predict` uses. On a dataset this small, a batch size well below the default 64 gives the
optimiser enough steps in 20 epochs; with the default, early stopping often ends the run
before the model has learnt anything. MCC, not accuracy, is the metric here: with 5%
positives, predicting the majority class alone is 95% accurate.

### Non-tabular data: precomputed embeddings

Images, text or time series can be used through embeddings from any pretrained model. Reduce
the embeddings to `n_qubits` dimensions and pass `use_classical_encoder=False`, so the quantum
layer reads them directly:

```python
from hqnn_forge.preprocessing import PCANormalizer
from hqnn_forge.sklearn import HybridClassifierEstimator

# emb_train, emb_test: (n_samples, d) arrays from a pretrained model; y_train: 0/1 labels
pca = PCANormalizer(n_components=8)  # centre, keep 8 components, standardise them, tanh(·)·π
Z_train = pca.fit_transform(emb_train).numpy()
Z_test = pca.transform(emb_test).numpy()

clf = HybridClassifierEstimator(n_qubits=8, use_classical_encoder=False)
clf.fit(Z_train, y_train)
proba = clf.predict_proba(Z_test)[:, 1]
```

Without the encoder, each input value is used unscaled as a rotation angle, so it has to lie
in (−π, π) already. Values outside that range wrap around modulo 2π, and distant inputs can
land on the same angle. `PCANormalizer` keeps them inside with `tanh(·)·π`. The input width
must equal `n_qubits`, and the model refuses anything else.

### Worked example: credit-card fraud

`hqnn_forge.data.load_credit_card_fraud` loads the Kaggle Credit Card Fraud Detection dataset
(284,807 transactions, 492 frauds, 0.17% positive), the benchmark the library was first
developed on. The CSV is not redistributable. Download it once with the Kaggle CLI
(`kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip`), or pass
`download=True`:

```python
from hqnn_forge.data import load_credit_card_fraud

data = load_credit_card_fraud()  # data/raw/creditcard.csv, or $HQNN_FORGE_DATA
X, y = data.X, data.y  # (284807, 30) float64, (284807,) int64
```

### The models directly, in PyTorch

```python
import torch
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.utils import FocalLoss

model = HybridBinaryClassifier(n_input_features=8, n_qubits=8, n_layers=2)
loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

x = torch.randn(16, 8)  # batch of 16 samples, 8 PCA features
y = torch.randint(0, 2, (16,)).float()

logits = model(x)
loss = loss_fn(logits.squeeze(), y)
loss.backward()
```

See `examples/quick_start.py` for a full training loop on a synthetic imbalanced dataset.

---

## Architecture

Three hybrid topologies share the same building blocks. The two binary ones return a raw
logit of shape `(batch, 1)`: apply `torch.sigmoid` for a probability, or pass it straight to
`FocalLoss`. The multiclass one returns `(batch, n_classes)` logits.

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

### `MulticlassHybridClassifier` (multiclass)

The serial trunk with `n_classes` heads, `Linear(n_qubits → n_classes)`, all reading the same
quantum layer, so the quantum parameter count does not depend on `n_classes`. Its forward pass
returns raw logits `(batch, n_classes)`.

- `strategy="softmax"` (default): `predict_proba` is the softmax over classes; train with
  `nn.CrossEntropyLoss`.
- `strategy="one_vs_rest"`: each head is one class against the rest, and `predict_proba` is the
  per-class sigmoid normalised to sum to one; train with `nn.BCEWithLogitsLoss` on
  `model.one_hot(y)`.

`predict` returns the argmax of the logits as `torch.long` labels. It does not follow the
binary `predict(x, threshold)` contract of `BinaryClassifierBase`, and it does not yet support
`embedding_rotation`, `entangler`, `readout` or `encoder_activation` (#226).

Options shared by both models:

- `encoding_type="angle"` (default) or `"iqp"` (Havlíček-style feature map with pairwise
  `x_i x_j` phases).
- `init_strategy="restricted"` (one σ for the whole circuit), `"block_local"` (the same σ in
  the first layer, narrowing by up to √2 towards the last) or `"normal"` (plain
  `N(0, init_std²)`, `init_std=0.1` by default); see `hqnn_forge.initializers`.
- `embedding_rotation="X"` (default) or `"Y"`: the Pauli axis of the angle embedding
  (angle encoding only). `"Z"` raises: a single `RZ` embedding on `|0⟩` is a global phase,
  so the quantum layer would ignore its inputs.
- `entangler="ring"` (default: CNOT ring then per-qubit `Rot`), `"strongly_entangling"`
  (`qml.StronglyEntanglingLayers`: `Rot` first, then a CNOT ring whose range grows with the
  layer index) or `"brickwork"` (nearest-neighbour CNOT pairs without wrap-around, so each
  ⟨Z_i⟩ readout depends on a few neighbouring qubits at shallow depth rather than on all of
  them).
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

### Classical control

`hqnn_forge.utils.classical_baseline(model)` builds the classical model a hybrid result should
be compared with: an untrained `ClassicalBaseline` MLP, to be trained from scratch on the same
data. Its trainable parameter count is matched to `model.count_parameters()`, which counts every
rotation angle as one parameter, the same convention as the MCC/kParam figures, so the two
models are compared at the same parameter budget. The serial model's control is one hidden
layer in place of encoder, circuit and head; the parallel model's is its classical branch plus a
head, widened to the matching width. The published SHNN's 122 parameters get a 121-parameter
control. A seeded hybrid (`init_seed`) gets a control seeded with the same seed.
Switching a trained model's circuit off with `disable_quantum_layer` measures something
else, how much that model depends on the circuit.

---

## Folder Structure

```
hqnn_forge/
├── encoding/        Quantum feature maps (angle, IQP, amplitude, data re-uploading)
├── circuits/        Reusable VQC ansatz primitives
├── initializers/    Small-angle (restricted-variance) weight initialisation
├── preprocessing/   PCA + normalisation, stratified folds, SMOTE (no sklearn runtime dep)
├── models/          Full hybrid architectures
├── training/        Train/validate loop with early stopping
├── evaluation/      Decision threshold search, confusion metrics, PR-AUC, score per
│                    parameter, paired Wilcoxon tests, plots (needs matplotlib)
├── diagnostics/     Circuit cost (depth, gates, inert parameters), gradient variance,
│                    Fisher information and effective dimension
├── data/            Dataset loader (Kaggle credit-card fraud)
├── utils/           Imbalance-robust losses, checkpoint save/load, quantum-layer ablation,
│                    eval-mode context manager
├── kernels.py       Quantum kernel matrices from the encoding layers (QSVM)
├── noise.py         Depolarizing noise, post hoc for robustness sweeps or during training
└── sklearn.py       scikit-learn estimator wrapper (cross_val_score, GridSearchCV, Pipeline);
                     needs the `sklearn` extra
```

---

## Reproducing the published SHNN

`HybridBinaryClassifier.published_shnn()` matches the published model structurally, and
`tests/test_published_shnn_parity.py` pins that. Whether the library also reproduces the
published *numbers* (MCC 0.5758 ± 0.0371, MCC/kParam 4.720) is checked by an opt-in run of the
benchmark's recipe: 5-fold CV with SMOTE on the training folds, 100 epochs. It needs the Kaggle
dataset and takes days of simulation:

```bash
HQNN_FORGE_REPRODUCE=1 HQNN_FORGE_DATA=data/raw \
    pytest tests/test_published_shnn_reproduction.py -m reproducibility -s
```

The module docstring lists the recipe and every deliberate deviation from the benchmark code.
**Status:** not yet measured. The numbers go here once a full run has finished.

---

## Development Setup

The project is managed with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --all-extras
uvx pre-commit install
```

`uv sync --all-extras` creates `.venv` with the project installed in editable mode and every
extra (`lightning`, `sklearn`, `examples`, `dev`) at the versions pinned in `uv.lock`. It is
the environment the `test-locked` CI job builds with `uv sync --locked --all-extras`; `--locked`
additionally fails instead of updating a `uv.lock` that no longer matches `pyproject.toml`.
Run tools inside it with `uv run`, e.g. `uv run pytest`, or activate `.venv`.

Without uv, `pip install -e ".[lightning,sklearn,examples,dev]"` installs the same extras at
the newest versions `pyproject.toml` allows, as the pip-based `test` CI job does. The pre-commit
hooks below still need uv.

The `dev` extra brings `ruff`, `mypy`, `vermin` and `pytest`. `uvx pre-commit install` registers the hooks
in `.pre-commit-config.yaml`, which run `ruff check --fix` and `ruff format` on every commit with
the settings from `pyproject.toml`. The hooks call ruff through `uv run`, so they need
[uv](https://docs.astral.sh/uv/getting-started/installation/) on the `PATH` and use the ruff
version pinned in `uv.lock`, the same one CI uses. To run them over the whole tree at any time:

```bash
uvx pre-commit run --all-files
```

The hooks cover the two ruff steps of the CI lint job, including the Python code blocks in
Markdown files. The lint job also type-checks the package, the tests and the examples with
mypy, and checks stdlib usage against Python 3.11 with vermin; the hooks do neither. Run
them before pushing:

```bash
uv run --frozen --all-extras mypy hqnn_forge tests examples
uv run --frozen --all-extras vermin --no-tips -t=3.11- --violations --eval-annotations \
    --exclude long hqnn_forge tests examples .github/scripts
```

[`CONTRIBUTING.md`](CONTRIBUTING.md#linting) lists every command the lint job runs.

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
- Abbas et al. (2021) — *The power of quantum neural networks*
- Berezniuk et al. (2020) — *A scale-dependent notion of effective dimension*
- Schuld et al. (2020) — *Circuit-centric quantum classifiers*
- Sim et al. (2019) — *Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms*
- Jones & Gacon (2020) — *Efficient calculation of gradients in classical simulations of variational quantum algorithms*
- Kandala et al. (2017) — *Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets*
- Havlíček et al. (2019) — *Supervised learning with quantum-enhanced feature spaces*
- Schuld & Killoran (2019) — *Quantum machine learning in feature Hilbert spaces*
- Hubregtsen et al. (2022) — *Training quantum embedding kernels on near-term quantum computers*
- Pérez-Salinas et al. (2020) — *Data re-uploading for a universal quantum classifier*
- Schuld, Sweke & Meyer (2021) — *Effect of data encoding on the expressive power of variational quantum-machine-learning models*
- Möttönen et al. (2005) — *Transformation of quantum states using uniformly controlled rotations*
- Schuld & Petruccione (2018) — *Supervised Learning with Quantum Computers*
- Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- King & Zeng (2001) — *Logistic Regression in Rare Events Data*
- Chawla et al. (2002) — *SMOTE: Synthetic Minority Over-sampling Technique*
- Wilcoxon (1945) — *Individual comparisons by ranking methods*
- Kerby (2014) — *The simple difference formula: an approach to teaching nonparametric correlation*
- Bergholm et al. (2022) — *PennyLane: Automatic differentiation of hybrid quantum-classical computations*
