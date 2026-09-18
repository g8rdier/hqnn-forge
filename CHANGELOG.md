# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffold and packaging setup
- PCANormalizer (pure-NumPy) for quantum angle encoding
- Small-angle restricted-variance initialiser (the σ formulas are this library's own
  heuristics, not a published prescription)
- Angle-embedding quantum encoding layer with adjoint diff
- Strongly-entangling and hardware-efficient VQC primitives
- Focal loss and weighted BCE for imbalanced classification
- HybridBinaryClassifier end-to-end model
- Smoke tests for quantum encoding layer
- Quick-start example for training
- `entangler` (`"ring"` / `"strongly_entangling"`) and `readout` (`"all"` / `"first"`) on the
  encoding layers; `embedding_rotation`, `entangler`, `readout`, `encoder_activation` and
  `init_strategy="normal"` with `init_std` on the classifiers, and `published_shnn()` on both
  `HybridBinaryClassifier` (the thesis's 122-parameter SHNN configuration) and
  `ParallelHybridClassifier` (that quantum branch beside the classical MLP branch)
- `AmplitudeEncodingLayer`: amplitude embedding of up to `2**n_qubits` features per sample,
  with zero-padding and L2 normalisation in `forward`, ahead of the same entangling ansatz;
  gradients with respect to the inputs are only supported under `backprop`
- `DataReuploadingLayer`: angle embedding repeated before every variational layer
  (Pérez-Salinas et al. 2020), with the same `entangler` and `readout` options and optional
  trainable per-upload input scaling; `rotation="Z"` requires `n_layers >= 2` and leaves its
  first upload, a phase on `|0⟩`, unscaled
- `apply_variational_layers` takes a `layer_offset`, so a circuit applying the blocks one at
  a time keeps the `"strongly_entangling"` ranges of the whole ansatz
- `hqnn_forge.kernels.quantum_kernel_matrix`: pairwise state-fidelity kernel
  `|⟨Φ(x_i)|Φ(x_j)⟩|²` from any encoding layer, for `SVC(kernel="precomputed")`
- `hqnn_forge.diagnostics.fisher_information_matrix` and `effective_dimension`: Fisher
  information spectrum of the quantum weights and the effective dimension of Abbas et al. (2021);
  the effective dimension requires `κ = γn / (2π log n) ≥ e` (`n_data ≥ 74` at `γ = 1`), below
  which the formula changes sign or explodes
- `lightning.gpu` and `lightning.kokkos` accepted as `device_name`, with a fallback chain
  through `lightning.qubit` to `default.qubit` and a `RuntimeWarning` per step
- `MulticlassHybridClassifier`: shared quantum layer with `n_classes` linear heads, softmax or
  one-vs-rest probabilities, argmax `predict`; supported by `save_checkpoint` /
  `load_checkpoint`
- `noise_level` / `noise_position` on `QuantumEncodingLayer`, `IQPEncodingLayer`,
  `HybridBinaryClassifier` and `ParallelHybridClassifier`: depolarizing noise applied in train
  mode so gradients flow through the noisy circuit, practical up to about 6 qubits;
  `examples/noise_aware_training.py` compares noiseless and noise-aware training under the
  post-hoc noise sweep. Both are weight-safe `load_checkpoint` overrides, and a checkpoint
  from before them loads as noiseless. `apply_depolarizing_noise` at `p = 0` suppresses the
  training channel, and `gradient_variance` measures the layer in eval mode, so neither sees
  the train-mode noise
- `CircuitSummary.n_inert_params` / `count_inert_parameters`: trainable gate parameters that
  can never reach a measurement, found structurally; the last layer's `Rot` ω angles under a
  `⟨Z⟩` readout are `n_qubits` such parameters in every default model

### Changed
- `load_checkpoint` fills constructor arguments a checkpoint predates from
  `checkpoint._LEGACY_DEFAULTS` — the behaviour from before each argument existed — with a
  `RuntimeWarning` naming them, instead of refusing the file. A checkpoint written before the
  options added above still rebuilds the model it holds; a config missing anything else is
  still refused
- The restricted-variance initialiser is no longer described as barren-plateau-safe or
  -aware. Measured with `gradient_variance` (2 layers, ⟨Z_0⟩ cost), it gives no gain in initial
  gradient variance for inputs spread over (-π, π), which both classifiers feed the circuit, and
  a gain growing from 1.1x to 1.75x between 4 and 8 qubits near zero input; over (-π, π) the
  ~3x decay per two qubits is the same under either init. README and docstrings now state that
- Raised the `pennylane` and `pennylane-lightning` floors from `>=0.38` to `>=0.45`, the lowest
  version CI runs; 0.38 is incompatible with `autoray>=0.7`, and 0.42 was only tested on
  Python 3.10
- Raised the `torch` floor from `>=2.2` to `>=2.3` and the `numpy` floor from `>=1.26` to
  `>=2.0`. `pennylane>=0.45` requires NumPy 2, and the torch 2.2 wheels were compiled against
  NumPy 1.x and fail to initialise NumPy 2; CI now installs every declared floor of the runtime
  dependencies and the `lightning` and `sklearn` extras (`test-lowest` job) and fails if one
  is unreachable, so a floor that stops working fails a PR instead of a user install
- `PCANormalizer`'s fitted attributes (`mean_`, `components_`, `explained_variance_`,
  `std_`) are absent before `fit`, as in scikit-learn, instead of set to `None`; reading one
  on an unfitted instance raises `AttributeError`. Check `is_fitted_` instead of comparing an
  attribute to `None`

### Fixed
- Device fallback raised `AttributeError` on PennyLane 0.45, where `qml.DeviceError` no longer
  exists; the chain now catches `pennylane.exceptions.DeviceError` and is exercised by a test
- `disable_quantum_layer` fills the quantum layer's readout width (`n_outputs`) rather than
  `n_qubits`, so an ablated model built with `readout="first"` matches its `Linear(1 → 1)` head
  instead of raising a shape error where the real model works
- `QuantumEncodingLayer` and `build_encoding_qnode` reject an unknown `rotation` axis at
  construction, where `entangler` and `readout` are already rejected; it used to construct
  cleanly and fail inside PennyLane on the first forward pass

### Removed
- `black` from the `dev` extra; `ruff format` is the only formatter, sharing the
  `line-length = 99` ruff already enforces, so the two tools can no longer disagree
- `requirements.txt`; `pyproject.toml` is now the only place dependencies are declared
- Python 3.10 support; `requires-python` is now `>=3.11`. 3.10 reaches end of life in
  October 2026 and current PennyLane releases no longer install on it, so CI tests 3.11
  (the floor) and 3.14 (the newest) instead
