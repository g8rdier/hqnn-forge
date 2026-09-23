# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffold and packaging setup
- PCANormalizer (pure-NumPy) for quantum angle encoding
- Small-angle restricted-variance initialiser (barren-plateau-aware; the σ formulas are
  this library's own heuristics, not a published prescription)
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

### Changed
- `load_checkpoint` fills constructor arguments a checkpoint predates from
  `checkpoint._LEGACY_DEFAULTS` — the behaviour from before each argument existed — with a
  `RuntimeWarning` naming them, instead of refusing the file. A checkpoint written before the
  options added above still rebuilds the model it holds; a config missing anything else is
  still refused
- Raised the `pennylane` and `pennylane-lightning` floors from `>=0.38` to `>=0.45`, the lowest
  version CI runs; 0.38 is incompatible with `autoray>=0.7`, and 0.42 was only tested on
  Python 3.10

### Fixed
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
