# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffold and packaging setup
- PCANormalizer (pure-NumPy) for quantum angle encoding
- Barren-plateau-safe restricted-variance initialiser
- Angle-embedding quantum encoding layer with adjoint diff
- Strongly-entangling and hardware-efficient VQC primitives
- Focal loss and weighted BCE for imbalanced classification
- HybridBinaryClassifier end-to-end model
- Smoke tests for quantum encoding layer
- Quick-start example for training

### Changed
- Raised the `pennylane` and `pennylane-lightning` floors from `>=0.38` to `>=0.42`, the lowest
  version CI runs; 0.38 is incompatible with `autoray>=0.7`

### Removed
- `requirements.txt`; `pyproject.toml` is now the only place dependencies are declared
- Python 3.10 support; `requires-python` is now `>=3.11`. 3.10 reaches end of life in
  October 2026 and current PennyLane releases no longer install on it, so CI tests 3.11
  (the floor) and 3.14 (the newest) instead
