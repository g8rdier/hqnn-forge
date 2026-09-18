"""Minimal pytest configuration for hqnn-forge."""

from __future__ import annotations

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "reproducibility: checks against the published benchmark configuration "
        "(deselect with -m 'not reproducibility')",
    )


def _grad(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor.grad`` after a backward pass, narrowed from ``Tensor | None``."""
    assert tensor.grad is not None
    return tensor.grad
