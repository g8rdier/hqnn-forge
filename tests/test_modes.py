"""
tests/test_modes.py
====================
Unit tests for hqnn_forge.utils.modes.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from hqnn_forge.utils.modes import eval_mode


def _net() -> nn.Sequential:
    """Nested two levels deep, so a restore that only touches the root is caught."""
    return nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 1)))


def _modes(module: nn.Module) -> list[bool]:
    return [submodule.training for submodule in module.modules()]


class _RecordsTrainCalls(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[bool] = []

    def train(self, mode: bool = True) -> _RecordsTrainCalls:
        self.calls.append(mode)
        return super().train(mode)


class TestEvalMode:
    def test_eval_inside_block(self) -> None:
        net = _net()
        with eval_mode(net):
            assert not any(_modes(net))

    def test_restores_train_mode(self) -> None:
        net = _net()
        with eval_mode(net):
            pass
        assert all(_modes(net))

    def test_leaves_eval_mode(self) -> None:
        net = _net().eval()
        with eval_mode(net):
            pass
        assert not any(_modes(net))

    def test_preserves_mixed_submodule_modes(self) -> None:
        """A blanket ``net.train(was_training)`` restore would re-enable dropout here."""
        net = _net()
        net[1][0].eval()
        before = _modes(net)
        with eval_mode(net):
            pass
        assert _modes(net) == before

    def test_restores_modes_when_block_raises(self) -> None:
        net = _net()
        net[1][0].eval()
        before = _modes(net)
        with pytest.raises(RuntimeError, match="boom"), eval_mode(net):
            raise RuntimeError("boom")
        assert _modes(net) == before

    def test_restore_calls_train_overrides(self) -> None:
        """Setting ``training`` directly would leave the override's last call at False."""
        child = _RecordsTrainCalls()
        with eval_mode(nn.Sequential(child)):
            pass
        assert child.calls[-1] is True

    def test_shared_submodule_keeps_its_mode(self) -> None:
        """``inner.train(True)`` runs after ``shared`` is restored and would overwrite it."""
        shared = nn.Dropout(0.5)
        inner = nn.Sequential(shared)
        net = nn.Sequential(shared, inner)
        shared.eval()
        with eval_mode(net):
            pass
        assert net.training
        assert inner.training
        assert not shared.training
