"""
tests/test_checkpoint.py
========================
Round-trip and failure tests for hqnn_forge.utils.checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import hqnn_forge
from hqnn_forge.models import HybridBinaryClassifier, ParallelHybridClassifier
from hqnn_forge.utils import load_checkpoint, save_checkpoint
from hqnn_forge.utils import checkpoint as ckpt

CPU = dict(device_name="default.qubit", diff_method="backprop")

MODELS = [
    pytest.param(HybridBinaryClassifier, dict(encoding_type="angle"), id="serial-angle"),
    pytest.param(HybridBinaryClassifier, dict(encoding_type="iqp", init_strategy="block_local"), id="serial-iqp"),
    pytest.param(ParallelHybridClassifier, dict(classical_hidden_dim=5, dropout_p=0.2), id="parallel"),
]


def _trained(cls: type, extra: dict) -> torch.nn.Module:
    """A model whose weights differ from any fresh initialisation."""
    torch.manual_seed(0)
    model = cls(n_input_features=6, n_qubits=3, n_layers=2, **CPU, **extra)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    x, y = torch.randn(16, 6), torch.randint(0, 2, (16,)).float()
    for _ in range(3):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y).backward()
        opt.step()
    return model


def _save_payload(payload: dict, path: Path) -> Path:
    torch.save(payload, path)
    return path


@pytest.fixture
def saved(tmp_path: Path) -> tuple[torch.nn.Module, Path]:
    model = _trained(HybridBinaryClassifier, {})
    path = tmp_path / "model.pt"
    save_checkpoint(model, path)
    return model, path


class TestRoundTrip:
    @pytest.mark.parametrize("cls, extra", MODELS)
    def test_identical_outputs_after_reload(self, cls: type, extra: dict, tmp_path: Path) -> None:
        model = _trained(cls, extra)
        path = tmp_path / "model.pt"
        save_checkpoint(model, path)

        torch.manual_seed(123)  # a different RNG state must not matter
        loaded = load_checkpoint(path)

        assert type(loaded) is cls
        assert loaded.get_config() == model.get_config()
        x = torch.randn(5, 6)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)
        for (name, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)

    def test_loaded_model_is_in_eval_mode(self, saved: tuple) -> None:
        _, path = saved
        assert not load_checkpoint(path).training

    def test_config_round_trips_every_constructor_argument(self) -> None:
        for cls in (HybridBinaryClassifier, ParallelHybridClassifier):
            model = cls(n_input_features=4, n_qubits=4, n_layers=1, use_classical_encoder=False, **CPU)
            assert set(model.get_config()) == ckpt._init_parameter_names(cls)
            rebuilt = cls(**model.get_config())
            assert rebuilt.get_config() == model.get_config()

    def test_get_config_returns_a_copy(self) -> None:
        model = HybridBinaryClassifier(n_input_features=4, n_qubits=4, n_layers=1, **CPU)
        model.get_config()["n_qubits"] = 99
        assert model.get_config()["n_qubits"] == 4

    def test_override_device_on_load(self, saved: tuple) -> None:
        model, path = saved
        loaded = load_checkpoint(path, diff_method="parameter-shift")
        assert loaded.get_config()["diff_method"] == "parameter-shift"
        x = torch.randn(3, 6)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), rtol=1e-6, atol=1e-6)

    def test_file_loads_with_weights_only(self, saved: tuple) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        assert payload["format_version"] == ckpt.FORMAT_VERSION
        assert payload["hqnn_forge_version"] == hqnn_forge.__version__
        assert payload["class_name"] == "HybridBinaryClassifier"


class TestFailures:
    def test_unsupported_model(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="supports the classifiers in hqnn_forge.models.*got torch.nn.modules.linear.Linear"):
            save_checkpoint(torch.nn.Linear(2, 1), tmp_path / "x.pt")

    def test_subclass_is_not_silently_saved_as_parent(self, tmp_path: Path) -> None:
        class Custom(HybridBinaryClassifier):
            pass

        model = Custom(n_input_features=4, n_qubits=4, n_layers=1, **CPU)
        with pytest.raises(TypeError, match="Custom"):
            save_checkpoint(model, tmp_path / "x.pt")

    def test_version_mismatch(self, saved: tuple, tmp_path: Path) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        payload["hqnn_forge_version"] = "0.0.1"
        other = _save_payload(payload, tmp_path / "old.pt")
        with pytest.raises(ValueError, match=r"written by hqnn_forge 0\.0\.1.*allow_version_mismatch=True"):
            load_checkpoint(other)
        assert isinstance(load_checkpoint(other, allow_version_mismatch=True), HybridBinaryClassifier)

    def test_format_version_mismatch(self, saved: tuple, tmp_path: Path) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        payload["format_version"] = ckpt.FORMAT_VERSION + 1
        with pytest.raises(ValueError, match="format version 2 is not supported"):
            load_checkpoint(_save_payload(payload, tmp_path / "future.pt"))

    def test_missing_constructor_field(self, saved: tuple, tmp_path: Path) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        del payload["config"]["encoding_type"]
        with pytest.raises(ValueError, match=r"missing \['encoding_type'\]"):
            load_checkpoint(_save_payload(payload, tmp_path / "partial.pt"))

    def test_unexpected_constructor_field(self, saved: tuple, tmp_path: Path) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        payload["config"]["n_heads"] = 4
        with pytest.raises(ValueError, match=r"unexpected \['n_heads'\]"):
            load_checkpoint(_save_payload(payload, tmp_path / "extra.pt"))

    def test_unknown_override(self, saved: tuple) -> None:
        _, path = saved
        with pytest.raises(ValueError, match=r"unknown constructor arguments.*\['n_heads'\]"):
            load_checkpoint(path, n_heads=4)

    def test_unknown_class(self, saved: tuple, tmp_path: Path) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        payload["class_name"] = "os.system"
        with pytest.raises(ValueError, match="unknown class 'os.system'"):
            load_checkpoint(_save_payload(payload, tmp_path / "evil.pt"))

    def test_architecture_override_fails_to_load_weights(self, saved: tuple) -> None:
        _, path = saved
        with pytest.raises(RuntimeError, match="size mismatch"):
            load_checkpoint(path, n_layers=3)

    def test_not_a_checkpoint(self, tmp_path: Path) -> None:
        path = _save_payload({"weights": torch.zeros(2)}, tmp_path / "plain.pt")
        with pytest.raises(ValueError, match="is not an hqnn_forge checkpoint"):
            load_checkpoint(path)

    def test_model_without_recorded_config(self) -> None:
        from hqnn_forge.models import BinaryClassifierBase

        class Bare(BinaryClassifierBase):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        with pytest.raises(NotImplementedError, match="Bare does not record its constructor arguments"):
            Bare().get_config()
