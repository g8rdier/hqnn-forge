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

    def test_map_location_places_the_returned_model(self, saved: tuple) -> None:
        # cls(**config) always builds on the CPU, so without an explicit move
        # map_location only relocated the tensors that load_state_dict then
        # copied back into CPU parameters -- it had no effect on the result.
        _, path = saved
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = load_checkpoint(path, map_location=device)
        assert all(p.device.type == device.type for p in loaded.parameters())
        assert all(b.device.type == device.type for b in loaded.buffers())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
    def test_map_location_cuda_does_not_return_a_cpu_model(self, saved: tuple) -> None:
        _, path = saved
        loaded = load_checkpoint(path, map_location="cuda", device_name="default.qubit")
        assert all(p.is_cuda for p in loaded.parameters())

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

    def test_architecture_override_needs_the_opt_in(self, saved: tuple) -> None:
        _, path = saved
        with pytest.raises(ValueError, match=r"\['n_layers'\] describe the saved architecture"):
            load_checkpoint(path, n_layers=3)

    def test_opted_in_architecture_override_still_checks_shapes(self, saved: tuple) -> None:
        _, path = saved
        with pytest.raises(RuntimeError, match="size mismatch"):
            load_checkpoint(path, n_layers=3, allow_architecture_override=True)

    def test_encoding_type_override_is_refused_not_silently_loaded(self, saved: tuple) -> None:
        """
        The override that shape checks cannot catch.

        QuantumEncodingLayer and IQPEncodingLayer both register
        quantum_layer.qlayer.weights at (n_layers, n_qubits, 3), so an angle
        checkpoint loads into an IQP model without a size mismatch and simply
        predicts something else.  Only the opt-in stands between a user and
        that model, so assert both halves: refused by default, and genuinely
        wrong once allowed.
        """
        model, path = saved
        with pytest.raises(ValueError, match=r"\['encoding_type'\].*allow_architecture_override=True"):
            load_checkpoint(path, encoding_type="iqp")

        forced = load_checkpoint(path, encoding_type="iqp", allow_architecture_override=True)
        assert forced.get_config()["encoding_type"] == "iqp"
        x = torch.randn(5, 6)
        model.eval()
        with torch.no_grad():
            assert not torch.allclose(forced(x), model(x), rtol=1e-3, atol=1e-3)

    def test_init_strategy_override_is_refused(self, saved: tuple) -> None:
        # Not a shape change either: it only picks how fresh weights are drawn,
        # which the loaded state dict then overwrites -- so overriding it just
        # bakes a wrong init_strategy into the rebuilt get_config().
        _, path = saved
        with pytest.raises(ValueError, match=r"\['init_strategy'\]"):
            load_checkpoint(path, init_strategy="block_local")

    def test_runtime_overrides_need_no_opt_in(self, saved: tuple) -> None:
        _, path = saved
        loaded = load_checkpoint(path, **CPU)
        assert loaded.get_config()["diff_method"] == "backprop"
        assert set(ckpt.RUNTIME_ONLY_ARGS) == {"device_name", "diff_method"}

    def test_missing_state_dict(self, saved: tuple, tmp_path: Path) -> None:
        _, path = saved
        payload = torch.load(path, weights_only=True)
        del payload["state_dict"]
        with pytest.raises(ValueError, match="has no 'state_dict'"):
            load_checkpoint(_save_payload(payload, tmp_path / "noweights.pt"))

    def test_not_a_checkpoint(self, tmp_path: Path) -> None:
        path = _save_payload({"weights": torch.zeros(2)}, tmp_path / "plain.pt")
        with pytest.raises(ValueError, match="is not an hqnn_forge checkpoint"):
            load_checkpoint(path)

    def test_foreign_file_is_a_value_error_not_a_torch_error(self, tmp_path: Path) -> None:
        # torch.load raises KeyError from its zip reader on a plain file, which
        # would escape load_checkpoint before the structural check runs.
        path = tmp_path / "notes.txt"
        path.write_text("this is not a checkpoint\n")
        with pytest.raises(ValueError, match="could not be read as a torch archive"):
            load_checkpoint(path)

    def test_missing_file_still_raises_oserror(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nope.pt")

    def test_model_without_recorded_config(self) -> None:
        from hqnn_forge.models import BinaryClassifierBase

        class Bare(BinaryClassifierBase):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        with pytest.raises(NotImplementedError, match="Bare does not record its constructor arguments"):
            Bare().get_config()
