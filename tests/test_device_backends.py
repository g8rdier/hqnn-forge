"""
tests/test_device_backends.py
=============================
Device resolution in hqnn_forge.encoding.angle_embedding._resolve_device:
the accelerated backends when they are installed, and the fallback chain
``requested → lightning.qubit → default.qubit`` deterministically, by making
``qml.device`` fail for chosen names, whatever this machine has.
"""

from __future__ import annotations

import warnings

import pennylane as qml
import pytest
import torch
from pennylane.exceptions import AllocationError, DeviceError

from hqnn_forge.encoding import (
    AUTO_BACKPROP_MAX_QUBITS,
    AmplitudeEncodingLayer,
    DataReuploadingLayer,
    QuantumEncodingLayer,
    resolve_backend,
)
from hqnn_forge.encoding import angle_embedding as ae
from hqnn_forge.encoding._common import KNOWN_DEVICES
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer
from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.utils import load_checkpoint, save_checkpoint

GPU_BACKENDS = ("lightning.gpu", "lightning.kokkos")


def _available(name: str) -> bool:
    try:
        qml.device(name, wires=1)
    except ae._DEVICE_FAILURES:
        return False
    return True


def _failing_device(failing: dict[str, type[BaseException]]):
    """A stand-in for ``qml.device`` that raises for the names in ``failing``."""
    real = qml.device

    def device(name: str, *args, **kwargs):
        if name in failing:
            raise failing[name](f"{name} unavailable in this test")
        return real(name, *args, **kwargs)

    return device


# ---------------------------------------------------------------------------
# Accelerated backends: exercised when present, skipped otherwise
# ---------------------------------------------------------------------------


class TestAcceleratedBackends:
    @pytest.mark.parametrize("name", GPU_BACKENDS)
    def test_layer_runs_on_backend_when_available(self, name: str) -> None:
        if not _available(name):
            pytest.skip(f"{name} is not installed or has no usable device here")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # no fallback expected
            layer = QuantumEncodingLayer(n_qubits=3, n_layers=1, device_name=name)  # type: ignore[arg-type]
        assert layer.qlayer.qnode.device.name == name
        reference = QuantumEncodingLayer(
            n_qubits=3, n_layers=1, device_name="default.qubit", diff_method="backprop"
        )
        reference.load_state_dict(layer.state_dict())

        x = torch.rand(4, 3)
        out, expected = layer(x), reference(x)
        assert out.shape == (4, 3)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5, check_dtype=False)

        # A non-uniform upstream gradient, so a permuted batch or output axis
        # in the backend's adjoint path shows up in weights.grad.
        upstream = torch.arange(1.0, 13.0).reshape(4, 3)
        (out * upstream.to(out.dtype)).sum().backward()
        (expected * upstream.to(expected.dtype)).sum().backward()
        torch.testing.assert_close(
            layer.qlayer.weights.grad,
            reference.qlayer.weights.grad,
            rtol=1e-5,
            atol=1e-5,
            check_dtype=False,
        )

    @pytest.mark.parametrize("name", GPU_BACKENDS)
    def test_backend_names_are_known_to_the_fallback_chain(self, name: str) -> None:
        assert name in KNOWN_DEVICES


# ---------------------------------------------------------------------------
# Fallback chain, deterministic
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_missing_gpu_backend_falls_back_to_lightning_then_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ae.qml,
            "device",
            _failing_device({"lightning.gpu": DeviceError, "lightning.qubit": ImportError}),
        )
        with pytest.warns(RuntimeWarning) as record:
            dev = ae._resolve_device("lightning.gpu", 2)
        assert dev.name == "default.qubit"
        messages = [str(w.message) for w in record]
        assert len(messages) == 2
        assert "lightning.gpu" in messages[0] and "lightning.qubit" in messages[0]
        assert "lightning.qubit" in messages[1] and "default.qubit" in messages[1]
        assert "Install pennylane-lightning" in messages[1]

    def test_missing_gpu_backend_stops_at_lightning_when_it_works(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not _available("lightning.qubit"):
            pytest.skip("pennylane-lightning not installed")
        monkeypatch.setattr(ae.qml, "device", _failing_device({"lightning.kokkos": RuntimeError}))
        with pytest.warns(
            RuntimeWarning, match="lightning.kokkos.*Falling back to 'lightning.qubit'"
        ):
            dev = ae._resolve_device("lightning.kokkos", 2)
        assert dev.name == "lightning.qubit"

    def test_lightning_qubit_falls_straight_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ae.qml, "device", _failing_device({"lightning.qubit": OSError}))
        with pytest.warns(RuntimeWarning) as record:
            dev = ae._resolve_device("lightning.qubit", 2)
        assert dev.name == "default.qubit"
        assert len(record) == 1

    def test_default_qubit_has_no_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ae.qml, "device", _failing_device({"default.qubit": DeviceError}))
        with pytest.raises(DeviceError):
            ae._resolve_device("default.qubit", 2)

    @pytest.mark.parametrize("name", ["default.qbit", "no.such.device"])
    def test_an_unknown_name_raises_instead_of_falling_back(self, name: str) -> None:
        """A typo must not fail like a missing plugin and run on another simulator."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # raised, not fallen back
            with pytest.raises(DeviceError):
                ae._resolve_device(name, 2)

    def test_any_other_pennylane_device_is_constructed_as_given(self) -> None:
        # #314: plugin and hardware devices by name.  default.mixed stands in
        # for one here, as a registered device outside the fallback chain.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert ae._resolve_device("default.mixed", 2).name == "default.mixed"

    def test_unregistered_plugin_falls_back_without_attribute_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The branch that was unreachable on PennyLane 0.45 (qml.DeviceError no
        longer exists, see #154): the ``DeviceError`` PennyLane raises for a
        name no installed plugin registers must warn and fall back.
        """
        real = qml.device

        def device(name: str, *args, **kwargs):
            # What PennyLane itself raises for a name no plugin registers.
            return real("no.such.device" if name == "lightning.gpu" else name, *args, **kwargs)

        monkeypatch.setattr(ae.qml, "device", device)
        with pytest.warns(RuntimeWarning, match="DeviceError.*Falling back"):
            dev = ae._resolve_device("lightning.gpu", 2)
        assert dev.name in ("lightning.qubit", "default.qubit")

    @pytest.mark.parametrize(
        "error",
        [
            AllocationError("state vector too large"),
            RuntimeError("[cudaMalloc] out of memory"),
        ],
    )
    def test_out_of_memory_is_raised_not_retried_on_the_host(
        self, monkeypatch: pytest.MonkeyPatch, error: BaseException
    ) -> None:
        """Every backend needs the same 2**n state; falling back cannot fit it."""
        tried: list[str] = []

        def device(name: str, *args, **kwargs):
            tried.append(name)
            raise error

        monkeypatch.setattr(ae.qml, "device", device)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(type(error), match=str(error).split()[-1]):
                ae._resolve_device("lightning.gpu", 30)
        assert tried == ["lightning.gpu"]

    def test_no_warning_when_the_requested_device_works(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert ae._resolve_device("default.qubit", 2).name == "default.qubit"

    def test_iqp_layer_and_classifier_share_the_chain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ae.qml,
            "device",
            _failing_device({"lightning.gpu": DeviceError, "lightning.qubit": DeviceError}),
        )
        with pytest.warns(RuntimeWarning):
            iqp = IQPEncodingLayer(
                n_qubits=2, n_layers=1, device_name="lightning.gpu", diff_method="backprop"
            )
        assert iqp.qlayer.qnode.device.name == "default.qubit"
        with pytest.warns(RuntimeWarning):
            model = HybridBinaryClassifier(
                n_input_features=2,
                n_qubits=2,
                n_layers=1,
                device_name="lightning.gpu",
                diff_method="backprop",
            )
        assert model.quantum_layer.qlayer.qnode.device.name == "default.qubit"
        assert model(torch.rand(3, 2)).shape == (3, 1)


# ---------------------------------------------------------------------------
# device_name="auto" / diff_method="auto" (#349)
# ---------------------------------------------------------------------------

requires_lightning = pytest.mark.skipif(
    not _available("lightning.qubit"), reason="pennylane-lightning is not installed"
)


def _backend(layer: torch.nn.Module) -> tuple[str, str]:
    qnode = layer.qlayer.qnode  # type: ignore[union-attr]
    return qnode.device.name, str(qnode.diff_method)


class TestAuto:
    @pytest.mark.parametrize(
        ("device_name", "diff_method", "n_qubits", "expected"),
        [
            ("auto", "auto", 2, ("default.qubit", "backprop")),
            ("auto", "auto", AUTO_BACKPROP_MAX_QUBITS, ("default.qubit", "backprop")),
            ("auto", "auto", AUTO_BACKPROP_MAX_QUBITS + 1, ("lightning.qubit", "adjoint")),
            # An explicit method steers the device to the one it is fast on.
            ("auto", "adjoint", 2, ("lightning.qubit", "adjoint")),
            ("auto", "backprop", 20, ("default.qubit", "backprop")),
            ("auto", "parameter-shift", 2, ("default.qubit", "parameter-shift")),
            ("auto", "parameter-shift", 20, ("lightning.qubit", "parameter-shift")),
            # An explicit device picks the method.
            ("default.qubit", "auto", 20, ("default.qubit", "backprop")),
            ("default.mixed", "auto", 2, ("default.mixed", "backprop")),
            ("lightning.qubit", "auto", 2, ("lightning.qubit", "adjoint")),
            ("lightning.gpu", "auto", 2, ("lightning.gpu", "adjoint")),
            ("qiskit.aer", "auto", 2, ("qiskit.aer", "parameter-shift")),
            # Explicit on both sides is left alone, even when it is slow.
            ("default.qubit", "adjoint", 2, ("default.qubit", "adjoint")),
        ],
    )
    def test_resolution_rules(
        self, device_name: str, diff_method: str, n_qubits: int, expected: tuple[str, str]
    ) -> None:
        assert resolve_backend(device_name, diff_method, n_qubits) == expected  # type: ignore[arg-type]

    @pytest.mark.parametrize("n_qubits", [2, 20])
    def test_shots_pick_parameter_shift(self, n_qubits: int) -> None:
        _, method = resolve_backend("auto", "auto", n_qubits, shots=100)
        assert method == "parameter-shift"

    def test_require_backprop_overrides_the_size_rule(self) -> None:
        assert resolve_backend("auto", "auto", 20, require_backprop=True) == (
            "default.qubit",
            "backprop",
        )

    @pytest.mark.parametrize(
        "layer_cls", [QuantumEncodingLayer, IQPEncodingLayer, DataReuploadingLayer]
    )
    def test_layers_default_to_auto(self, layer_cls: type[torch.nn.Module]) -> None:
        layer = layer_cls(n_qubits=AUTO_BACKPROP_MAX_QUBITS, n_layers=1)
        assert _backend(layer) == ("default.qubit", "backprop")

    @requires_lightning
    def test_switch_point_above_the_threshold(self) -> None:
        layer = QuantumEncodingLayer(n_qubits=AUTO_BACKPROP_MAX_QUBITS + 1, n_layers=1)
        assert _backend(layer) == ("lightning.qubit", "adjoint")

    def test_missing_lightning_falls_back_and_keeps_adjoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ae.qml, "device", _failing_device({"lightning.qubit": DeviceError}))
        with pytest.warns(RuntimeWarning, match="lightning.qubit"):
            layer = QuantumEncodingLayer(n_qubits=AUTO_BACKPROP_MAX_QUBITS + 1, n_layers=1)
        assert _backend(layer) == ("default.qubit", "adjoint")

    def test_shots_layer_trains_with_parameter_shift(self) -> None:
        layer = QuantumEncodingLayer(n_qubits=2, n_layers=1, shots=50)
        assert _backend(layer) == ("default.qubit", "parameter-shift")
        layer(torch.rand(3, 2)).sum().backward()
        assert layer.qlayer.weights.grad is not None

    def test_classifier_records_auto_and_round_trips(self, tmp_path) -> None:
        model = HybridBinaryClassifier(n_input_features=3, n_qubits=2, n_layers=1)
        config = model.get_config()
        assert (config["device_name"], config["diff_method"]) == ("auto", "auto")
        assert _backend(model.quantum_layer) == ("default.qubit", "backprop")
        save_checkpoint(model, tmp_path / "m.pt")
        loaded = load_checkpoint(tmp_path / "m.pt")
        assert loaded.get_config() == config
        x = torch.rand(4, 3)
        torch.testing.assert_close(loaded.predict_proba(x), model.predict_proba(x))

    def test_amplitude_with_encoder_gets_backprop_above_the_threshold(self) -> None:
        n = AUTO_BACKPROP_MAX_QUBITS + 1
        model = HybridBinaryClassifier(
            n_input_features=3, n_qubits=n, n_layers=1, encoding_type="amplitude"
        )
        assert _backend(model.quantum_layer) == ("default.qubit", "backprop")

    @requires_lightning
    def test_amplitude_without_encoder_follows_the_size_rule(self) -> None:
        n = AUTO_BACKPROP_MAX_QUBITS + 1
        model = HybridBinaryClassifier(
            n_input_features=3,
            n_qubits=n,
            n_layers=1,
            encoding_type="amplitude",
            use_classical_encoder=False,
        )
        assert _backend(model.quantum_layer) == ("lightning.qubit", "adjoint")

    def test_amplitude_with_encoder_still_refuses_explicit_adjoint(self) -> None:
        with pytest.raises(ValueError, match="only correct"):
            HybridBinaryClassifier(
                n_input_features=3,
                n_qubits=2,
                n_layers=1,
                encoding_type="amplitude",
                diff_method="adjoint",
            )

    def test_amplitude_layer_input_gradient_works_by_default(self) -> None:
        layer = AmplitudeEncodingLayer(n_qubits=2, n_layers=1)
        x = torch.rand(3, 4, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_explicit_choices_are_untouched(self) -> None:
        model = HybridBinaryClassifier(
            n_input_features=2,
            n_qubits=2,
            n_layers=1,
            device_name="default.qubit",
            diff_method="parameter-shift",
        )
        assert _backend(model.quantum_layer) == ("default.qubit", "parameter-shift")
