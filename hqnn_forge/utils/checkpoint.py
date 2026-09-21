"""
hqnn_forge.utils.checkpoint
===========================
Save and reload a hybrid classifier together with its architecture.

``torch.save(model.state_dict())`` alone is not enough to get a model back:
the state dict holds weights, not ``n_qubits``, ``n_layers``,
``encoding_type`` or the other constructor arguments needed to build a module
those weights fit.  A checkpoint written here stores both, plus the class
name and the library version, and ``load_checkpoint`` rebuilds the model and
loads the weights in one call.

Format
------
A ``torch.save`` file containing a plain dict::

    {
        "format_version": 1,
        "hqnn_forge_version": "0.1.0",
        "class_name": "HybridBinaryClassifier",
        "config": {...constructor kwargs...},
        "state_dict": {...},
    }

Only primitives and tensors are stored, so the file loads with
``torch.load(weights_only=True)``: loading a checkpoint never executes code
from it, and only classes in ``hqnn_forge.models`` can be rebuilt.
"""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any

import torch

import hqnn_forge

if TYPE_CHECKING:
    # Type-only: importing this at runtime would be circular, since
    # hqnn_forge.models imports hqnn_forge.utils.
    from hqnn_forge.models.base import BinaryClassifierBase

#: Bumped whenever the dict layout above changes incompatibly.
FORMAT_VERSION: int = 1

#: Constructor arguments that select *how* the saved architecture runs rather
#: than *what* it is.  Overriding these on load cannot invalidate the stored
#: weights, so ``load_checkpoint`` accepts them without an opt-in; everything
#: else needs ``allow_architecture_override=True``.
RUNTIME_ONLY_ARGS: frozenset[str] = frozenset({"device_name", "diff_method"})

PathLike = str | os.PathLike[str]


def _registry() -> dict[str, type[BinaryClassifierBase]]:
    # Imported lazily: hqnn_forge.models imports hqnn_forge.utils, so a
    # module-level import here would be circular.
    from hqnn_forge import models

    return {name: getattr(models, name) for name in models.__all__ if name != "BinaryClassifierBase"}


def save_checkpoint(model: BinaryClassifierBase, path: PathLike) -> None:
    """
    Write ``model``'s class, constructor arguments and weights to ``path``.

    Parameters
    ----------
    model:
        A classifier from ``hqnn_forge.models``.
    path:
        Destination file; overwritten if it exists.

    Raises
    ------
    TypeError
        If ``model`` is not one of the library's classifiers.
    """
    class_name = type(model).__name__
    if _registry().get(class_name) is not type(model):
        raise TypeError(
            f"save_checkpoint supports the classifiers in hqnn_forge.models "
            f"({', '.join(sorted(_registry()))}); got {type(model).__module__}.{class_name}."
        )
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "hqnn_forge_version": hqnn_forge.__version__,
        "class_name": class_name,
        "config": model.get_config(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: PathLike,
    *,
    map_location: str | torch.device = "cpu",
    allow_version_mismatch: bool = False,
    allow_architecture_override: bool = False,
    **overrides: Any,
) -> BinaryClassifierBase:
    """
    Rebuild a classifier saved with :func:`save_checkpoint`.

    Parameters
    ----------
    path:
        Checkpoint file.
    map_location:
        Where the rebuilt model ends up: the stored tensors are read onto this
        device and the model is moved there before its weights are loaded.
        Default: ``"cpu"``.  Note that this places the *classical* layers and
        the variational parameters only -- which simulator executes the
        circuit is chosen by the ``device_name`` argument, so moving a model to
        a GPU generally means overriding ``device_name`` as well.
    allow_version_mismatch:
        Load a checkpoint written by a different ``hqnn_forge`` version.  Off
        by default because weight layouts are not guaranteed stable across
        versions before 1.0.
    allow_architecture_override:
        Permit ``**overrides`` outside :data:`RUNTIME_ONLY_ARGS`.  Off by
        default: an architecture override makes the rebuilt model something
        other than the one that was saved, and the mismatch is not always
        loud.  ``encoding_type`` is the dangerous case -- ``QuantumEncodingLayer``
        and ``IQPEncodingLayer`` both register their weights at
        ``(n_layers, n_qubits, 3)``, so an ``"angle"`` checkpoint loaded as
        ``"iqp"`` fits, raises nothing, and predicts differently.
    **overrides:
        Constructor arguments that replace the stored ones.  Without
        ``allow_architecture_override``, only :data:`RUNTIME_ONLY_ARGS`
        (``device_name``, ``diff_method``) may be given -- typically to run a
        saved model on a different simulator.

    Returns
    -------
    BinaryClassifierBase
        The rebuilt model with the saved weights, in eval mode.

    Raises
    ------
    ValueError
        If the file is not a checkpoint or is incomplete, on an unknown format
        version, a library version mismatch (unless allowed), an unknown class,
        an architecture override without ``allow_architecture_override``, or a
        config with missing or unexpected constructor fields.
    RuntimeError
        If the stored weights do not fit the rebuilt architecture.
    """
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except OSError:
        # Missing file, unreadable path: the caller's problem, not a malformed
        # checkpoint.
        raise
    except Exception as exc:
        # torch.load fails its own way on anything that is not a torch archive
        # (a text file raises KeyError from the zip reader), so the structural
        # check below is never reached for a genuinely foreign file.
        raise ValueError(
            f"{os.fspath(path)!r} could not be read as a torch archive, so it is "
            f"not an hqnn_forge checkpoint."
        ) from exc

    if not isinstance(payload, dict) or "format_version" not in payload:
        raise ValueError(f"{os.fspath(path)!r} is not an hqnn_forge checkpoint.")

    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {payload['format_version']} is not supported; "
            f"this hqnn_forge reads format version {FORMAT_VERSION}."
        )

    if "state_dict" not in payload:
        raise ValueError(
            f"{os.fspath(path)!r} has no 'state_dict'; the checkpoint is incomplete."
        )

    saved_version = payload.get("hqnn_forge_version")
    if saved_version != hqnn_forge.__version__ and not allow_version_mismatch:
        raise ValueError(
            f"checkpoint was written by hqnn_forge {saved_version}, this is "
            f"{hqnn_forge.__version__}.  Pass allow_version_mismatch=True to load it anyway."
        )

    class_name = payload.get("class_name")
    registry = _registry()
    if class_name not in registry:
        raise ValueError(
            f"checkpoint holds unknown class {class_name!r}; expected one of {sorted(registry)}."
        )
    cls = registry[class_name]

    expected = _init_parameter_names(cls)
    unknown_overrides = sorted(set(overrides) - expected)
    if unknown_overrides:
        raise ValueError(
            f"unknown constructor arguments for {class_name}: {unknown_overrides}."
        )

    architecture_overrides = sorted(set(overrides) - RUNTIME_ONLY_ARGS)
    if architecture_overrides and not allow_architecture_override:
        raise ValueError(
            f"{architecture_overrides} describe the saved architecture, not how it "
            f"runs, so overriding them rebuilds a different model than the weights "
            f"were trained in -- and the mismatch is not always caught: an 'angle' "
            f"checkpoint loads without error as encoding_type='iqp' and predicts "
            f"differently.  Only {sorted(RUNTIME_ONLY_ARGS)} may be overridden; pass "
            f"allow_architecture_override=True if that is really what you want."
        )

    config = dict(payload.get("config") or {})
    config.update(overrides)

    missing = sorted(expected - set(config))
    unexpected = sorted(set(config) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(
            f"checkpoint config does not match {class_name}'s constructor: {'; '.join(details)}."
        )

    model = cls(**config)
    # Before load_state_dict, so the stored tensors -- already read onto
    # map_location -- are copied into parameters that live there too.  Building
    # the model alone always puts it on the CPU.
    model.to(map_location)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _init_parameter_names(cls: type) -> set[str]:
    """
    Keyword names of ``cls.__init__``.

    A checkpoint must carry every one of them, not only those without a
    default: get_config records them all, and a default that changed between
    versions would otherwise silently change the rebuilt model.
    """
    return {
        name
        for name, p in inspect.signature(cls).parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
