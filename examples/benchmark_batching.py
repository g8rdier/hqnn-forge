"""
examples/benchmark_batching.py
==============================
How a batch runs through an encoding layer, and what that costs (#312).

For every ``diff_method`` except ``backprop``, the encoders split a batch into
one tape per sample before the gradient transform sees it
(``hqnn_forge.encoding._common.expand_batch_dimension``).  This script
compares, for each encoder, forward and forward+backward time of

* ``split``           -- lightning.qubit / adjoint, one tape per sample (the library default);
* ``native``          -- lightning.qubit / adjoint on the broadcast tape, no split;
* ``split+batch_obs`` -- the split, on a lightning device built with ``batch_obs=True``;
* ``backprop``        -- default.qubit / backprop, which vectorises the batch;

and checks that every variant's outputs and gradients agree with ``split`` (to a
float32 relative tolerance).  ``--crossover`` then times one training step at
batch 64 over a range of qubit counts for the two main paths and reports the
peak memory of each, in a fresh interpreter per point.

Run::

    python examples/benchmark_batching.py            # the comparison table
    python examples/benchmark_batching.py --crossover

Results on one laptop CPU (PennyLane 0.45.1, pennylane-lightning 0.45.0,
torch 2.14), angle layer, 2 layers, forward+backward:

    batch 128, 8 qubits:  split 643 ms, native 760 ms, backprop 48 ms
    batch 64, crossover:  qubits   lightning/adjoint   default.qubit/backprop
                               8     0.59 s   +11 MB      0.04 s    +11 MB
                              12     0.57 s   +18 MB      0.30 s   +299 MB
                              14     1.42 s   +22 MB      1.62 s  +1182 MB
                              16    10.1  s   +35 MB      9.8  s  +3161 MB

Timings vary by some tens of percent between runs.  Native broadcasting is
correct on lightning's adjoint path in this PennyLane version but is not
faster than the split, so the split stays.  backprop is the fast path for
batches of small circuits (for a single sample lightning is faster), and loses
on memory from about 14 qubits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any

import pennylane as qml
import torch

from hqnn_forge.encoding import AmplitudeEncodingLayer, DataReuploadingLayer, QuantumEncodingLayer
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer

ENCODERS: dict[str, tuple[Callable[..., Any], dict[str, Any], bool]] = {
    # name: (class, extra options, needs one feature per qubit)
    "angle": (QuantumEncodingLayer, {}, True),
    "iqp": (IQPEncodingLayer, {}, True),
    "reuploading": (DataReuploadingLayer, {"trainable_input_scaling": True}, True),
    "amplitude": (AmplitudeEncodingLayer, {}, False),
}


def _variants(cls: Callable[..., Any], n: int, extra: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(0)
    split = cls(
        n_qubits=n, n_layers=2, device_name="lightning.qubit", diff_method="adjoint", **extra
    )

    def copy(device: str, diff_method: str) -> Any:
        layer = cls(n_qubits=n, n_layers=2, device_name=device, diff_method=diff_method, **extra)
        layer.load_state_dict(split.state_dict())
        return layer

    native = copy("lightning.qubit", "adjoint")
    q = native.qlayer.qnode
    native.qlayer.qnode = qml.QNode(q.func, q.device, interface="torch", diff_method="adjoint")
    batch_obs = copy("lightning.qubit", "adjoint")
    q = batch_obs.qlayer.qnode
    batch_obs.qlayer.qnode = qml.transforms.broadcast_expand(
        qml.QNode(
            q.func,
            qml.device("lightning.qubit", wires=n, batch_obs=True),
            interface="torch",
            diff_method="adjoint",
        )
    )
    return {
        "split": split,
        "native": native,
        "split+batch_obs": batch_obs,
        "backprop": copy("default.qubit", "backprop"),
    }


def _run(
    layer: Any, x: torch.Tensor, input_grads: bool
) -> tuple[list[torch.Tensor], float, float]:
    x = x.clone().requires_grad_(input_grads)
    layer.zero_grad()
    start = time.perf_counter()
    out = layer(x)
    forward = time.perf_counter() - start
    start = time.perf_counter()
    out.sum().backward()
    backward = time.perf_counter() - start
    assert layer.qlayer.weights.grad is not None
    tensors = [out.detach(), layer.qlayer.weights.grad.clone()]
    if input_grads:
        assert x.grad is not None
        tensors.append(x.grad.clone())
    return tensors, forward, backward


def _agree(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    # float32: relative to the largest entry, as summed gradients grow with the batch.
    return all(
        (u - v).abs().max() <= 1e-5 * max(1.0, float(u.abs().max()))
        for u, v in zip(a, b, strict=True)
    )


def compare(qubits: tuple[int, ...], batches: tuple[int, ...]) -> None:
    names = ("split", "native", "split+batch_obs", "backprop")
    print(f"{'encoder':12s} {'n':>2s} {'batch':>5s}  " + "  ".join(f"{v:>22s}" for v in names))
    for name, (cls, extra, per_qubit) in ENCODERS.items():
        # The amplitude layer refuses input gradients outside backprop.
        input_grads = name != "amplitude"
        for n in qubits:
            variants = _variants(cls, n, extra)
            width = n if per_qubit else 2**n
            for batch in batches:
                x = torch.rand(batch, width, generator=torch.Generator().manual_seed(1)) + 0.05
                reference, *_ = _run(variants["split"], x, input_grads)
                cells = []
                for layer in variants.values():
                    tensors, fwd, bwd = _run(layer, x, input_grads)
                    flag = "" if _agree(reference, tensors) else " DIFFERS"
                    cells.append(f"{(fwd + bwd) * 1e3:9.1f} ms{flag:>10s}")
                print(f"{name:12s} {n:2d} {batch:5d}  " + "  ".join(f"{c:>22s}" for c in cells))


_POINT = """
import json, resource, sys, time, warnings, torch
warnings.simplefilter("ignore")
from hqnn_forge.encoding import QuantumEncodingLayer
device, method, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.manual_seed(0)
layer = QuantumEncodingLayer(n_qubits=n, n_layers=2, device_name=device, diff_method=method)
x = torch.rand(64, n) * 2 - 1
layer(x[:2]).sum().backward()
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
start = time.perf_counter()
layer(x).sum().backward()
seconds = time.perf_counter() - start
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"seconds": seconds, "mb": (peak - base) / 1024}))
"""


def crossover(qubits: tuple[int, ...]) -> None:
    print(f"{'qubits':>6s}  {'lightning/adjoint':>22s}  {'default.qubit/backprop':>24s}")
    for n in qubits:
        cells = []
        for device, method in (("lightning.qubit", "adjoint"), ("default.qubit", "backprop")):
            out = subprocess.run(
                [sys.executable, "-c", _POINT, device, method, str(n)],
                capture_output=True,
                text=True,
                check=True,
            )
            point = json.loads(out.stdout.strip().splitlines()[-1])
            cells.append(f"{point['seconds']:7.2f} s {point['mb']:+7.0f} MB")
        print(f"{n:6d}  {cells[0]:>22s}  {cells[1]:>24s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--crossover", action="store_true", help="time a training step by qubit count"
    )
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    if args.crossover:
        crossover((8, 10, 12, 14, 16))
    else:
        compare(qubits=(4, 8), batches=(1, 16, 128))


if __name__ == "__main__":
    main()
