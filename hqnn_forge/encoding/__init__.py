"""
hqnn_forge.encoding
===================
Quantum feature-map modules for projecting classical tabular vectors into
an n-qubit Hilbert space.

Every public name of the encoder submodules is exported here, and
``from hqnn_forge.encoding import …`` is the documented import path; a new
encoder adds its layer and QNode factory to this list.

Exported symbols
----------------
Encoders:

QuantumEncodingLayer    nn.Module: angle embedding + entangling VQC (TorchLayer).
build_encoding_qnode    Factory that wires the angle-embedding QNode to a device and diff method.
AngleEmbeddingQNode     Alias of build_encoding_qnode.
IQPEncodingLayer        nn.Module: IQP embedding (pairwise ZZ phases) + the same VQC.
build_iqp_qnode         Factory for the IQP-embedding QNode.
AmplitudeEncodingLayer  nn.Module: up to 2**n_qubits features as state amplitudes.
build_amplitude_qnode   Factory for the amplitude-embedding QNode.
DataReuploadingLayer    nn.Module: angle embedding repeated before every layer.
build_data_reuploading_qnode  Factory for the data re-uploading QNode.

Circuit building blocks shared by the encoders:

apply_variational_layers  The entangler + Rot blocks, inside a QNode.
readout_wires           Wires measured under a readout option.
measure_z               The ⟨Z_i⟩ measurements a circuit returns.

Option types, for annotating calls:

DeviceName              Literal of the supported PennyLane devices.
DiffMethod              Literal of the supported differentiation methods.
Entangler               Literal of the entangler options.
Readout                 Literal of the readout options.
RotationAxis            Literal of the embedding rotation axes.
"""

from hqnn_forge.encoding.amplitude_embedding import (
    AmplitudeEncodingLayer,
    build_amplitude_qnode,
)
from hqnn_forge.encoding.angle_embedding import (
    AngleEmbeddingQNode,
    DeviceName,
    DiffMethod,
    Entangler,
    QuantumEncodingLayer,
    Readout,
    RotationAxis,
    apply_variational_layers,
    build_encoding_qnode,
    measure_z,
    readout_wires,
)
from hqnn_forge.encoding.data_reuploading import (
    DataReuploadingLayer,
    build_data_reuploading_qnode,
)
from hqnn_forge.encoding.iqp_embedding import IQPEncodingLayer, build_iqp_qnode

__all__: list[str] = [
    "AmplitudeEncodingLayer",
    "AngleEmbeddingQNode",
    "DataReuploadingLayer",
    "DeviceName",
    "DiffMethod",
    "Entangler",
    "IQPEncodingLayer",
    "QuantumEncodingLayer",
    "Readout",
    "RotationAxis",
    "apply_variational_layers",
    "build_amplitude_qnode",
    "build_data_reuploading_qnode",
    "build_encoding_qnode",
    "build_iqp_qnode",
    "measure_z",
    "readout_wires",
]
