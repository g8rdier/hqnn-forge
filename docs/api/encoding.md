# Encoding layers

::: hqnn_forge.encoding

## IQP embedding

`IQPEncodingLayer` is not re-exported from `hqnn_forge.encoding` (see #169);
import it from its module.

::: hqnn_forge.encoding.iqp_embedding
    options:
      show_root_heading: false
      members:
        - IQPEncodingLayer
        - build_iqp_qnode

## Building blocks

The pieces every encoder shares, for writing a new one (see the contributor
docs on extending the library).

::: hqnn_forge.encoding.angle_embedding.apply_variational_layers

::: hqnn_forge.encoding.angle_embedding.validate_circuit_options

::: hqnn_forge.encoding.angle_embedding.check_inputs

::: hqnn_forge.encoding.angle_embedding.DeviceName

::: hqnn_forge.encoding.angle_embedding.DiffMethod

::: hqnn_forge.encoding.data_reuploading.input_scaling_shape
