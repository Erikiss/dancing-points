import torch
import onnx


def export_named_onnx_autoregressive_mlp(model, filename, input_names, input_shapes, output_names, feature_dims,
                                         named_model_class,
                                         std_in=None, mean_in=None, std_out=None, mean_out=None, dynamic_axes=None,
                                         meta_dict=None, dummy_inputs=None):
    model.eval()
    if dummy_inputs is None:
        dummy_inputs = tuple(torch.randn(*input_shape, device=list(model.parameters())[0].device) for input_shape in input_shapes)
    named_model = named_model_class(model, feature_dims, std_in, mean_in, std_out, mean_out)
    torch.onnx.export(named_model, dummy_inputs, filename, verbose=False, input_names=input_names,
                      output_names=output_names, dynamic_axes=dynamic_axes)
    if meta_dict is not None:
        add_meta_data(filename, meta_dict)


def add_meta_data(filename, meta_dict):
    model = onnx.load(filename)
    for key, value in meta_dict.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value
    onnx.save(model, filename)
