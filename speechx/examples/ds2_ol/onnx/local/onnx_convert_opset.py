#!/usr/bin/env python3
import argparse

import onnx
from onnx import version_converter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=__doc__)
    parser.add_argument("--model-file",
                        type=str,
                        required=True,
                        help='path/to/the/model.onnx.')
    parser.add_argument("--save-model",
                        type=str,
                        required=True,
                        help='path/to/saved/model.onnx.')
    # Models must be opset10 or higher to be quantized.
    parser.add_argument("--target-opset",
                        type=int,
                        default=11,
                        help='path/to/the/model.onnx.')

    args = parser.parse_args()

    print(f"to opset: {args.target_opset}")

    # Preprocessing: load the model to be converted.
    model_path = args.model_file
    original_model = onnx.load(model_path)

    # print('The model before conversion:\n{}'.format(original_model))

    # A full list of supported adapters can be found here:
    # https://github.com/onnx/onnx/blob/main/onnx/version_converter.py#L21
    # Apply the version conversion on the original model
    converted_model = version_converter.convert_version(original_model,
                                                        args.target_opset)

    # print('The model after conversion:\n{}'.format(converted_model))
    onnx.save(converted_model, args.save_model)
