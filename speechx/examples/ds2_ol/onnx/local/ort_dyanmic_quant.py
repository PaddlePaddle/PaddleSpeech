#!/usr/bin/env python3
import argparse

from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import QuantType


def quantize_onnx_model(onnx_model_path,
                        quantized_model_path,
                        nodes_to_exclude=[]):
    print("Starting quantization...")

    quantize_dynamic(
        onnx_model_path,
        quantized_model_path,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude)

    print(f"Quantized model saved to: {quantized_model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-in",
        type=str,
        required=True,
        help="ONNX model", )
    parser.add_argument(
        "--model-out",
        type=str,
        required=True,
        default='model.quant.onnx',
        help="ONNX model", )
    parser.add_argument(
        "--nodes-to-exclude",
        type=str,
        required=True,
        help="nodes to exclude. e.g. conv,linear.", )

    args = parser.parse_args()

    nodes_to_exclude = args.nodes_to_exclude.split(',')
    quantize_onnx_model(args.model_in, args.model_out, nodes_to_exclude)


if __name__ == "__main__":
    main()
