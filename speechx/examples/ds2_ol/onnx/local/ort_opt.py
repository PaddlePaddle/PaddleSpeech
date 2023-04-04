#!/usr/bin/env python3
import argparse

import onnxruntime as ort

# onnxruntime optimizer.
# https://onnxruntime.ai/docs/performance/graph-optimizations.html
# https://onnxruntime.ai/docs/api/python/api_summary.html#api


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_in',
                        required=True,
                        type=str,
                        help='Path to onnx model.')
    parser.add_argument('--opt_level',
                        required=True,
                        type=int,
                        default=0,
                        choices=[0, 1, 2],
                        help='Path to onnx model.')
    parser.add_argument('--model_out',
                        required=True,
                        help='path to save the optimized model.')
    parser.add_argument('--debug', default=False, help='output debug info.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    sess_options = ort.SessionOptions()

    # Set graph optimization level
    print(f"opt level: {args.opt_level}")
    if args.opt_level == 0:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif args.opt_level == 1:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = args.model_out

    session = ort.InferenceSession(args.model_in, sess_options)
