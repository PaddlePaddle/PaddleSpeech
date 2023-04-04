#!/usr/bin/env python3 -W ignore::DeprecationWarning
# prune model by output names
import argparse
import copy
import sys

import onnx


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        required=True,
                        help='Path of directory saved the input model.')
    parser.add_argument('--output_names',
                        required=True,
                        nargs='+',
                        help='The outputs of pruned model.')
    parser.add_argument('--save_file',
                        required=True,
                        help='Path to save the new onnx model.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if len(set(args.output_names)) < len(args.output_names):
        print(
            "[ERROR] There's dumplicate name in --output_names, which is not allowed."
        )
        sys.exit(-1)

    model = onnx.load(args.model)

    # collect all node outputs and graph output
    output_tensor_names = set()
    for node in model.graph.node:
        for out in node.output:
            # may contain model output
            output_tensor_names.add(out)

    # for out in model.graph.output:
    #     output_tensor_names.add(out.name)

    for output_name in args.output_names:
        if output_name not in output_tensor_names:
            print(
                "[ERROR] Cannot find output tensor name '{}' in onnx model graph."
                .format(output_name))
            sys.exit(-1)

    output_node_indices = set()  # has output names
    output_to_node = dict()  # all node outputs
    for i, node in enumerate(model.graph.node):
        for out in node.output:
            output_to_node[out] = i
            if out in args.output_names:
                output_node_indices.add(i)

    # from outputs find all the ancestors
    reserved_node_indices = copy.deepcopy(
        output_node_indices)  # nodes need to keep
    reserved_inputs = set()  # model input to keep
    new_output_node_indices = copy.deepcopy(output_node_indices)

    while True and len(new_output_node_indices) > 0:
        output_node_indices = copy.deepcopy(new_output_node_indices)

        new_output_node_indices = set()

        for out_node_idx in output_node_indices:
            # backtrace to parenet
            for ipt in model.graph.node[out_node_idx].input:
                if ipt in output_to_node:
                    reserved_node_indices.add(output_to_node[ipt])
                    new_output_node_indices.add(output_to_node[ipt])
                else:
                    reserved_inputs.add(ipt)

    num_inputs = len(model.graph.input)
    num_outputs = len(model.graph.output)
    num_nodes = len(model.graph.node)
    print(
        f"old graph has {num_inputs} inputs, {num_outputs} outpus, {num_nodes} nodes"
    )
    print(f"{len(reserved_node_indices)} node to keep.")

    # del node not to keep
    for idx in range(num_nodes - 1, -1, -1):
        if idx not in reserved_node_indices:
            del model.graph.node[idx]

    # del graph input not to keep
    for idx in range(num_inputs - 1, -1, -1):
        if model.graph.input[idx].name not in reserved_inputs:
            del model.graph.input[idx]

    # del old graph outputs
    for i in range(num_outputs):
        del model.graph.output[0]

    # new graph output as user input
    for out in args.output_names:
        model.graph.output.extend([onnx.ValueInfoProto(name=out)])

    # infer shape
    try:
        from onnx_infer_shape import SymbolicShapeInference
        model = SymbolicShapeInference.infer_shapes(model,
                                                    int_max=2**31 - 1,
                                                    auto_merge=True,
                                                    guess_output_rank=False,
                                                    verbose=1)
    except Exception as e:
        print(f"skip infer shape step: {e}")

    # check onnx model
    onnx.checker.check_model(model)
    # save onnx model
    onnx.save(model, args.save_file)
    print("[Finished] The new model saved in {}.".format(args.save_file))
    print("[DEBUG INFO] The inputs of new model: {}".format(
        [x.name for x in model.graph.input]))
    print("[DEBUG INFO] The outputs of new model: {}".format(
        [x.name for x in model.graph.output]))
