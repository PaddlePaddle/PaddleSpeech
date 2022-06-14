#!/usr/bin/env python3 -W ignore::DeprecationWarning
# https://github.com/jiangjiajun/PaddleUtils/blob/main/paddle/README.md#1-%E8%A3%81%E5%89%AApaddle%E6%A8%A1%E5%9E%8B
import argparse
import sys
from typing import List

# paddle prune model.

def prepend_feed_ops(program,
                     feed_target_names: List[str],
                     feed_holder_name='feed'):
    import paddle.fluid.core as core
    if len(feed_target_names) == 0:
        return

    global_block = program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True, )

    for i, name in enumerate(feed_target_names, 0):
        if not global_block.has_var(name):
            print(
                f"The input[{i}]: '{name}' doesn't exist in pruned inference program, which will be ignored in new saved model."
            )
            continue

        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i}, )


def append_fetch_ops(program,
                     fetch_target_names: List[str],
                     fetch_holder_name='fetch'):
    """in the place, we will add the fetch op

    Args:
        program (_type_): inference program
        fetch_target_names (List[str]): target names
        fetch_holder_name (str, optional): fetch op name. Defaults to 'fetch'.
    """
    import paddle.fluid.core as core
    global_block = program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True, )

    print(f"the len of fetch_target_names: {len(fetch_target_names)}")

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i}, )


def insert_fetch(program,
                 fetch_target_names: List[str],
                 fetch_holder_name='fetch'):
    """in the place, we will add the fetch op

    Args:
        program (_type_): inference program
        fetch_target_names (List[str]): target names
        fetch_holder_name (str, optional): fetch op name. Defaults to 'fetch'.
    """
    global_block = program.global_block()

    # remove fetch
    need_to_remove_op_index = []
    for i, op in enumerate(global_block.ops):
        if op.type == 'fetch':
            need_to_remove_op_index.append(i)

    for index in reversed(need_to_remove_op_index):
        global_block._remove_op(index)

    program.desc.flush()

    # append new fetch
    append_fetch_ops(program, fetch_target_names, fetch_holder_name)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        required=True,
        help='Path of directory saved the input model.')
    parser.add_argument(
        '--model_filename', required=True, help='model.pdmodel.')
    parser.add_argument(
        '--params_filename', required=True, help='model.pdiparams.')
    parser.add_argument(
        '--output_names',
        required=True,
        help='The outputs of model. sep by comma')
    parser.add_argument(
        '--save_dir',
        required=True,
        help='directory to save the exported model.')
    parser.add_argument('--debug', default=False, help='output debug info.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    args.output_names = args.output_names.split(",")

    if len(set(args.output_names)) < len(args.output_names):
        print(
            f"[ERROR] There's dumplicate name in --output_names {args.output_names}, which is not allowed."
        )
        sys.exit(-1)

    import paddle
    paddle.enable_static()
    # hack prepend_feed_ops
    paddle.fluid.io.prepend_feed_ops = prepend_feed_ops

    import paddle.fluid as fluid

    print("start to load paddle model")
    exe = fluid.Executor(fluid.CPUPlace())
    prog, ipts, outs = fluid.io.load_inference_model(
        args.model_dir,
        exe,
        model_filename=args.model_filename,
        params_filename=args.params_filename)

    print("start to load insert fetch op")
    new_outputs = []
    insert_fetch(prog, args.output_names)
    for out_name in args.output_names:
        new_outputs.append(prog.global_block().var(out_name))

    # not equal to paddle.static.save_inference_model
    fluid.io.save_inference_model(
        args.save_dir,
        ipts,
        new_outputs,
        exe,
        prog,
        model_filename=args.model_filename,
        params_filename=args.params_filename)

    if args.debug:
        for op in prog.global_block().ops:
            print(op)
