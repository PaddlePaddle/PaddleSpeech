#!/usr/bin/env python3 -W ignore::DeprecationWarning
# https://github.com/jiangjiajun/PaddleUtils/blob/main/paddle/README.md#2-%E4%BF%AE%E6%94%B9paddle%E6%A8%A1%E5%9E%8B%E8%BE%93%E5%85%A5shape
import argparse

# paddle inference shape

def process_old_ops_desc(program):
    """set matmul op head_number attr to 1 is not exist.

    Args:
        program (_type_): _description_
    """
    for i in range(len(program.blocks[0].ops)):
        if program.blocks[0].ops[i].type == "matmul":
            if not program.blocks[0].ops[i].has_attr("head_number"):
                program.blocks[0].ops[i]._set_attr("head_number", 1)


def infer_shape(program, input_shape_dict):
    # 2002002
    model_version = program.desc._version()
    # 2.2.2
    paddle_version = paddle.__version__
    major_ver = model_version // 1000000
    minor_ver = (model_version - major_ver * 1000000) // 1000
    patch_ver = model_version - major_ver * 1000000 - minor_ver * 1000
    model_version = "{}.{}.{}".format(major_ver, minor_ver, patch_ver)
    if model_version != paddle_version:
        print(
            f"[WARNING] The model is saved by paddlepaddle v{model_version}, but now your paddlepaddle is version of {paddle_version}, this difference may cause error, it is recommend you reinstall a same version of paddlepaddle for this model"
        )

    OP_WITHOUT_KERNEL_SET = {
        'feed', 'fetch', 'recurrent', 'go', 'rnn_memory_helper_grad',
        'conditional_block', 'while', 'send', 'recv', 'listen_and_serv',
        'fl_listen_and_serv', 'ncclInit', 'select', 'checkpoint_notify',
        'gen_bkcl_id', 'c_gen_bkcl_id', 'gen_nccl_id', 'c_gen_nccl_id',
        'c_comm_init', 'c_sync_calc_stream', 'c_sync_comm_stream',
        'queue_generator', 'dequeue', 'enqueue', 'heter_listen_and_serv',
        'c_wait_comm', 'c_wait_compute', 'c_gen_hccl_id', 'c_comm_init_hccl',
        'copy_cross_scope'
    }

    for k, v in input_shape_dict.items():
        program.blocks[0].var(k).desc.set_shape(v)

    for i in range(len(program.blocks)):
        for j in range(len(program.blocks[0].ops)):
            # for ops
            if program.blocks[i].ops[j].type in OP_WITHOUT_KERNEL_SET:
                print(f"not infer: {program.blocks[i].ops[j].type} op")
                continue
            print(f"infer: {program.blocks[i].ops[j].type} op")
            program.blocks[i].ops[j].desc.infer_shape(program.blocks[i].desc)


def parse_arguments():
    # python pd_infer_shape.py --model_dir data/exp/deepspeech2_online/checkpoints \
    #                          --model_filename avg_1.jit.pdmodel\
    #                          --params_filename avg_1.jit.pdiparams \
    #                          --save_dir . \
    #                          --input_shape_dict="{'audio_chunk':[1,-1,161], 'audio_chunk_lens':[1], 'chunk_state_c_box':[5, 1, 1024], 'chunk_state_h_box':[5,1,1024]}"
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
        '--save_dir',
        required=True,
        help='directory to save the exported model.')
    parser.add_argument(
        '--input_shape_dict', required=True, help="The new shape information.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    import paddle
    paddle.enable_static()
    import paddle.fluid as fluid

    input_shape_dict_str = args.input_shape_dict
    input_shape_dict = eval(input_shape_dict_str)

    print("Start to load paddle model...")
    exe = fluid.Executor(fluid.CPUPlace())

    prog, ipts, outs = fluid.io.load_inference_model(
        args.model_dir,
        exe,
        model_filename=args.model_filename,
        params_filename=args.params_filename)

    process_old_ops_desc(prog)
    infer_shape(prog, input_shape_dict)

    fluid.io.save_inference_model(
        args.save_dir,
        ipts,
        outs,
        exe,
        prog,
        model_filename=args.model_filename,
        params_filename=args.params_filename)
