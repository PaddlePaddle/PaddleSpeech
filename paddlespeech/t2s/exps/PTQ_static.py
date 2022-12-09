import argparse
import random

import jsonlines
import numpy as np
import paddle
from paddleslim.quant import quant_post_static

from paddlespeech.t2s.exps.syn_utils import get_dev_dataloader
from paddlespeech.t2s.utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paddle Slim Static with acoustic model & vocoder.")

    parser.add_argument(
        "--batch_size", type=int, default=1, help="Minibatch size.")
    parser.add_argument("--batch_num", type=int, default=1, help="Batch number")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu=0, use cpu.")
    # model_path save_path
    parser.add_argument(
        "--inference_dir", type=str, help="dir to save inference models")
    parser.add_argument(
        '--model_name',
        type=str,
        default='fastspeech2_csmsc',
        choices=[
            'speedyspeech_csmsc',
            'fastspeech2_csmsc',
            'fastspeech2_aishell3',
            'fastspeech2_ljspeech',
            'fastspeech2_vctk',
            'fastspeech2_mix',
            'pwgan_csmsc',
            'pwgan_aishell3',
            'pwgan_ljspeech',
            'pwgan_vctk',
            'mb_melgan_csmsc',
            'hifigan_csmsc',
            'hifigan_aishell3',
            'hifigan_ljspeech',
            'hifigan_vctk',
        ],
        help='Choose model type of tts task.')

    parser.add_argument(
        "--algo", type=str, default='avg', help="calibration algorithm.")
    parser.add_argument(
        "--round_type",
        type=str,
        default='round',
        help="The method of converting the quantized weights.")
    parser.add_argument(
        "--hist_percent",
        type=float,
        default=0.9999,
        help="The percentile of algo:hist.")
    parser.add_argument(
        "--is_full_quantize",
        type=str2bool,
        default=False,
        help="Whether is full quantization or not.")
    parser.add_argument(
        "--bias_correction",
        type=str2bool,
        default=False,
        help="Whether to use bias correction.")
    parser.add_argument(
        "--ce_test", type=str2bool, default=False, help="Whether to CE test.")
    parser.add_argument(
        "--onnx_format",
        type=str2bool,
        default=False,
        help="Whether to export the quantized model with format of ONNX.")
    parser.add_argument(
        "--phones-dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--speaker-dict",
        type=str,
        default=None,
        help="speaker id map file for multiple speaker model.")
    parser.add_argument("--dev-metadata", type=str, help="dev data.")
    parser.add_argument(
        "--quantizable_op_type",
        type=list,
        nargs='+',
        default=[
            "conv2d_transpose", "conv2d", "depthwise_conv2d", "mul", "matmul",
            "matmul_v2"
        ],
        help="The list of op types that will be quantized.")

    args = parser.parse_args()
    return args


def quantize(args):
    shuffle = True
    if args.ce_test:
        # set seed
        seed = 111
        np.random.seed(seed)
        paddle.seed(seed)
        random.seed(seed)
        shuffle = False

    place = paddle.CUDAPlace(0) if args.ngpu > 0 else paddle.CPUPlace()
    with jsonlines.open(args.dev_metadata, 'r') as reader:
        dev_metadata = list(reader)

    dataloader = get_dev_dataloader(
        dev_metadata=dev_metadata,
        am=args.model_name,
        batch_size=args.batch_size,
        speaker_dict=args.speaker_dict,
        shuffle=shuffle)

    exe = paddle.static.Executor(place)
    exe.run()

    print("onnx_format:", args.onnx_format)

    quant_post_static(
        executor=exe,
        model_dir=args.inference_dir,
        quantize_model_path=args.inference_dir + "/" + args.model_name +
        "_quant",
        data_loader=dataloader,
        model_filename=args.model_name + ".pdmodel",
        params_filename=args.model_name + ".pdiparams",
        save_model_filename=args.model_name + ".pdmodel",
        save_params_filename=args.model_name + ".pdiparams",
        batch_size=args.batch_size,
        algo=args.algo,
        round_type=args.round_type,
        hist_percent=args.hist_percent,
        is_full_quantize=args.is_full_quantize,
        bias_correction=args.bias_correction,
        onnx_format=args.onnx_format,
        quantizable_op_type=args.quantizable_op_type)


def main():
    args = parse_args()
    new_quantizable_op_type = []
    for item in args.quantizable_op_type:
        new_quantizable_op_type.append(''.join(item))
    args.quantizable_op_type = new_quantizable_op_type
    paddle.enable_static()
    quantize(args)


if __name__ == '__main__':
    main()
