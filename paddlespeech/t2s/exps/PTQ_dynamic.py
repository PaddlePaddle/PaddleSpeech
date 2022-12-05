# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

import paddle
from paddleslim.quant import quant_post_dynamic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paddle Slim with acoustic model & vocoder.")
    # acoustic model
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
            'tacotron2_csmsc',
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
            'wavernn_csmsc',
        ],
        help='Choose model type of tts task.')

    parser.add_argument(
        "--inference_dir", type=str, help="dir to save inference models")
    parser.add_argument(
        "--weight_bits",
        type=int,
        default=8,
        choices=[8, 16],
        help="The bits for the quantized weight, and it should be 8 or 16. Default is 8.",
    )

    args, _ = parser.parse_known_args()
    return args


# only inference for models trained with csmsc now
def main():
    args = parse_args()
    paddle.enable_static()
    quant_post_dynamic(
        model_dir=args.inference_dir,
        save_model_dir=args.inference_dir,
        model_filename=args.model_name + ".pdmodel",
        params_filename=args.model_name + ".pdiparams",
        save_model_filename=args.model_name + "_" + str(args.weight_bits) +
        "bits.pdmodel",
        save_params_filename=args.model_name + "_" + str(args.weight_bits) +
        "bits.pdiparams",
        weight_bits=args.weight_bits, )


if __name__ == "__main__":
    main()
