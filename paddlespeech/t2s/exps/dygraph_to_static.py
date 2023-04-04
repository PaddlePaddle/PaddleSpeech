# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.exps.syn_utils import am_to_static
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.exps.syn_utils import voc_to_static


def am_dygraph_to_static(args):
    with open(args.am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))
    am_inference = get_am_inference(am=args.am,
                                    am_config=am_config,
                                    am_ckpt=args.am_ckpt,
                                    am_stat=args.am_stat,
                                    phones_dict=args.phones_dict,
                                    tones_dict=args.tones_dict,
                                    speaker_dict=args.speaker_dict)
    print("acoustic model done!")

    # dygraph to static
    am_inference = am_to_static(am_inference=am_inference,
                                am=args.am,
                                inference_dir=args.inference_dir,
                                speaker_dict=args.speaker_dict)
    print("finish to convert dygraph acoustic model to static!")


def voc_dygraph_to_static(args):
    with open(args.voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))
    voc_inference = get_voc_inference(voc=args.voc,
                                      voc_config=voc_config,
                                      voc_ckpt=args.voc_ckpt,
                                      voc_stat=args.voc_stat)
    print("voc done!")

    # dygraph to static
    voc_inference = voc_to_static(voc_inference=voc_inference,
                                  voc=args.voc,
                                  inference_dir=args.inference_dir)
    print("finish to convert dygraph vocoder model to static!")


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="Synthesize with acoustic model & vocoder")
    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=["am", "voc"],
        help='Choose the model type of dynamic to static, am or voc')
    # acoustic model
    parser.add_argument('--am',
                        type=str,
                        default='fastspeech2_csmsc',
                        choices=[
                            'speedyspeech_csmsc',
                            'speedyspeech_aishell3',
                            'fastspeech2_csmsc',
                            'fastspeech2_ljspeech',
                            'fastspeech2_aishell3',
                            'fastspeech2_vctk',
                            'tacotron2_csmsc',
                            'tacotron2_ljspeech',
                            'fastspeech2_mix',
                            'fastspeech2_canton',
                            'fastspeech2_male-zh',
                            'fastspeech2_male-en',
                            'fastspeech2_male-mix',
                        ],
                        help='Choose acoustic model type of tts task.')
    parser.add_argument('--am_config',
                        type=str,
                        default=None,
                        help='Config of acoustic model.')
    parser.add_argument('--am_ckpt',
                        type=str,
                        default=None,
                        help='Checkpoint file of acoustic model.')
    parser.add_argument(
        "--am_stat",
        type=str,
        default=None,
        help=
        "mean and standard deviation used to normalize spectrogram when training acoustic model."
    )
    parser.add_argument("--phones_dict",
                        type=str,
                        default=None,
                        help="phone vocabulary file.")
    parser.add_argument("--tones_dict",
                        type=str,
                        default=None,
                        help="tone vocabulary file.")
    parser.add_argument("--speaker_dict",
                        type=str,
                        default=None,
                        help="speaker id map file.")
    # vocoder
    parser.add_argument('--voc',
                        type=str,
                        default='pwgan_csmsc',
                        choices=[
                            'pwgan_csmsc',
                            'pwgan_ljspeech',
                            'pwgan_aishell3',
                            'pwgan_vctk',
                            'mb_melgan_csmsc',
                            'style_melgan_csmsc',
                            'hifigan_csmsc',
                            'hifigan_ljspeech',
                            'hifigan_aishell3',
                            'hifigan_vctk',
                            'wavernn_csmsc',
                            'pwgan_male',
                            'hifigan_male',
                            'pwgan_opencpop',
                            'hifigan_opencpop',
                        ],
                        help='Choose vocoder type of tts task.')
    parser.add_argument('--voc_config',
                        type=str,
                        default=None,
                        help='Config of voc.')
    parser.add_argument('--voc_ckpt',
                        type=str,
                        default=None,
                        help='Checkpoint file of voc.')
    parser.add_argument(
        "--voc_stat",
        type=str,
        default=None,
        help=
        "mean and standard deviation used to normalize spectrogram when training voc."
    )
    # other
    parser.add_argument("--inference_dir",
                        type=str,
                        default=None,
                        help="dir to save inference models")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.type == "am":
        am_dygraph_to_static(args)
    elif args.type == "voc":
        voc_dygraph_to_static(args)
    else:
        print("type should be in ['am', 'voc'] !")


if __name__ == "__main__":
    main()
