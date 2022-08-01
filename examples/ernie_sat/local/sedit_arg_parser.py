# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


def parse_args():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with acoustic model & vocoder")
    # acoustic model
    parser.add_argument(
        '--am',
        type=str,
        default='fastspeech2_csmsc',
        choices=[
            'speedyspeech_csmsc', 'fastspeech2_csmsc', 'fastspeech2_ljspeech',
            'fastspeech2_aishell3', 'fastspeech2_vctk', 'tacotron2_csmsc',
            'tacotron2_ljspeech', 'tacotron2_aishell3'
        ],
        help='Choose acoustic model type of tts task.')
    parser.add_argument(
        '--am_config',
        type=str,
        default=None,
        help='Config of acoustic model. Use deault config when it is None.')
    parser.add_argument(
        '--am_ckpt',
        type=str,
        default=None,
        help='Checkpoint file of acoustic model.')
    parser.add_argument(
        "--am_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training acoustic model."
    )
    parser.add_argument(
        "--phones_dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--tones_dict", type=str, default=None, help="tone vocabulary file.")
    parser.add_argument(
        "--speaker_dict", type=str, default=None, help="speaker id map file.")

    # vocoder
    parser.add_argument(
        '--voc',
        type=str,
        default='pwgan_aishell3',
        choices=[
            'pwgan_csmsc', 'pwgan_ljspeech', 'pwgan_aishell3', 'pwgan_vctk',
            'mb_melgan_csmsc', 'wavernn_csmsc', 'hifigan_csmsc',
            'hifigan_ljspeech', 'hifigan_aishell3', 'hifigan_vctk',
            'style_melgan_csmsc'
        ],
        help='Choose vocoder type of tts task.')
    parser.add_argument(
        '--voc_config',
        type=str,
        default=None,
        help='Config of voc. Use deault config when it is None.')
    parser.add_argument(
        '--voc_ckpt', type=str, default=None, help='Checkpoint file of voc.')
    parser.add_argument(
        "--voc_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training voc."
    )
    # other
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")

    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--uid", type=str, help="uid")
    parser.add_argument("--new_str", type=str, help="new string")
    parser.add_argument("--prefix", type=str, help="prefix")
    parser.add_argument(
        "--source_lang", type=str, default="english", help="source language")
    parser.add_argument(
        "--target_lang", type=str, default="english", help="target language")
    parser.add_argument("--output_name", type=str, help="output name")
    parser.add_argument("--task_name", type=str, help="task name")

    # pre
    args = parser.parse_args()
    return args
