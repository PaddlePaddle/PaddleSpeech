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
from pathlib import Path

import soundfile as sf
from timer import timer

from paddlespeech.t2s.exps.syn_utils import get_am_output
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_predictor
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.exps.syn_utils import get_voc_output
from paddlespeech.t2s.utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paddle Infernce with acoustic model & vocoder.")
    # acoustic model
    parser.add_argument(
        '--am',
        type=str,
        default='fastspeech2_csmsc',
        choices=[
            'speedyspeech_csmsc', 'fastspeech2_csmsc', 'fastspeech2_aishell3',
            'fastspeech2_vctk', 'tacotron2_csmsc'
        ],
        help='Choose acoustic model type of tts task.')
    parser.add_argument(
        "--phones_dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--tones_dict", type=str, default=None, help="tone vocabulary file.")
    parser.add_argument(
        "--speaker_dict", type=str, default=None, help="speaker id map file.")
    parser.add_argument(
        '--spk_id',
        type=int,
        default=0,
        help='spk id for multi speaker acoustic model')
    # voc
    parser.add_argument(
        '--voc',
        type=str,
        default='pwgan_csmsc',
        choices=[
            'pwgan_csmsc', 'mb_melgan_csmsc', 'hifigan_csmsc', 'pwgan_aishell3',
            'pwgan_vctk', 'wavernn_csmsc'
        ],
        help='Choose vocoder type of tts task.')
    # other
    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        help='Choose model language. zh or en')
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line")
    parser.add_argument(
        "--inference_dir", type=str, help="dir to save inference models")
    parser.add_argument("--output_dir", type=str, help="output dir")
    # inference
    parser.add_argument(
        "--use_trt",
        type=str2bool,
        default=False,
        help="Whether to use inference engin TensorRT.", )
    parser.add_argument(
        "--int8",
        type=str2bool,
        default=False,
        help="Whether to use int8 inference.", )
    parser.add_argument(
        "--fp16",
        type=str2bool,
        default=False,
        help="Whether to use float16 inference.", )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device selected for inference.", )

    args, _ = parser.parse_known_args()
    return args


# only inference for models trained with csmsc now
def main():
    args = parse_args()
    # frontend
    frontend = get_frontend(args)

    # am_predictor
    am_predictor = get_predictor(args, filed='am')
    # model: {model_name}_{dataset}
    am_dataset = args.am[args.am.rindex('_') + 1:]

    # voc_predictor
    voc_predictor = get_predictor(args, filed='voc')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = get_sentences(args)

    merge_sentences = True
    fs = 24000 if am_dataset != 'ljspeech' else 22050
    # warmup
    for utt_id, sentence in sentences[:3]:
        with timer() as t:
            am_output_data = get_am_output(
                args,
                am_predictor=am_predictor,
                frontend=frontend,
                merge_sentences=merge_sentences,
                input=sentence)
            wav = get_voc_output(
                voc_predictor=voc_predictor, input=am_output_data)
        speed = wav.size / t.elapse
        rtf = fs / speed
        print(
            f"{utt_id}, mel: {am_output_data.shape}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )

    print("warm up done!")

    N = 0
    T = 0
    for utt_id, sentence in sentences:
        with timer() as t:
            am_output_data = get_am_output(
                args,
                am_predictor=am_predictor,
                frontend=frontend,
                merge_sentences=merge_sentences,
                input=sentence)
            wav = get_voc_output(
                voc_predictor=voc_predictor, input=am_output_data)

        N += wav.size
        T += t.elapse
        speed = wav.size / t.elapse
        rtf = fs / speed

        sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=24000)
        print(
            f"{utt_id}, mel: {am_output_data.shape}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )

        print(f"{utt_id} done!")
    print(f"generation speed: {N / T}Hz, RTF: {fs / (N / T) }")


if __name__ == "__main__":
    main()
