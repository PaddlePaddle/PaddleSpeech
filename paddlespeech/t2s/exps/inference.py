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

import paddle
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
            'speedyspeech_csmsc',
            'fastspeech2_csmsc',
            'fastspeech2_aishell3',
            'fastspeech2_ljspeech',
            'fastspeech2_vctk',
            'tacotron2_csmsc',
            'fastspeech2_mix',
            'fastspeech2_male-zh',
            'fastspeech2_male-en',
            'fastspeech2_male-mix',
            'fastspeech2_canton',
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
            'pwgan_male',
            'hifigan_male',
        ],
        help='Choose vocoder type of tts task.')
    # other
    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        help='Choose model language. zh or en or mix')
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
        help="whether to use TensorRT or not in GPU", )
    parser.add_argument(
        "--use_mkldnn",
        type=str2bool,
        default=False,
        help="whether to use MKLDNN or not in CPU.", )
    parser.add_argument(
        "--precision",
        type=str,
        default='fp32',
        choices=['fp32', 'fp16', 'bf16', 'int8'],
        help="mode of running")
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu", "xpu", "npu", "mlu", "gcu"],
        help="Device selected for inference.", )
    parser.add_argument('--cpu_threads', type=int, default=1)

    args, _ = parser.parse_known_args()
    return args


# only inference for models trained with csmsc now
def main():
    args = parse_args()

    paddle.set_device(args.device)

    # frontend
    frontend = get_frontend(
        lang=args.lang,
        phones_dict=args.phones_dict,
        tones_dict=args.tones_dict)

    # am_predictor
    am_predictor = get_predictor(
        model_dir=args.inference_dir,
        model_file=args.am + ".pdmodel",
        params_file=args.am + ".pdiparams",
        device=args.device,
        use_trt=args.use_trt,
        use_mkldnn=args.use_mkldnn,
        cpu_threads=args.cpu_threads,
        precision=args.precision)
    # model: {model_name}_{dataset}
    am_dataset = args.am[args.am.rindex('_') + 1:]

    # voc_predictor
    voc_predictor = get_predictor(
        model_dir=args.inference_dir,
        model_file=args.voc + ".pdmodel",
        params_file=args.voc + ".pdiparams",
        device=args.device,
        use_trt=args.use_trt,
        use_mkldnn=args.use_mkldnn,
        cpu_threads=args.cpu_threads,
        precision=args.precision)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = get_sentences(text_file=args.text, lang=args.lang)

    merge_sentences = True
    fs = 24000 if am_dataset != 'ljspeech' else 22050
    # warmup
    for utt_id, sentence in sentences[:3]:
        with timer() as t:
            mel = get_am_output(
                input=sentence,
                am_predictor=am_predictor,
                am=args.am,
                frontend=frontend,
                lang=args.lang,
                merge_sentences=merge_sentences,
                speaker_dict=args.speaker_dict,
                spk_id=args.spk_id, )
            wav = get_voc_output(voc_predictor=voc_predictor, input=mel)
        speed = wav.size / t.elapse
        rtf = fs / speed
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )

    print("warm up done!")

    N = 0
    T = 0
    for utt_id, sentence in sentences:
        with timer() as t:
            mel = get_am_output(
                input=sentence,
                am_predictor=am_predictor,
                am=args.am,
                frontend=frontend,
                lang=args.lang,
                merge_sentences=merge_sentences,
                speaker_dict=args.speaker_dict,
                spk_id=args.spk_id, )
            wav = get_voc_output(voc_predictor=voc_predictor, input=mel)

        N += wav.size
        T += t.elapse
        speed = wav.size / t.elapse
        rtf = fs / speed

        sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=fs)
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )

        print(f"{utt_id} done!")
    print(f"generation speed: {N / T}Hz, RTF: {fs / (N / T) }")


if __name__ == "__main__":
    main()
