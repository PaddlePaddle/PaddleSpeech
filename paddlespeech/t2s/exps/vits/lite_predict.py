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
from pathlib import Path

import soundfile as sf
from timer import timer

from paddlespeech.t2s.exps.lite_syn_utils import get_lite_am_output
from paddlespeech.t2s.exps.lite_syn_utils import get_lite_predictor
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paddle Infernce with acoustic model & vocoder.")
    # acoustic model
    parser.add_argument(
        '--am',
        type=str,
        default='vits_csmsc',
        choices=[
            'vits_csmsc',
            'vits_aishell3',
        ],
        help='Choose acoustic model type of tts task.')
    parser.add_argument(
        "--phones_dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--speaker_dict", type=str, default=None, help="speaker id map file.")
    parser.add_argument(
        '--spk_id',
        type=int,
        default=0,
        help='spk id for multi speaker acoustic model')
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
        "--add-blank",
        type=str2bool,
        default=True,
        help="whether to add blank between phones")
    parser.add_argument(
        "--inference_dir", type=str, help="dir to save inference models")
    parser.add_argument("--output_dir", type=str, help="output dir")

    args, _ = parser.parse_known_args()
    return args


# only inference for models trained with csmsc now
def main():
    args = parse_args()

    # frontend
    frontend = get_frontend(
        lang=args.lang,
        phones_dict=args.phones_dict)

    # am_predictor
    # vits can only run in arm
    am_predictor = get_lite_predictor(
        model_dir=args.inference_dir, model_file=args.am + "_arm.nb")
    # model: {model_name}_{dataset}
    am_dataset = args.am[args.am.rindex('_') + 1:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = get_sentences(text_file=args.text, lang=args.lang)

    merge_sentences = True
    add_blank = args.add_blank
    fs = 22050
    # warmup
    for utt_id, sentence in sentences[:3]:
        with timer() as t:
            wav = get_lite_am_output(
                input=sentence,
                am_predictor=am_predictor,
                am=args.am,
                frontend=frontend,
                lang=args.lang,
                merge_sentences=merge_sentences,
                speaker_dict=args.speaker_dict,
                spk_id=args.spk_id,
                add_blank=add_blank)

        speed = wav.size / t.elapse
        rtf = fs / speed
        print(
            f"{utt_id}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )

    print("warm up done!")

    N = 0
    T = 0
    for utt_id, sentence in sentences:
        with timer() as t:
            wav = get_lite_am_output(
                input=sentence,
                am_predictor=am_predictor,
                am=args.am,
                frontend=frontend,
                lang=args.lang,
                merge_sentences=merge_sentences,
                speaker_dict=args.speaker_dict,
                spk_id=args.spk_id,
                add_blank=add_blank)

        N += wav.size
        T += t.elapse
        speed = wav.size / t.elapse
        rtf = fs / speed

        sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=fs)
        print(
            f"{utt_id}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )

        print(f"{utt_id} done!")
    print(f"generation speed: {N / T}Hz, RTF: {fs / (N / T) }")


if __name__ == "__main__":
    main()
