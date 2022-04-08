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

import numpy
import soundfile as sf
from paddle import inference
from timer import timer

from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.utils import str2bool


def get_predictor(args, filed='am'):
    full_name = ''
    if filed == 'am':
        full_name = args.am
    elif filed == 'voc':
        full_name = args.voc
    model_name = full_name[:full_name.rindex('_')]
    config = inference.Config(
        str(Path(args.inference_dir) / (full_name + ".pdmodel")),
        str(Path(args.inference_dir) / (full_name + ".pdiparams")))
    if args.device == "gpu":
        config.enable_use_gpu(100, 0)
    elif args.device == "cpu":
        config.disable_gpu()
    # This line must be commented for fastspeech2, if not, it will OOM
    if model_name != 'fastspeech2':
        config.enable_memory_optim()
    predictor = inference.create_predictor(config)
    return predictor


def get_am_output(args, am_predictor, frontend, merge_sentences, input):
    am_name = args.am[:args.am.rindex('_')]
    am_dataset = args.am[args.am.rindex('_') + 1:]
    am_input_names = am_predictor.get_input_names()
    get_tone_ids = False
    get_spk_id = False
    if am_name == 'speedyspeech':
        get_tone_ids = True
    if am_dataset in {"aishell3", "vctk"} and args.speaker_dict:
        get_spk_id = True
        spk_id = numpy.array([args.spk_id])
    if args.lang == 'zh':
        input_ids = frontend.get_input_ids(
            input, merge_sentences=merge_sentences, get_tone_ids=get_tone_ids)
        phone_ids = input_ids["phone_ids"]
    elif args.lang == 'en':
        input_ids = frontend.get_input_ids(
            input, merge_sentences=merge_sentences)
        phone_ids = input_ids["phone_ids"]
    else:
        print("lang should in {'zh', 'en'}!")

    if get_tone_ids:
        tone_ids = input_ids["tone_ids"]
        tones = tone_ids[0].numpy()
        tones_handle = am_predictor.get_input_handle(am_input_names[1])
        tones_handle.reshape(tones.shape)
        tones_handle.copy_from_cpu(tones)
    if get_spk_id:
        spk_id_handle = am_predictor.get_input_handle(am_input_names[1])
        spk_id_handle.reshape(spk_id.shape)
        spk_id_handle.copy_from_cpu(spk_id)
    phones = phone_ids[0].numpy()
    phones_handle = am_predictor.get_input_handle(am_input_names[0])
    phones_handle.reshape(phones.shape)
    phones_handle.copy_from_cpu(phones)

    am_predictor.run()
    am_output_names = am_predictor.get_output_names()
    am_output_handle = am_predictor.get_output_handle(am_output_names[0])
    am_output_data = am_output_handle.copy_to_cpu()
    return am_output_data


def get_voc_output(args, voc_predictor, input):
    voc_input_names = voc_predictor.get_input_names()
    mel_handle = voc_predictor.get_input_handle(voc_input_names[0])
    mel_handle.reshape(input.shape)
    mel_handle.copy_from_cpu(input)

    voc_predictor.run()
    voc_output_names = voc_predictor.get_output_names()
    voc_output_handle = voc_predictor.get_output_handle(voc_output_names[0])
    wav = voc_output_handle.copy_to_cpu()
    return wav


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paddle Infernce with speedyspeech & parallel wavegan.")
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
                args, voc_predictor=voc_predictor, input=am_output_data)
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
                args, voc_predictor=voc_predictor, input=am_output_data)

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
