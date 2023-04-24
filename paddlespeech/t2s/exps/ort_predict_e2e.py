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

import numpy as np
import paddle
import soundfile as sf
from timer import timer

from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.exps.syn_utils import get_sess
from paddlespeech.t2s.exps.syn_utils import run_frontend
from paddlespeech.t2s.utils import str2bool


def ort_predict(args):

    # frontend
    frontend = get_frontend(
        lang=args.lang,
        phones_dict=args.phones_dict,
        tones_dict=args.tones_dict)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences = get_sentences(text_file=args.text, lang=args.lang)

    am_name = args.am[:args.am.rindex('_')]
    am_dataset = args.am[args.am.rindex('_') + 1:]
    fs = 24000 if am_dataset != 'ljspeech' else 22050

    am_sess = get_sess(
        model_path=str(Path(args.inference_dir) / (args.am + '.onnx')),
        device=args.device,
        cpu_threads=args.cpu_threads,
        use_trt=args.use_trt)

    # vocoder
    voc_sess = get_sess(
        model_path=str(Path(args.inference_dir) / (args.voc + '.onnx')),
        device=args.device,
        cpu_threads=args.cpu_threads,
        use_trt=args.use_trt)

    merge_sentences = True

    # frontend warmup
    # Loading model cost 0.5+ seconds
    if args.lang == 'zh':
        frontend.get_input_ids(
            "你好，欢迎使用飞桨框架进行深度学习研究！", merge_sentences=merge_sentences)
    else:
        frontend.get_input_ids(
            "hello, thank you, thank you very much",
            merge_sentences=merge_sentences)

    # am warmup
    spk_id = [args.spk_id]
    for T in [27, 38, 54]:
        am_input_feed = {}
        if am_name == 'fastspeech2':
            if args.lang == 'en':
                phone_ids = np.random.randint(1, 78, size=(T, ))
            else:
                phone_ids = np.random.randint(1, 266, size=(T, ))
            am_input_feed.update({'text': phone_ids})
            if am_dataset in {"aishell3", "vctk", "mix"}:
                am_input_feed.update({'spk_id': spk_id})
        elif am_name == 'speedyspeech':
            phone_ids = np.random.randint(1, 92, size=(T, ))
            tone_ids = np.random.randint(1, 5, size=(T, ))
            am_input_feed.update({'phones': phone_ids, 'tones': tone_ids})
        am_sess.run(None, input_feed=am_input_feed)

    # voc warmup
    for T in [227, 308, 544]:
        data = np.random.rand(T, 80).astype("float32")
        voc_sess.run(None, input_feed={"logmel": data})
    print("warm up done!")

    N = 0
    T = 0
    merge_sentences = False
    get_tone_ids = False
    if am_name == 'speedyspeech':
        get_tone_ids = True
    am_input_feed = {}
    for utt_id, sentence in sentences:
        with timer() as t:
            frontend_dict = run_frontend(
                frontend=frontend,
                text=sentence,
                merge_sentences=merge_sentences,
                get_tone_ids=get_tone_ids,
                lang=args.lang)
            phone_ids = frontend_dict['phone_ids']
            flags = 0
            for i in range(len(phone_ids)):
                part_phone_ids = phone_ids[i].numpy()
                if am_name == 'fastspeech2':
                    am_input_feed.update({'text': part_phone_ids})
                    if am_dataset in {"aishell3", "vctk", "mix"}:
                        am_input_feed.update({'spk_id': spk_id})
                elif am_name == 'speedyspeech':
                    part_tone_ids = frontend_dict['tone_ids'][i].numpy()
                    am_input_feed.update({
                        'phones': part_phone_ids,
                        'tones': part_tone_ids
                    })
                mel = am_sess.run(output_names=None, input_feed=am_input_feed)
                mel = mel[0]
                wav = voc_sess.run(
                    output_names=None, input_feed={'logmel': mel})
                wav = wav[0]
                if flags == 0:
                    wav_all = wav
                    flags = 1
                else:
                    wav_all = np.concatenate([wav_all, wav])
        wav = wav_all
        N += len(wav)
        T += t.elapse
        speed = len(wav) / t.elapse
        rtf = fs / speed
        sf.write(str(output_dir / (utt_id + ".wav")), wav, samplerate=fs)
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {len(wav)}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )
    print(f"generation speed: {N / T}Hz, RTF: {fs / (N / T) }")


def parse_args():
    parser = argparse.ArgumentParser(description="Infernce with onnxruntime.")
    # acoustic model
    parser.add_argument(
        '--am',
        type=str,
        default='fastspeech2_csmsc',
        choices=[
            'fastspeech2_csmsc',
            'fastspeech2_aishell3',
            'fastspeech2_ljspeech',
            'fastspeech2_vctk',
            'speedyspeech_csmsc',
            'fastspeech2_mix',
        ],
        help='Choose acoustic model type of tts task.')
    parser.add_argument(
        "--phones_dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--tones_dict", type=str, default=None, help="tone vocabulary file.")
    parser.add_argument(
        '--spk_id',
        type=int,
        default=0,
        help='spk id for multi speaker acoustic model')

    # voc
    parser.add_argument(
        '--voc',
        type=str,
        default='hifigan_csmsc',
        choices=[
            'pwgan_csmsc',
            'pwgan_aishell3',
            'pwgan_ljspeech',
            'pwgan_vctk',
            'hifigan_csmsc',
            'hifigan_aishell3',
            'hifigan_ljspeech',
            'hifigan_vctk',
            'mb_melgan_csmsc',
        ],
        help='Choose vocoder type of tts task.')
    # other
    parser.add_argument(
        "--inference_dir", type=str, help="dir to save inference models")
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line")
    parser.add_argument("--output_dir", type=str, help="output dir")
    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        help='Choose model language. zh or en')

    # inference
    parser.add_argument(
        "--use_trt",
        type=str2bool,
        default=False,
        help="Whether to use inference engin TensorRT.", )

    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device selected for inference.", )
    parser.add_argument('--cpu_threads', type=int, default=1)

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    paddle.set_device(args.device)

    ort_predict(args)


if __name__ == "__main__":
    main()
