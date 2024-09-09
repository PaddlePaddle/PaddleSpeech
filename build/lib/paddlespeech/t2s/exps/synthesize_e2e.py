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
from pprint import pprint

import paddle
import soundfile as sf
import yaml
from timer import timer
from yacs.config import CfgNode

from paddlespeech.t2s.exps.syn_utils import am_to_static
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.exps.syn_utils import get_sentences_svs
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.exps.syn_utils import run_frontend
from paddlespeech.t2s.exps.syn_utils import voc_to_static
from paddlespeech.t2s.utils import str2bool


def evaluate(args):

    # Init body.
    with open(args.am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))
    with open(args.voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(am_config)
    print(voc_config)

    # frontend
    frontend = get_frontend(
        lang=args.lang,
        phones_dict=args.phones_dict,
        tones_dict=args.tones_dict,
        pinyin_phone=args.pinyin_phone,
        use_rhy=args.use_rhy)
    print("frontend done!")

    # acoustic model
    am_name = args.am[:args.am.rindex('_')]
    am_dataset = args.am[args.am.rindex('_') + 1:]
    am_inference = get_am_inference(
        am=args.am,
        am_config=am_config,
        am_ckpt=args.am_ckpt,
        am_stat=args.am_stat,
        phones_dict=args.phones_dict,
        tones_dict=args.tones_dict,
        speaker_dict=args.speaker_dict,
        speech_stretchs=args.speech_stretchs, )
    print("acoustic model done!")

    # vocoder
    voc_inference = get_voc_inference(
        voc=args.voc,
        voc_config=voc_config,
        voc_ckpt=args.voc_ckpt,
        voc_stat=args.voc_stat)
    print("voc done!")

    # whether dygraph to static
    if args.inference_dir:
        print("convert am and voc to static model.")
        # acoustic model
        am_inference = am_to_static(
            am_inference=am_inference,
            am=args.am,
            inference_dir=args.inference_dir,
            speaker_dict=args.speaker_dict)
        # vocoder
        voc_inference = voc_to_static(
            voc_inference=voc_inference,
            voc=args.voc,
            inference_dir=args.inference_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_sentences = False
    # Avoid not stopping at the end of a sub sentence when tacotron2_ljspeech dygraph to static graph
    # but still not stopping in the end (NOTE by yuantian01 Feb 9 2022)
    if am_name == 'tacotron2':
        merge_sentences = True

    get_tone_ids = False
    if am_name == 'speedyspeech':
        get_tone_ids = True

    # wav samples
    N = 0
    # inference time cost
    T = 0

    # [(uid, text), ]
    if am_name == 'diffsinger':
        sentences = get_sentences_svs(text_file=args.text)
    else:
        sentences = get_sentences(text_file=args.text, lang=args.lang)

    for utt_id, sentence in sentences:
        print(f"{utt_id} {sentence}")
        with timer() as t:
            if am_name == "diffsinger":
                text = ""
                svs_input = sentence
            else:
                text = sentence
                svs_input = None

            # frontend
            frontend_dict = run_frontend(
                frontend=frontend,
                text=text,
                merge_sentences=merge_sentences,
                get_tone_ids=get_tone_ids,
                lang=args.lang,
                svs_input=svs_input)
            phone_ids = frontend_dict['phone_ids']
            # pprint(f"{utt_id} {phone_ids}")

            with paddle.no_grad():
                flags = 0
                for i in range(len(phone_ids)):
                    # sub phone, split by `sp` or punctuation.
                    part_phone_ids = phone_ids[i]

                    # acoustic model
                    if am_name == 'fastspeech2':
                        # multi speaker
                        if am_dataset in {"aishell3", "vctk", "mix", "canton"}:
                            # multi-speaker
                            spk_id = paddle.to_tensor([args.spk_id])
                            mel = am_inference(part_phone_ids, spk_id)
                        else:
                            # single-speaker
                            mel = am_inference(part_phone_ids)
                    elif am_name == 'speedyspeech':
                        part_tone_ids = frontend_dict['tone_ids'][i]
                        if am_dataset in {"aishell3", "vctk", "mix"}:
                            # multi-speaker
                            spk_id = paddle.to_tensor([args.spk_id])
                            mel = am_inference(part_phone_ids, part_tone_ids,
                                               spk_id)
                        else:
                            # single-speaker
                            mel = am_inference(part_phone_ids, part_tone_ids)
                    elif am_name == 'tacotron2':
                        mel = am_inference(part_phone_ids)
                    elif am_name == 'diffsinger':
                        part_note_ids = frontend_dict['note_ids'][i]
                        part_note_durs = frontend_dict['note_durs'][i]
                        part_is_slurs = frontend_dict['is_slurs'][i]
                        mel = am_inference(
                            text=part_phone_ids,
                            note=part_note_ids,
                            note_dur=part_note_durs,
                            is_slur=part_is_slurs, )

                    # vocoder
                    wav = voc_inference(mel)
                    if flags == 0:
                        wav_all = wav
                        flags = 1
                    else:
                        wav_all = paddle.concat([wav_all, wav])

        wav = wav_all.numpy()
        N += wav.size
        T += t.elapse

        # samples per second
        speed = wav.size / t.elapse
        # generate one second wav need `RTF` seconds
        rtf = am_config.fs / speed
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )

        sf.write(
            str(output_dir / (utt_id + ".wav")), wav, samplerate=am_config.fs)
        print(f"{utt_id} done!")

    print(f"generation speed: {N / T}Hz, RTF: {am_config.fs / (N / T) }")


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="Synthesize with acoustic model & vocoder")
    # acoustic model
    parser.add_argument(
        '--am',
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
            'diffsinger_opencpop',
        ],
        help='Choose acoustic model type of tts task.')
    parser.add_argument(
        '--am_config', type=str, default=None, help='Config of acoustic model.')
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
    parser.add_argument(
        '--spk_id',
        type=int,
        default=0,
        help='spk id for multi speaker acoustic model')
    # vocoder
    parser.add_argument(
        '--voc',
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
    parser.add_argument(
        '--voc_config', type=str, default=None, help='Config of voc.')
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
        '--lang',
        type=str,
        default='zh',
        choices=['zh', 'en', 'mix', 'canton', 'sing'],
        help='Choose model language. zh or en or mix')

    parser.add_argument(
        "--inference_dir",
        type=str,
        default=None,
        help="dir to save inference models")
    parser.add_argument(
        "--ngpu",
        type=int,
        default=1,
        help="if wish to use gpu, set ngpu > 0, otherwise use xpu, npu, mlu or cpu."
    )
    parser.add_argument(
        "--nxpu",
        type=int,
        default=0,
        help="if wish to use xpu, set ngpu == 0 and nxpu > 0, otherwise use gpu, npu, mlu or cpu."
    )
    parser.add_argument(
        "--nnpu",
        type=int,
        default=0,
        help="if wish to use npu, set ngpu == 0 and nnpu > 0, otherwise use gpu, xpu, mlu or cpu."
    )
    parser.add_argument(
        "--nmlu",
        type=int,
        default=0,
        help="if wish to use xpu, set ngpu == 0 and nmlu > 0, otherwise use gpu, xpu, npu or cpu."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output_dir", type=str, help="output dir.")
    parser.add_argument(
        "--use_rhy",
        type=str2bool,
        default=False,
        help="run rhythm frontend or not")
    parser.add_argument(
        "--pinyin_phone",
        type=str,
        default=None,
        help="pinyin to phone map file, using on sing_frontend.")
    parser.add_argument(
        "--speech_stretchs",
        type=str,
        default=None,
        help="The min and max values of the mel spectrum, using on diffusion of diffsinger."
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.ngpu > 0:
        paddle.set_device("gpu")
    elif args.nxpu > 0:
        paddle.set_device("xpu")
    elif args.nnpu > 0:
        paddle.set_device("npu")
    elif args.nmlu > 0:
        paddle.set_device("mlu")
    elif args.ngpu == 0 and args.nxpu == 0 and args.nnpu == 0 and args.nmlu == 0:
        paddle.set_device("cpu")
    else:
        print(
            "one of ngpu, nxpu, nnpu or nmlu should be greater than 0 or all of them equal to 0"
        )

    evaluate(args)


if __name__ == "__main__":
    main()
