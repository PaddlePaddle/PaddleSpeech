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
import logging
from pathlib import Path

import jsonlines
import numpy as np
import paddle
import soundfile as sf
import yaml
from timer import timer
from yacs.config import CfgNode

from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_test_dataset
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.utils import str2bool


def evaluate(args):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for evaluation
    with jsonlines.open(args.test_metadata, 'r') as reader:
        test_metadata = list(reader)

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
    test_dataset = get_test_dataset(
        test_metadata=test_metadata,
        am=args.am,
        speaker_dict=args.speaker_dict,
        voice_cloning=args.voice_cloning)

    # vocoder
    voc_inference = get_voc_inference(
        voc=args.voc,
        voc_config=voc_config,
        voc_ckpt=args.voc_ckpt,
        voc_stat=args.voc_stat)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = 0
    T = 0

    for datum in test_dataset:
        utt_id = datum["utt_id"]
        with timer() as t:
            with paddle.no_grad():
                # acoustic model
                if am_name == 'fastspeech2':
                    phone_ids = paddle.to_tensor(datum["text"])
                    spk_emb = None
                    spk_id = None
                    # multi speaker
                    if args.voice_cloning and "spk_emb" in datum:
                        spk_emb = paddle.to_tensor(np.load(datum["spk_emb"]))
                    elif "spk_id" in datum:
                        spk_id = paddle.to_tensor(datum["spk_id"])
                    mel = am_inference(
                        phone_ids, spk_id=spk_id, spk_emb=spk_emb)
                elif am_name == 'speedyspeech':
                    phone_ids = paddle.to_tensor(datum["phones"])
                    tone_ids = paddle.to_tensor(datum["tones"])
                    mel = am_inference(phone_ids, tone_ids)
                elif am_name == 'tacotron2':
                    phone_ids = paddle.to_tensor(datum["text"])
                    spk_emb = None
                    # multi speaker
                    if args.voice_cloning and "spk_emb" in datum:
                        spk_emb = paddle.to_tensor(np.load(datum["spk_emb"]))
                    mel = am_inference(phone_ids, spk_emb=spk_emb)
                elif am_name == 'diffsinger':
                    phone_ids = paddle.to_tensor(datum["text"])
                    note = paddle.to_tensor(datum["note"])
                    note_dur = paddle.to_tensor(datum["note_dur"])
                    is_slur = paddle.to_tensor(datum["is_slur"])
                    # get_mel_fs2 = False, means mel from diffusion, get_mel_fs2 = True, means mel from fastspeech2.
                    get_mel_fs2 = False
                    # mel: [T, mel_bin]
                    mel = am_inference(
                        phone_ids,
                        note=note,
                        note_dur=note_dur,
                        is_slur=is_slur,
                        get_mel_fs2=get_mel_fs2)
                # vocoder
                wav = voc_inference(mel)

            wav = wav.numpy()
            N += wav.size
            T += t.elapse
            speed = wav.size / t.elapse
            rtf = am_config.fs / speed
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {wav.size}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
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
            'fastspeech2_csmsc',
            'fastspeech2_ljspeech',
            'fastspeech2_aishell3',
            'fastspeech2_vctk',
            'tacotron2_csmsc',
            'tacotron2_ljspeech',
            'tacotron2_aishell3',
            'fastspeech2_mix',
            'fastspeech2_canton',
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
        "--voice-cloning",
        type=str2bool,
        default=False,
        help="whether training voice cloning model.")
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
            'wavernn_csmsc',
            'hifigan_csmsc',
            'hifigan_ljspeech',
            'hifigan_aishell3',
            'hifigan_vctk',
            'style_melgan_csmsc',
            "pwgan_opencpop",
            "hifigan_opencpop",
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
    parser.add_argument("--test_metadata", type=str, help="test metadata.")
    parser.add_argument("--output_dir", type=str, help="output dir.")
    parser.add_argument(
        "--speech_stretchs",
        type=str,
        default=None,
        help="The min and max values of the mel spectrum.")

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
