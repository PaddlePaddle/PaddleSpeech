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
import os
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf
import yaml
from paddle import jit
from paddle.static import InputSpec
from yacs.config import CfgNode

from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.modules.normalizer import ZScore

model_alias = {
    # acoustic model
    "speedyspeech":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeech",
    "speedyspeech_inference":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeechInference",
    "fastspeech2":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2",
    "fastspeech2_inference":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2Inference",
    # voc
    "pwgan":
    "paddlespeech.t2s.models.parallel_wavegan:PWGGenerator",
    "pwgan_inference":
    "paddlespeech.t2s.models.parallel_wavegan:PWGInference",
    "mb_melgan":
    "paddlespeech.t2s.models.melgan:MelGANGenerator",
    "mb_melgan_inference":
    "paddlespeech.t2s.models.melgan:MelGANInference",
    "style_melgan":
    "paddlespeech.t2s.models.melgan:StyleMelGANGenerator",
    "style_melgan_inference":
    "paddlespeech.t2s.models.melgan:StyleMelGANInference",
    "hifigan":
    "paddlespeech.t2s.models.hifigan:HiFiGANGenerator",
    "hifigan_inference":
    "paddlespeech.t2s.models.hifigan:HiFiGANInference",
}


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

    # construct dataset for evaluation
    sentences = []
    with open(args.text, 'rt') as f:
        for line in f:
            items = line.strip().split()
            utt_id = items[0]
            if args.lang == 'zh':
                sentence = "".join(items[1:])
            elif args.lang == 'en':
                sentence = " ".join(items[1:])
            sentences.append((utt_id, sentence))

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    tone_size = None
    if args.tones_dict:
        with open(args.tones_dict, "r") as f:
            tone_id = [line.strip().split() for line in f.readlines()]
        tone_size = len(tone_id)
        print("tone_size:", tone_size)

    spk_num = None
    if args.speaker_dict:
        with open(args.speaker_dict, 'rt') as f:
            spk_id = [line.strip().split() for line in f.readlines()]
        spk_num = len(spk_id)
        print("spk_num:", spk_num)

    # frontend
    if args.lang == 'zh':
        frontend = Frontend(
            phone_vocab_path=args.phones_dict, tone_vocab_path=args.tones_dict)
    elif args.lang == 'en':
        frontend = English(phone_vocab_path=args.phones_dict)
    print("frontend done!")

    # acoustic model
    odim = am_config.n_mels
    # model: {model_name}_{dataset}
    am_name = args.am[:args.am.rindex('_')]
    am_dataset = args.am[args.am.rindex('_') + 1:]

    am_class = dynamic_import(am_name, model_alias)
    am_inference_class = dynamic_import(am_name + '_inference', model_alias)

    if am_name == 'fastspeech2':
        am = am_class(
            idim=vocab_size, odim=odim, spk_num=spk_num, **am_config["model"])
    elif am_name == 'speedyspeech':
        am = am_class(
            vocab_size=vocab_size, tone_size=tone_size, **am_config["model"])

    am.set_state_dict(paddle.load(args.am_ckpt)["main_params"])
    am.eval()
    am_mu, am_std = np.load(args.am_stat)
    am_mu = paddle.to_tensor(am_mu)
    am_std = paddle.to_tensor(am_std)
    am_normalizer = ZScore(am_mu, am_std)
    am_inference = am_inference_class(am_normalizer, am)
    am_inference.eval()
    print("acoustic model done!")

    # vocoder
    # model: {model_name}_{dataset}
    voc_name = args.voc[:args.voc.rindex('_')]
    voc_class = dynamic_import(voc_name, model_alias)
    voc_inference_class = dynamic_import(voc_name + '_inference', model_alias)
    voc = voc_class(**voc_config["generator_params"])
    voc.set_state_dict(paddle.load(args.voc_ckpt)["generator_params"])
    voc.remove_weight_norm()
    voc.eval()
    voc_mu, voc_std = np.load(args.voc_stat)
    voc_mu = paddle.to_tensor(voc_mu)
    voc_std = paddle.to_tensor(voc_std)
    voc_normalizer = ZScore(voc_mu, voc_std)
    voc_inference = voc_inference_class(voc_normalizer, voc)
    voc_inference.eval()
    print("voc done!")

    # whether dygraph to static
    if args.inference_dir:
        # acoustic model
        if am_name == 'fastspeech2':
            if am_dataset in {"aishell3", "vctk"} and args.speaker_dict:
                print(
                    "Haven't test dygraph to static for multi speaker fastspeech2 now!"
                )
            else:
                am_inference = jit.to_static(
                    am_inference,
                    input_spec=[InputSpec([-1], dtype=paddle.int64)])
                paddle.jit.save(am_inference,
                                os.path.join(args.inference_dir, args.am))
                am_inference = paddle.jit.load(
                    os.path.join(args.inference_dir, args.am))
        elif am_name == 'speedyspeech':
            am_inference = jit.to_static(
                am_inference,
                input_spec=[
                    InputSpec([-1], dtype=paddle.int64),
                    InputSpec([-1], dtype=paddle.int64)
                ])

            paddle.jit.save(am_inference,
                            os.path.join(args.inference_dir, args.am))
            am_inference = paddle.jit.load(
                os.path.join(args.inference_dir, args.am))

        # vocoder
        voc_inference = jit.to_static(
            voc_inference,
            input_spec=[
                InputSpec([-1, 80], dtype=paddle.float32),
            ])
        paddle.jit.save(voc_inference,
                        os.path.join(args.inference_dir, args.voc))
        voc_inference = paddle.jit.load(
            os.path.join(args.inference_dir, args.voc))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merge_sentences = False
    for utt_id, sentence in sentences:
        get_tone_ids = False
        if am_name == 'speedyspeech':
            get_tone_ids = True
        if args.lang == 'zh':
            input_ids = frontend.get_input_ids(
                sentence,
                merge_sentences=merge_sentences,
                get_tone_ids=get_tone_ids)
            phone_ids = input_ids["phone_ids"]
            if get_tone_ids:
                tone_ids = input_ids["tone_ids"]
        elif args.lang == 'en':
            input_ids = frontend.get_input_ids(
                sentence, merge_sentences=merge_sentences)
            phone_ids = input_ids["phone_ids"]
        else:
            print("lang should in {'zh', 'en'}!")
        with paddle.no_grad():
            flags = 0
            for i in range(len(phone_ids)):
                part_phone_ids = phone_ids[i]
                # acoustic model
                if am_name == 'fastspeech2':
                    # multi speaker
                    if am_dataset in {"aishell3", "vctk"}:
                        spk_id = paddle.to_tensor(args.spk_id)
                        mel = am_inference(part_phone_ids, spk_id)
                    else:
                        mel = am_inference(part_phone_ids)
                elif am_name == 'speedyspeech':
                    part_tone_ids = tone_ids[i]
                    mel = am_inference(part_phone_ids, part_tone_ids)
                # vocoder
                wav = voc_inference(mel)
                if flags == 0:
                    wav_all = wav
                    flags = 1
                else:
                    wav_all = paddle.concat([wav_all, wav])
        sf.write(
            str(output_dir / (utt_id + ".wav")),
            wav_all.numpy(),
            samplerate=am_config.fs)
        print(f"{utt_id} done!")


def main():
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
            'fastspeech2_aishell3', 'fastspeech2_vctk'
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
            'pwgan_csmsc', 'pwgan_ljspeech', 'pwgan_aishell3', 'pwgan_vctk',
            'mb_melgan_csmsc', 'style_melgan_csmsc', 'hifigan_csmsc'
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
        '--lang',
        type=str,
        default='zh',
        help='Choose model language. zh or en')

    parser.add_argument(
        "--inference_dir",
        type=str,
        default=None,
        help="dir to save inference models")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output_dir", type=str, help="output dir.")

    args = parser.parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    evaluate(args)


if __name__ == "__main__":
    main()
