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
import os

import numpy as np
import paddle
from paddle import jit
from paddle.static import InputSpec

from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.datasets.data_table import DataTable
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
    "tacotron2":
    "paddlespeech.t2s.models.tacotron2:Tacotron2",
    "tacotron2_inference":
    "paddlespeech.t2s.models.tacotron2:Tacotron2Inference",
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
    "wavernn":
    "paddlespeech.t2s.models.wavernn:WaveRNN",
    "wavernn_inference":
    "paddlespeech.t2s.models.wavernn:WaveRNNInference",
}


# input
def get_sentences(args):
    # construct dataset for evaluation
    sentences = []
    with open(args.text, 'rt') as f:
        for line in f:
            items = line.strip().split()
            utt_id = items[0]
            if 'lang' in args and args.lang == 'zh':
                sentence = "".join(items[1:])
            elif 'lang' in args and args.lang == 'en':
                sentence = " ".join(items[1:])
            sentences.append((utt_id, sentence))
    return sentences


def get_test_dataset(args, test_metadata, am_name, am_dataset):
    if am_name == 'fastspeech2':
        fields = ["utt_id", "text"]
        if am_dataset in {"aishell3", "vctk"} and args.speaker_dict:
            print("multiple speaker fastspeech2!")
            fields += ["spk_id"]
        elif 'voice_cloning' in args and args.voice_cloning:
            print("voice cloning!")
            fields += ["spk_emb"]
        else:
            print("single speaker fastspeech2!")
    elif am_name == 'speedyspeech':
        fields = ["utt_id", "phones", "tones"]
    elif am_name == 'tacotron2':
        fields = ["utt_id", "text"]
        if 'voice_cloning' in args and args.voice_cloning:
            print("voice cloning!")
            fields += ["spk_emb"]

    test_dataset = DataTable(data=test_metadata, fields=fields)
    return test_dataset


# frontend
def get_frontend(args):
    if 'lang' in args and args.lang == 'zh':
        frontend = Frontend(
            phone_vocab_path=args.phones_dict, tone_vocab_path=args.tones_dict)
    elif 'lang' in args and args.lang == 'en':
        frontend = English(phone_vocab_path=args.phones_dict)
    else:
        print("wrong lang!")
    print("frontend done!")
    return frontend


# dygraph
def get_am_inference(args, am_config):
    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    tone_size = None
    if 'tones_dict' in args and args.tones_dict:
        with open(args.tones_dict, "r") as f:
            tone_id = [line.strip().split() for line in f.readlines()]
        tone_size = len(tone_id)
        print("tone_size:", tone_size)

    spk_num = None
    if 'speaker_dict' in args and args.speaker_dict:
        with open(args.speaker_dict, 'rt') as f:
            spk_id = [line.strip().split() for line in f.readlines()]
        spk_num = len(spk_id)
        print("spk_num:", spk_num)

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
            vocab_size=vocab_size,
            tone_size=tone_size,
            spk_num=spk_num,
            **am_config["model"])
    elif am_name == 'tacotron2':
        am = am_class(idim=vocab_size, odim=odim, **am_config["model"])

    am.set_state_dict(paddle.load(args.am_ckpt)["main_params"])
    am.eval()
    am_mu, am_std = np.load(args.am_stat)
    am_mu = paddle.to_tensor(am_mu)
    am_std = paddle.to_tensor(am_std)
    am_normalizer = ZScore(am_mu, am_std)
    am_inference = am_inference_class(am_normalizer, am)
    am_inference.eval()
    print("acoustic model done!")
    return am_inference, am_name, am_dataset


def get_voc_inference(args, voc_config):
    # model: {model_name}_{dataset}
    voc_name = args.voc[:args.voc.rindex('_')]
    voc_class = dynamic_import(voc_name, model_alias)
    voc_inference_class = dynamic_import(voc_name + '_inference', model_alias)
    if voc_name != 'wavernn':
        voc = voc_class(**voc_config["generator_params"])
        voc.set_state_dict(paddle.load(args.voc_ckpt)["generator_params"])
        voc.remove_weight_norm()
        voc.eval()
    else:
        voc = voc_class(**voc_config["model"])
        voc.set_state_dict(paddle.load(args.voc_ckpt)["main_params"])
        voc.eval()

    voc_mu, voc_std = np.load(args.voc_stat)
    voc_mu = paddle.to_tensor(voc_mu)
    voc_std = paddle.to_tensor(voc_std)
    voc_normalizer = ZScore(voc_mu, voc_std)
    voc_inference = voc_inference_class(voc_normalizer, voc)
    voc_inference.eval()
    print("voc done!")
    return voc_inference


# to static
def am_to_static(args, am_inference, am_name, am_dataset):
    if am_name == 'fastspeech2':
        if am_dataset in {"aishell3", "vctk"} and args.speaker_dict:
            am_inference = jit.to_static(
                am_inference,
                input_spec=[
                    InputSpec([-1], dtype=paddle.int64),
                    InputSpec([1], dtype=paddle.int64),
                ], )
        else:
            am_inference = jit.to_static(
                am_inference, input_spec=[InputSpec([-1], dtype=paddle.int64)])

    elif am_name == 'speedyspeech':
        if am_dataset in {"aishell3", "vctk"} and args.speaker_dict:
            am_inference = jit.to_static(
                am_inference,
                input_spec=[
                    InputSpec([-1], dtype=paddle.int64),  # text
                    InputSpec([-1], dtype=paddle.int64),  # tone
                    InputSpec([1], dtype=paddle.int64),  # spk_id
                    None  # duration
                ])
        else:
            am_inference = jit.to_static(
                am_inference,
                input_spec=[
                    InputSpec([-1], dtype=paddle.int64),
                    InputSpec([-1], dtype=paddle.int64)
                ])

    elif am_name == 'tacotron2':
        am_inference = jit.to_static(
            am_inference, input_spec=[InputSpec([-1], dtype=paddle.int64)])

    paddle.jit.save(am_inference, os.path.join(args.inference_dir, args.am))
    am_inference = paddle.jit.load(os.path.join(args.inference_dir, args.am))
    return am_inference


def voc_to_static(args, voc_inference):
    voc_inference = jit.to_static(
        voc_inference, input_spec=[
            InputSpec([-1, 80], dtype=paddle.float32),
        ])
    paddle.jit.save(voc_inference, os.path.join(args.inference_dir, args.voc))
    voc_inference = paddle.jit.load(os.path.join(args.inference_dir, args.voc))
    return voc_inference
