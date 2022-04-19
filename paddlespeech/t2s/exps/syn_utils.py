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
import math
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import paddle
from paddle import inference
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


def denorm(data, mean, std):
    return data * std + mean


def get_chunks(data, chunk_size, pad_size):
    data_len = data.shape[1]
    chunks = []
    n = math.ceil(data_len / chunk_size)
    for i in range(n):
        start = max(0, i * chunk_size - pad_size)
        end = min((i + 1) * chunk_size + pad_size, data_len)
        chunks.append(data[:, start:end, :])
    return chunks


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


# inference
def get_predictor(args, filed='am'):
    full_name = ''
    if filed == 'am':
        full_name = args.am
    elif filed == 'voc':
        full_name = args.voc
    config = inference.Config(
        str(Path(args.inference_dir) / (full_name + ".pdmodel")),
        str(Path(args.inference_dir) / (full_name + ".pdiparams")))
    if args.device == "gpu":
        config.enable_use_gpu(100, 0)
    elif args.device == "cpu":
        config.disable_gpu()
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
        spk_id = np.array([args.spk_id])
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


def get_voc_output(voc_predictor, input):
    voc_input_names = voc_predictor.get_input_names()
    mel_handle = voc_predictor.get_input_handle(voc_input_names[0])
    mel_handle.reshape(input.shape)
    mel_handle.copy_from_cpu(input)

    voc_predictor.run()
    voc_output_names = voc_predictor.get_output_names()
    voc_output_handle = voc_predictor.get_output_handle(voc_output_names[0])
    wav = voc_output_handle.copy_to_cpu()
    return wav


# streaming am
def get_streaming_am_predictor(args):
    full_name = args.am
    am_encoder_infer_config = inference.Config(
        str(
            Path(args.inference_dir) /
            (full_name + "_am_encoder_infer" + ".pdmodel")),
        str(
            Path(args.inference_dir) /
            (full_name + "_am_encoder_infer" + ".pdiparams")))
    am_decoder_config = inference.Config(
        str(
            Path(args.inference_dir) /
            (full_name + "_am_decoder" + ".pdmodel")),
        str(
            Path(args.inference_dir) /
            (full_name + "_am_decoder" + ".pdiparams")))
    am_postnet_config = inference.Config(
        str(
            Path(args.inference_dir) /
            (full_name + "_am_postnet" + ".pdmodel")),
        str(
            Path(args.inference_dir) /
            (full_name + "_am_postnet" + ".pdiparams")))
    if args.device == "gpu":
        am_encoder_infer_config.enable_use_gpu(100, 0)
        am_decoder_config.enable_use_gpu(100, 0)
        am_postnet_config.enable_use_gpu(100, 0)
    elif args.device == "cpu":
        am_encoder_infer_config.disable_gpu()
        am_decoder_config.disable_gpu()
        am_postnet_config.disable_gpu()

    am_encoder_infer_config.enable_memory_optim()
    am_decoder_config.enable_memory_optim()
    am_postnet_config.enable_memory_optim()

    am_encoder_infer_predictor = inference.create_predictor(
        am_encoder_infer_config)
    am_decoder_predictor = inference.create_predictor(am_decoder_config)
    am_postnet_predictor = inference.create_predictor(am_postnet_config)
    return am_encoder_infer_predictor, am_decoder_predictor, am_postnet_predictor


def get_am_sublayer_output(am_sublayer_predictor, input):
    am_sublayer_input_names = am_sublayer_predictor.get_input_names()
    input_handle = am_sublayer_predictor.get_input_handle(
        am_sublayer_input_names[0])
    input_handle.reshape(input.shape)
    input_handle.copy_from_cpu(input)

    am_sublayer_predictor.run()
    am_sublayer_names = am_sublayer_predictor.get_output_names()
    am_sublayer_handle = am_sublayer_predictor.get_output_handle(
        am_sublayer_names[0])
    am_sublayer_output = am_sublayer_handle.copy_to_cpu()
    return am_sublayer_output


def get_streaming_am_output(args, am_encoder_infer_predictor,
                            am_decoder_predictor, am_postnet_predictor,
                            frontend, merge_sentences, input):
    get_tone_ids = False
    if args.lang == 'zh':
        input_ids = frontend.get_input_ids(
            input, merge_sentences=merge_sentences, get_tone_ids=get_tone_ids)
        phone_ids = input_ids["phone_ids"]
    else:
        print("lang should be 'zh' here!")

    phones = phone_ids[0].numpy()
    am_encoder_infer_output = get_am_sublayer_output(
        am_encoder_infer_predictor, input=phones)

    am_decoder_output = get_am_sublayer_output(
        am_decoder_predictor, input=am_encoder_infer_output)

    am_postnet_output = get_am_sublayer_output(
        am_postnet_predictor, input=np.transpose(am_decoder_output, (0, 2, 1)))
    am_output_data = am_decoder_output + np.transpose(am_postnet_output,
                                                      (0, 2, 1))
    normalized_mel = am_output_data[0]
    return normalized_mel


def get_sess(args, filed='am'):
    full_name = ''
    if filed == 'am':
        full_name = args.am
    elif filed == 'voc':
        full_name = args.voc
    model_dir = str(Path(args.inference_dir) / (full_name + ".onnx"))
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if args.device == "gpu":
        # fastspeech2/mb_melgan can't use trt now!
        if args.use_trt:
            providers = ['TensorrtExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider']
    elif args.device == "cpu":
        providers = ['CPUExecutionProvider']
    sess_options.intra_op_num_threads = args.cpu_threads
    sess = ort.InferenceSession(
        model_dir, providers=providers, sess_options=sess_options)
    return sess


# streaming am
def get_streaming_am_sess(args):
    full_name = args.am
    am_encoder_infer_model_dir = str(
        Path(args.inference_dir) / (full_name + "_am_encoder_infer" + ".onnx"))
    am_decoder_model_dir = str(
        Path(args.inference_dir) / (full_name + "_am_decoder" + ".onnx"))
    am_postnet_model_dir = str(
        Path(args.inference_dir) / (full_name + "_am_postnet" + ".onnx"))
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if args.device == "gpu":
        # fastspeech2/mb_melgan can't use trt now!
        if args.use_trt:
            providers = ['TensorrtExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider']
    elif args.device == "cpu":
        providers = ['CPUExecutionProvider']
    sess_options.intra_op_num_threads = args.cpu_threads
    am_encoder_infer_sess = ort.InferenceSession(
        am_encoder_infer_model_dir,
        providers=providers,
        sess_options=sess_options)
    am_decoder_sess = ort.InferenceSession(
        am_decoder_model_dir, providers=providers, sess_options=sess_options)
    am_postnet_sess = ort.InferenceSession(
        am_postnet_model_dir, providers=providers, sess_options=sess_options)
    return am_encoder_infer_sess, am_decoder_sess, am_postnet_sess
