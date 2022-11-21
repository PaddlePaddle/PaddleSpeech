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
import re
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import onnxruntime as ort
import paddle
from paddle import inference
from paddle import jit
from paddle.static import InputSpec
from paddlelite.lite import create_paddle_predictor
from paddlelite.lite import MobileConfig
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.mix_frontend import MixFrontend
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.modules.normalizer import ZScore
from paddlespeech.utils.dynamic_import import dynamic_import

# remove [W:onnxruntime: xxx] from ort
ort.set_default_logger_severity(3)

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
    "erniesat":
    "paddlespeech.t2s.models.ernie_sat:ErnieSAT",
    "erniesat_inference":
    "paddlespeech.t2s.models.ernie_sat:ErnieSATInference",
}


def denorm(data, mean, std):
    return data * std + mean


def norm(data, mean, std):
    return (data - mean) / std


def get_chunks(data, block_size: int, pad_size: int):
    data_len = data.shape[1]
    chunks = []
    n = math.ceil(data_len / block_size)
    for i in range(n):
        start = max(0, i * block_size - pad_size)
        end = min((i + 1) * block_size + pad_size, data_len)
        chunks.append(data[:, start:end, :])
    return chunks


# input
def get_sentences(text_file: Optional[os.PathLike], lang: str='zh'):
    # construct dataset for evaluation
    sentences = []
    with open(text_file, 'rt') as f:
        for line in f:
            if line.strip() != "":
                items = re.split(r"\s+", line.strip(), 1)
                utt_id = items[0]
                if lang == 'zh':
                    sentence = "".join(items[1:])
                elif lang == 'en':
                    sentence = " ".join(items[1:])
                elif lang == 'mix':
                    sentence = " ".join(items[1:])
            sentences.append((utt_id, sentence))
    return sentences


def get_test_dataset(test_metadata: List[Dict[str, Any]],
                     am: str,
                     speaker_dict: Optional[os.PathLike]=None,
                     voice_cloning: bool=False):
    # model: {model_name}_{dataset}
    am_name = am[:am.rindex('_')]
    am_dataset = am[am.rindex('_') + 1:]
    converters = {}
    if am_name == 'fastspeech2':
        fields = ["utt_id", "text"]
        if am_dataset in {"aishell3", "vctk",
                          "mix"} and speaker_dict is not None:
            print("multiple speaker fastspeech2!")
            fields += ["spk_id"]
        elif voice_cloning:
            print("voice cloning!")
            fields += ["spk_emb"]
        else:
            print("single speaker fastspeech2!")
    elif am_name == 'speedyspeech':
        fields = ["utt_id", "phones", "tones"]
    elif am_name == 'tacotron2':
        fields = ["utt_id", "text"]
        if voice_cloning:
            print("voice cloning!")
            fields += ["spk_emb"]
    elif am_name == 'erniesat':
        fields = [
            "utt_id", "text", "text_lengths", "speech", "speech_lengths",
            "align_start", "align_end"
        ]
        converters = {"speech": np.load}
    else:
        print("wrong am, please input right am!!!")

    test_dataset = DataTable(
        data=test_metadata, fields=fields, converters=converters)
    return test_dataset


# frontend
def get_frontend(lang: str='zh',
                 phones_dict: Optional[os.PathLike]=None,
                 tones_dict: Optional[os.PathLike]=None):
    if lang == 'zh':
        frontend = Frontend(
            phone_vocab_path=phones_dict, tone_vocab_path=tones_dict)
    elif lang == 'en':
        frontend = English(phone_vocab_path=phones_dict)
    elif lang == 'mix':
        frontend = MixFrontend(
            phone_vocab_path=phones_dict, tone_vocab_path=tones_dict)
    else:
        print("wrong lang!")
    return frontend


def run_frontend(frontend: object,
                 text: str,
                 merge_sentences: bool=False,
                 get_tone_ids: bool=False,
                 lang: str='zh',
                 to_tensor: bool=True):
    outs = dict()
    if lang == 'zh':
        input_ids = {}
        if text.strip() != "" and re.match(r".*?<speak>.*?</speak>.*", text,
                                           re.DOTALL):
            input_ids = frontend.get_input_ids_ssml(
                text,
                merge_sentences=merge_sentences,
                get_tone_ids=get_tone_ids,
                to_tensor=to_tensor)
        else:
            input_ids = frontend.get_input_ids(
                text,
                merge_sentences=merge_sentences,
                get_tone_ids=get_tone_ids,
                to_tensor=to_tensor)
        phone_ids = input_ids["phone_ids"]
        if get_tone_ids:
            tone_ids = input_ids["tone_ids"]
            outs.update({'tone_ids': tone_ids})
    elif lang == 'en':
        input_ids = frontend.get_input_ids(
            text, merge_sentences=merge_sentences, to_tensor=to_tensor)
        phone_ids = input_ids["phone_ids"]
    elif lang == 'mix':
        input_ids = frontend.get_input_ids(
            text, merge_sentences=merge_sentences, to_tensor=to_tensor)
        phone_ids = input_ids["phone_ids"]
    else:
        print("lang should in {'zh', 'en', 'mix'}!")
    outs.update({'phone_ids': phone_ids})
    return outs


# dygraph
def get_am_inference(am: str='fastspeech2_csmsc',
                     am_config: CfgNode=None,
                     am_ckpt: Optional[os.PathLike]=None,
                     am_stat: Optional[os.PathLike]=None,
                     phones_dict: Optional[os.PathLike]=None,
                     tones_dict: Optional[os.PathLike]=None,
                     speaker_dict: Optional[os.PathLike]=None,
                     return_am: bool=False):
    with open(phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    tone_size = None
    if tones_dict is not None:
        with open(tones_dict, "r") as f:
            tone_id = [line.strip().split() for line in f.readlines()]
        tone_size = len(tone_id)
    spk_num = None
    if speaker_dict is not None:
        with open(speaker_dict, 'rt') as f:
            spk_id = [line.strip().split() for line in f.readlines()]
        spk_num = len(spk_id)
    odim = am_config.n_mels
    # model: {model_name}_{dataset}
    am_name = am[:am.rindex('_')]
    am_dataset = am[am.rindex('_') + 1:]
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
    elif am_name == 'erniesat':
        am = am_class(idim=vocab_size, odim=odim, **am_config["model"])
    else:
        print("wrong am, please input right am!!!")

    am.set_state_dict(paddle.load(am_ckpt)["main_params"])
    am.eval()
    am_mu, am_std = np.load(am_stat)
    am_mu = paddle.to_tensor(am_mu)
    am_std = paddle.to_tensor(am_std)
    am_normalizer = ZScore(am_mu, am_std)
    am_inference = am_inference_class(am_normalizer, am)
    am_inference.eval()
    if return_am:
        return am_inference, am
    else:
        return am_inference


def get_voc_inference(
        voc: str='pwgan_csmsc',
        voc_config: Optional[os.PathLike]=None,
        voc_ckpt: Optional[os.PathLike]=None,
        voc_stat: Optional[os.PathLike]=None, ):
    # model: {model_name}_{dataset}
    voc_name = voc[:voc.rindex('_')]
    voc_class = dynamic_import(voc_name, model_alias)
    voc_inference_class = dynamic_import(voc_name + '_inference', model_alias)
    if voc_name != 'wavernn':
        voc = voc_class(**voc_config["generator_params"])
        voc.set_state_dict(paddle.load(voc_ckpt)["generator_params"])
        voc.remove_weight_norm()
        voc.eval()
    else:
        voc = voc_class(**voc_config["model"])
        voc.set_state_dict(paddle.load(voc_ckpt)["main_params"])
        voc.eval()

    voc_mu, voc_std = np.load(voc_stat)
    voc_mu = paddle.to_tensor(voc_mu)
    voc_std = paddle.to_tensor(voc_std)
    voc_normalizer = ZScore(voc_mu, voc_std)
    voc_inference = voc_inference_class(voc_normalizer, voc)
    voc_inference.eval()
    return voc_inference


# dygraph to static graph
def am_to_static(am_inference,
                 am: str='fastspeech2_csmsc',
                 inference_dir=Optional[os.PathLike],
                 speaker_dict: Optional[os.PathLike]=None):
    # model: {model_name}_{dataset}
    am_name = am[:am.rindex('_')]
    am_dataset = am[am.rindex('_') + 1:]
    if am_name == 'fastspeech2':
        if am_dataset in {"aishell3", "vctk",
                          "mix"} and speaker_dict is not None:
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
        if am_dataset in {"aishell3", "vctk",
                          "mix"} and speaker_dict is not None:
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

    paddle.jit.save(am_inference, os.path.join(inference_dir, am))
    am_inference = paddle.jit.load(os.path.join(inference_dir, am))
    return am_inference


def voc_to_static(voc_inference,
                  voc: str='pwgan_csmsc',
                  inference_dir=Optional[os.PathLike]):
    voc_inference = jit.to_static(
        voc_inference, input_spec=[
            InputSpec([-1, 80], dtype=paddle.float32),
        ])
    paddle.jit.save(voc_inference, os.path.join(inference_dir, voc))
    voc_inference = paddle.jit.load(os.path.join(inference_dir, voc))
    return voc_inference


# inference
def get_predictor(model_dir: Optional[os.PathLike]=None,
                  model_file: Optional[os.PathLike]=None,
                  params_file: Optional[os.PathLike]=None,
                  device: str='cpu'):

    config = inference.Config(
        str(Path(model_dir) / model_file), str(Path(model_dir) / params_file))
    if device == "gpu":
        config.enable_use_gpu(100, 0)
    elif device == "cpu":
        config.disable_gpu()
    config.enable_memory_optim()
    predictor = inference.create_predictor(config)
    return predictor


def get_am_output(
        input: str,
        am_predictor: paddle.nn.Layer,
        am: str,
        frontend: object,
        lang: str='zh',
        merge_sentences: bool=True,
        speaker_dict: Optional[os.PathLike]=None,
        spk_id: int=0, ):
    am_name = am[:am.rindex('_')]
    am_dataset = am[am.rindex('_') + 1:]
    am_input_names = am_predictor.get_input_names()
    get_spk_id = False
    get_tone_ids = False
    if am_name == 'speedyspeech':
        get_tone_ids = True
    if am_dataset in {"aishell3", "vctk", "mix"} and speaker_dict:
        get_spk_id = True
        spk_id = np.array([spk_id])

    frontend_dict = run_frontend(
        frontend=frontend,
        text=input,
        merge_sentences=merge_sentences,
        get_tone_ids=get_tone_ids,
        lang=lang)

    if get_tone_ids:
        tone_ids = frontend_dict['tone_ids']
        tones = tone_ids[0].numpy()
        tones_handle = am_predictor.get_input_handle(am_input_names[1])
        tones_handle.reshape(tones.shape)
        tones_handle.copy_from_cpu(tones)
    if get_spk_id:
        spk_id_handle = am_predictor.get_input_handle(am_input_names[1])
        spk_id_handle.reshape(spk_id.shape)
        spk_id_handle.copy_from_cpu(spk_id)
    phone_ids = frontend_dict['phone_ids']
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


def get_streaming_am_output(input: str,
                            am_encoder_infer_predictor,
                            am_decoder_predictor,
                            am_postnet_predictor,
                            frontend,
                            lang: str='zh',
                            merge_sentences: bool=True):
    get_tone_ids = False
    frontend_dict = run_frontend(
        frontend=frontend,
        text=input,
        merge_sentences=merge_sentences,
        get_tone_ids=get_tone_ids,
        lang=lang)
    phone_ids = frontend_dict['phone_ids']
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


# onnx
def get_sess(model_path: Optional[os.PathLike],
             device: str='cpu',
             cpu_threads: int=1,
             use_trt: bool=False):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if 'gpu' in device.lower():
        device_id = int(device.split(':')[1]) if len(
            device.split(':')) == 2 else 0
        # fastspeech2/mb_melgan can't use trt now!
        if use_trt:
            provider_name = 'TensorrtExecutionProvider'
        else:
            provider_name = 'CUDAExecutionProvider'
        providers = [(provider_name, {'device_id': device_id})]
    elif device.lower() == 'cpu':
        providers = ['CPUExecutionProvider']
    sess_options.intra_op_num_threads = cpu_threads
    sess = ort.InferenceSession(
        model_path, providers=providers, sess_options=sess_options)
    return sess


# Paddle-Lite
def get_lite_predictor(model_dir: Optional[os.PathLike]=None,
                       model_file: Optional[os.PathLike]=None,
                       cpu_threads: int=1):
    config = MobileConfig()
    config.set_model_from_file(str(Path(model_dir) / model_file))
    predictor = create_paddle_predictor(config)
    return predictor


def get_lite_am_output(
        input: str,
        am_predictor,
        am: str,
        frontend: object,
        lang: str='zh',
        merge_sentences: bool=True,
        speaker_dict: Optional[os.PathLike]=None,
        spk_id: int=0, ):
    am_name = am[:am.rindex('_')]
    am_dataset = am[am.rindex('_') + 1:]
    get_spk_id = False
    get_tone_ids = False
    if am_name == 'speedyspeech':
        get_tone_ids = True
    if am_dataset in {"aishell3", "vctk", "mix"} and speaker_dict:
        get_spk_id = True
        spk_id = np.array([spk_id])

    frontend_dict = run_frontend(
        frontend=frontend,
        text=input,
        merge_sentences=merge_sentences,
        get_tone_ids=get_tone_ids,
        lang=lang)

    if get_tone_ids:
        tone_ids = frontend_dict['tone_ids']
        tones = tone_ids[0].numpy()
        tones_handle = am_predictor.get_input(1)
        tones_handle.from_numpy(tones)

    if get_spk_id:
        spk_id_handle = am_predictor.get_input(1)
        spk_id_handle.from_numpy(spk_id)
    phone_ids = frontend_dict['phone_ids']
    phones = phone_ids[0].numpy()
    phones_handle = am_predictor.get_input(0)
    phones_handle.from_numpy(phones)
    am_predictor.run()
    am_output_handle = am_predictor.get_output(0)
    am_output_data = am_output_handle.numpy()
    return am_output_data


def get_lite_voc_output(voc_predictor, input):
    mel_handle = voc_predictor.get_input(0)
    mel_handle.from_numpy(input)
    voc_predictor.run()
    voc_output_handle = voc_predictor.get_output(0)
    wav = voc_output_handle.numpy()
    return wav


def get_lite_am_sublayer_output(am_sublayer_predictor, input):
    input_handle = am_sublayer_predictor.get_input(0)
    input_handle.from_numpy(input)

    am_sublayer_predictor.run()
    am_sublayer_handle = am_sublayer_predictor.get_output(0)
    am_sublayer_output = am_sublayer_handle.numpy()
    return am_sublayer_output


def get_lite_streaming_am_output(input: str,
                                 am_encoder_infer_predictor,
                                 am_decoder_predictor,
                                 am_postnet_predictor,
                                 frontend,
                                 lang: str='zh',
                                 merge_sentences: bool=True):
    get_tone_ids = False
    frontend_dict = run_frontend(
        frontend=frontend,
        text=input,
        merge_sentences=merge_sentences,
        get_tone_ids=get_tone_ids,
        lang=lang)
    phone_ids = frontend_dict['phone_ids']
    phones = phone_ids[0].numpy()
    am_encoder_infer_output = get_lite_am_sublayer_output(
        am_encoder_infer_predictor, input=phones)
    am_decoder_output = get_lite_am_sublayer_output(
        am_decoder_predictor, input=am_encoder_infer_output)
    am_postnet_output = get_lite_am_sublayer_output(
        am_postnet_predictor, input=np.transpose(am_decoder_output, (0, 2, 1)))
    am_output_data = am_decoder_output + np.transpose(am_postnet_output,
                                                      (0, 2, 1))
    normalized_mel = am_output_data[0]
    return normalized_mel
