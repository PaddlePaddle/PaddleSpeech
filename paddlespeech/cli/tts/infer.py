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
import time
from collections import OrderedDict
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from ..executor import BaseExecutor
from ..log import logger
from ..utils import cli_register
from ..utils import download_and_decompress
from ..utils import MODEL_HOME
from ..utils import stats_wrapper
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.modules.normalizer import ZScore

__all__ = ['TTSExecutor']

pretrained_models = {
    # speedyspeech
    "speedyspeech_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/speedyspeech/speedyspeech_nosil_baker_ckpt_0.5.zip',
        'md5':
        '9edce23b1a87f31b814d9477bf52afbc',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_11400.pdz',
        'speech_stats':
        'feats_stats.npy',
        'phones_dict':
        'phone_id_map.txt',
        'tones_dict':
        'tone_id_map.txt',
    },

    # fastspeech2
    "fastspeech2_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip',
        'md5':
        '637d28a5e53aa60275612ba4393d5f22',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_76000.pdz',
        'speech_stats':
        'speech_stats.npy',
        'phones_dict':
        'phone_id_map.txt',
    },
    "fastspeech2_ljspeech-en": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_ljspeech_ckpt_0.5.zip',
        'md5':
        'ffed800c93deaf16ca9b3af89bfcd747',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_100000.pdz',
        'speech_stats':
        'speech_stats.npy',
        'phones_dict':
        'phone_id_map.txt',
    },
    "fastspeech2_aishell3-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_aishell3_ckpt_0.4.zip',
        'md5':
        'f4dd4a5f49a4552b77981f544ab3392e',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_96400.pdz',
        'speech_stats':
        'speech_stats.npy',
        'phones_dict':
        'phone_id_map.txt',
        'speaker_dict':
        'speaker_id_map.txt',
    },
    "fastspeech2_vctk-en": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_vctk_ckpt_0.5.zip',
        'md5':
        '743e5024ca1e17a88c5c271db9779ba4',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_66200.pdz',
        'speech_stats':
        'speech_stats.npy',
        'phones_dict':
        'phone_id_map.txt',
        'speaker_dict':
        'speaker_id_map.txt',
    },
    # tacotron2
    "tacotron2_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/tacotron2/tacotron2_csmsc_ckpt_0.2.0.zip',
        'md5':
        '0df4b6f0bcbe0d73c5ed6df8867ab91a',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_30600.pdz',
        'speech_stats':
        'speech_stats.npy',
        'phones_dict':
        'phone_id_map.txt',
    },
    "tacotron2_ljspeech-en": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/tacotron2/tacotron2_ljspeech_ckpt_0.2.0.zip',
        'md5':
        '6a5eddd81ae0e81d16959b97481135f3',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_60300.pdz',
        'speech_stats':
        'speech_stats.npy',
        'phones_dict':
        'phone_id_map.txt',
    },

    # pwgan
    "pwgan_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip',
        'md5':
        '2e481633325b5bdf0a3823c714d2c117',
        'config':
        'pwg_default.yaml',
        'ckpt':
        'pwg_snapshot_iter_400000.pdz',
        'speech_stats':
        'pwg_stats.npy',
    },
    "pwgan_ljspeech-en": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_ljspeech_ckpt_0.5.zip',
        'md5':
        '53610ba9708fd3008ccaf8e99dacbaf0',
        'config':
        'pwg_default.yaml',
        'ckpt':
        'pwg_snapshot_iter_400000.pdz',
        'speech_stats':
        'pwg_stats.npy',
    },
    "pwgan_aishell3-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip',
        'md5':
        'd7598fa41ad362d62f85ffc0f07e3d84',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_1000000.pdz',
        'speech_stats':
        'feats_stats.npy',
    },
    "pwgan_vctk-en": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_vctk_ckpt_0.1.1.zip',
        'md5':
        'b3da1defcde3e578be71eb284cb89f2c',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_1500000.pdz',
        'speech_stats':
        'feats_stats.npy',
    },
    # mb_melgan
    "mb_melgan_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_ckpt_0.1.1.zip',
        'md5':
        'ee5f0604e20091f0d495b6ec4618b90d',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_1000000.pdz',
        'speech_stats':
        'feats_stats.npy',
    },
    # style_melgan
    "style_melgan_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/style_melgan/style_melgan_csmsc_ckpt_0.1.1.zip',
        'md5':
        '5de2d5348f396de0c966926b8c462755',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_1500000.pdz',
        'speech_stats':
        'feats_stats.npy',
    },
    # hifigan
    "hifigan_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_ckpt_0.1.1.zip',
        'md5':
        'dd40a3d88dfcf64513fba2f0f961ada6',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_2500000.pdz',
        'speech_stats':
        'feats_stats.npy',
    },
    "hifigan_ljspeech-en": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_ljspeech_ckpt_0.2.0.zip',
        'md5':
        '70e9131695decbca06a65fe51ed38a72',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_2500000.pdz',
        'speech_stats':
        'feats_stats.npy',
    },
    "hifigan_aishell3-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_ckpt_0.2.0.zip',
        'md5':
        '3bb49bc75032ed12f79c00c8cc79a09a',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_2500000.pdz',
        'speech_stats':
        'feats_stats.npy',
    },
    "hifigan_vctk-en": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_vctk_ckpt_0.2.0.zip',
        'md5':
        '7da8f88359bca2457e705d924cf27bd4',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_2500000.pdz',
        'speech_stats':
        'feats_stats.npy',
    },

    # wavernn
    "wavernn_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/wavernn/wavernn_csmsc_ckpt_0.2.0.zip',
        'md5':
        'ee37b752f09bcba8f2af3b777ca38e13',
        'config':
        'default.yaml',
        'ckpt':
        'snapshot_iter_400000.pdz',
        'speech_stats':
        'feats_stats.npy',
    }
}

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


@cli_register(
    name='paddlespeech.tts', description='Text to Speech infer command.')
class TTSExecutor(BaseExecutor):
    def __init__(self):
        super().__init__()

        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.tts', add_help=True)
        self.parser.add_argument(
            '--input', type=str, default=None, help='Input text to generate.')
        # acoustic model
        self.parser.add_argument(
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
            ],
            help='Choose acoustic model type of tts task.')
        self.parser.add_argument(
            '--am_config',
            type=str,
            default=None,
            help='Config of acoustic model. Use deault config when it is None.')
        self.parser.add_argument(
            '--am_ckpt',
            type=str,
            default=None,
            help='Checkpoint file of acoustic model.')
        self.parser.add_argument(
            "--am_stat",
            type=str,
            default=None,
            help="mean and standard deviation used to normalize spectrogram when training acoustic model."
        )
        self.parser.add_argument(
            "--phones_dict",
            type=str,
            default=None,
            help="phone vocabulary file.")
        self.parser.add_argument(
            "--tones_dict",
            type=str,
            default=None,
            help="tone vocabulary file.")
        self.parser.add_argument(
            "--speaker_dict",
            type=str,
            default=None,
            help="speaker id map file.")
        self.parser.add_argument(
            '--spk_id',
            type=int,
            default=0,
            help='spk id for multi speaker acoustic model')
        # vocoder
        self.parser.add_argument(
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
            ],
            help='Choose vocoder type of tts task.')

        self.parser.add_argument(
            '--voc_config',
            type=str,
            default=None,
            help='Config of voc. Use deault config when it is None.')
        self.parser.add_argument(
            '--voc_ckpt',
            type=str,
            default=None,
            help='Checkpoint file of voc.')
        self.parser.add_argument(
            "--voc_stat",
            type=str,
            default=None,
            help="mean and standard deviation used to normalize spectrogram when training voc."
        )
        # other
        self.parser.add_argument(
            '--lang',
            type=str,
            default='zh',
            help='Choose model language. zh or en')
        self.parser.add_argument(
            '--device',
            type=str,
            default=paddle.get_device(),
            help='Choose device to execute model inference.')

        self.parser.add_argument(
            '--output', type=str, default='output.wav', help='output file name')
        self.parser.add_argument(
            '-d',
            '--job_dump_result',
            action='store_true',
            help='Save job result into file.')
        self.parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            help='Increase logger verbosity of current task.')

    def _get_pretrained_path(self, tag: str) -> os.PathLike:
        """
        Download and returns pretrained resources path of current task.
        """
        support_models = list(pretrained_models.keys())
        assert tag in pretrained_models, 'The model "{}" you want to use has not been supported, please choose other models.\nThe support models includes:\n\t\t{}\n'.format(
            tag, '\n\t\t'.join(support_models))

        res_path = os.path.join(MODEL_HOME, tag)
        decompressed_path = download_and_decompress(pretrained_models[tag],
                                                    res_path)
        decompressed_path = os.path.abspath(decompressed_path)
        logger.info(
            'Use pretrained model stored in: {}'.format(decompressed_path))
        return decompressed_path

    def _init_from_path(
            self,
            am: str='fastspeech2_csmsc',
            am_config: Optional[os.PathLike]=None,
            am_ckpt: Optional[os.PathLike]=None,
            am_stat: Optional[os.PathLike]=None,
            phones_dict: Optional[os.PathLike]=None,
            tones_dict: Optional[os.PathLike]=None,
            speaker_dict: Optional[os.PathLike]=None,
            voc: str='pwgan_csmsc',
            voc_config: Optional[os.PathLike]=None,
            voc_ckpt: Optional[os.PathLike]=None,
            voc_stat: Optional[os.PathLike]=None,
            lang: str='zh', ):
        """
        Init model and other resources from a specific path.
        """
        if hasattr(self, 'am_inference') and hasattr(self, 'voc_inference'):
            logger.info('Models had been initialized.')
            return
        # am
        am_tag = am + '-' + lang
        if am_ckpt is None or am_config is None or am_stat is None or phones_dict is None:
            am_res_path = self._get_pretrained_path(am_tag)
            self.am_res_path = am_res_path
            self.am_config = os.path.join(am_res_path,
                                          pretrained_models[am_tag]['config'])
            self.am_ckpt = os.path.join(am_res_path,
                                        pretrained_models[am_tag]['ckpt'])
            self.am_stat = os.path.join(
                am_res_path, pretrained_models[am_tag]['speech_stats'])
            # must have phones_dict in acoustic
            self.phones_dict = os.path.join(
                am_res_path, pretrained_models[am_tag]['phones_dict'])
            print("self.phones_dict:", self.phones_dict)
            logger.info(am_res_path)
            logger.info(self.am_config)
            logger.info(self.am_ckpt)
        else:
            self.am_config = os.path.abspath(am_config)
            self.am_ckpt = os.path.abspath(am_ckpt)
            self.am_stat = os.path.abspath(am_stat)
            self.phones_dict = os.path.abspath(phones_dict)
            self.am_res_path = os.path.dirname(os.path.abspath(self.am_config))
        print("self.phones_dict:", self.phones_dict)

        # for speedyspeech
        self.tones_dict = None
        if 'tones_dict' in pretrained_models[am_tag]:
            self.tones_dict = os.path.join(
                am_res_path, pretrained_models[am_tag]['tones_dict'])
            if tones_dict:
                self.tones_dict = tones_dict

        # for multi speaker fastspeech2
        self.speaker_dict = None
        if 'speaker_dict' in pretrained_models[am_tag]:
            self.speaker_dict = os.path.join(
                am_res_path, pretrained_models[am_tag]['speaker_dict'])
            if speaker_dict:
                self.speaker_dict = speaker_dict

        # voc
        voc_tag = voc + '-' + lang
        if voc_ckpt is None or voc_config is None or voc_stat is None:
            voc_res_path = self._get_pretrained_path(voc_tag)
            self.voc_res_path = voc_res_path
            self.voc_config = os.path.join(voc_res_path,
                                           pretrained_models[voc_tag]['config'])
            self.voc_ckpt = os.path.join(voc_res_path,
                                         pretrained_models[voc_tag]['ckpt'])
            self.voc_stat = os.path.join(
                voc_res_path, pretrained_models[voc_tag]['speech_stats'])
            logger.info(voc_res_path)
            logger.info(self.voc_config)
            logger.info(self.voc_ckpt)
        else:
            self.voc_config = os.path.abspath(voc_config)
            self.voc_ckpt = os.path.abspath(voc_ckpt)
            self.voc_stat = os.path.abspath(voc_stat)
            self.voc_res_path = os.path.dirname(
                os.path.abspath(self.voc_config))

        # Init body.
        with open(self.am_config) as f:
            self.am_config = CfgNode(yaml.safe_load(f))
        with open(self.voc_config) as f:
            self.voc_config = CfgNode(yaml.safe_load(f))

        with open(self.phones_dict, "r") as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        vocab_size = len(phn_id)
        print("vocab_size:", vocab_size)

        tone_size = None
        if self.tones_dict:
            with open(self.tones_dict, "r") as f:
                tone_id = [line.strip().split() for line in f.readlines()]
            tone_size = len(tone_id)
            print("tone_size:", tone_size)

        spk_num = None
        if self.speaker_dict:
            with open(self.speaker_dict, 'rt') as f:
                spk_id = [line.strip().split() for line in f.readlines()]
            spk_num = len(spk_id)
            print("spk_num:", spk_num)

        # frontend
        if lang == 'zh':
            self.frontend = Frontend(
                phone_vocab_path=self.phones_dict,
                tone_vocab_path=self.tones_dict)

        elif lang == 'en':
            self.frontend = English(phone_vocab_path=self.phones_dict)
        print("frontend done!")

        # acoustic model
        odim = self.am_config.n_mels
        # model: {model_name}_{dataset}
        am_name = am[:am.rindex('_')]

        am_class = dynamic_import(am_name, model_alias)
        am_inference_class = dynamic_import(am_name + '_inference', model_alias)

        if am_name == 'fastspeech2':
            am = am_class(
                idim=vocab_size,
                odim=odim,
                spk_num=spk_num,
                **self.am_config["model"])
        elif am_name == 'speedyspeech':
            am = am_class(
                vocab_size=vocab_size,
                tone_size=tone_size,
                **self.am_config["model"])
        elif am_name == 'tacotron2':
            am = am_class(idim=vocab_size, odim=odim, **self.am_config["model"])

        am.set_state_dict(paddle.load(self.am_ckpt)["main_params"])
        am.eval()
        am_mu, am_std = np.load(self.am_stat)
        am_mu = paddle.to_tensor(am_mu)
        am_std = paddle.to_tensor(am_std)
        am_normalizer = ZScore(am_mu, am_std)
        self.am_inference = am_inference_class(am_normalizer, am)
        self.am_inference.eval()
        print("acoustic model done!")

        # vocoder
        # model: {model_name}_{dataset}
        voc_name = voc[:voc.rindex('_')]
        voc_class = dynamic_import(voc_name, model_alias)
        voc_inference_class = dynamic_import(voc_name + '_inference',
                                             model_alias)
        if voc_name != 'wavernn':
            voc = voc_class(**self.voc_config["generator_params"])
            voc.set_state_dict(paddle.load(self.voc_ckpt)["generator_params"])
            voc.remove_weight_norm()
            voc.eval()
        else:
            voc = voc_class(**self.voc_config["model"])
            voc.set_state_dict(paddle.load(self.voc_ckpt)["main_params"])
            voc.eval()
        voc_mu, voc_std = np.load(self.voc_stat)
        voc_mu = paddle.to_tensor(voc_mu)
        voc_std = paddle.to_tensor(voc_std)
        voc_normalizer = ZScore(voc_mu, voc_std)
        self.voc_inference = voc_inference_class(voc_normalizer, voc)
        self.voc_inference.eval()
        print("voc done!")

    def preprocess(self, input: Any, *args, **kwargs):
        """
        Input preprocess and return paddle.Tensor stored in self._inputs.
        Input content can be a text(tts), a file(asr, cls), a stream(not supported yet) or anything needed.

        Args:
            input (Any): Input text/file/stream or other content.
        """
        pass

    @paddle.no_grad()
    def infer(self,
              text: str,
              lang: str='zh',
              am: str='fastspeech2_csmsc',
              spk_id: int=0):
        """
        Model inference and result stored in self.output.
        """
        am_name = am[:am.rindex('_')]
        am_dataset = am[am.rindex('_') + 1:]
        get_tone_ids = False
        merge_sentences = False
        frontend_st = time.time()
        if am_name == 'speedyspeech':
            get_tone_ids = True
        if lang == 'zh':
            input_ids = self.frontend.get_input_ids(
                text,
                merge_sentences=merge_sentences,
                get_tone_ids=get_tone_ids)
            phone_ids = input_ids["phone_ids"]
            if get_tone_ids:
                tone_ids = input_ids["tone_ids"]
        elif lang == 'en':
            input_ids = self.frontend.get_input_ids(
                text, merge_sentences=merge_sentences)
            phone_ids = input_ids["phone_ids"]
        else:
            print("lang should in {'zh', 'en'}!")
        self.frontend_time = time.time() - frontend_st

        self.am_time = 0
        self.voc_time = 0
        flags = 0
        for i in range(len(phone_ids)):
            am_st = time.time()
            part_phone_ids = phone_ids[i]
            # am
            if am_name == 'speedyspeech':
                part_tone_ids = tone_ids[i]
                mel = self.am_inference(part_phone_ids, part_tone_ids)
            # fastspeech2
            else:
                # multi speaker
                if am_dataset in {"aishell3", "vctk"}:
                    mel = self.am_inference(
                        part_phone_ids, spk_id=paddle.to_tensor(spk_id))
                else:
                    mel = self.am_inference(part_phone_ids)
            self.am_time += (time.time() - am_st)
            # voc
            voc_st = time.time()
            wav = self.voc_inference(mel)
            if flags == 0:
                wav_all = wav
                flags = 1
            else:
                wav_all = paddle.concat([wav_all, wav])
            self.voc_time += (time.time() - voc_st)
        self._outputs['wav'] = wav_all

    def postprocess(self, output: str='output.wav') -> Union[str, os.PathLike]:
        """
        Output postprocess and return results.
        This method get model output from self._outputs and convert it into human-readable results.

        Returns:
            Union[str, os.PathLike]: Human-readable results such as texts and audio files.
        """
        output = os.path.abspath(os.path.expanduser(output))
        sf.write(
            output, self._outputs['wav'].numpy(), samplerate=self.am_config.fs)
        return output

    def execute(self, argv: List[str]) -> bool:
        """
        Command line entry.
        """

        args = self.parser.parse_args(argv)

        am = args.am
        am_config = args.am_config
        am_ckpt = args.am_ckpt
        am_stat = args.am_stat
        phones_dict = args.phones_dict
        print("phones_dict:", phones_dict)
        tones_dict = args.tones_dict
        speaker_dict = args.speaker_dict
        voc = args.voc
        voc_config = args.voc_config
        voc_ckpt = args.voc_ckpt
        voc_stat = args.voc_stat
        lang = args.lang
        device = args.device
        spk_id = args.spk_id

        if not args.verbose:
            self.disable_task_loggers()

        task_source = self.get_task_source(args.input)
        task_results = OrderedDict()
        has_exceptions = False

        for id_, input_ in task_source.items():
            if len(task_source) > 1:
                assert isinstance(args.output,
                                  str) and args.output.endswith('.wav')
                output = args.output.replace('.wav', f'_{id_}.wav')
            else:
                output = args.output

            try:
                res = self(
                    text=input_,
                    # acoustic model related
                    am=am,
                    am_config=am_config,
                    am_ckpt=am_ckpt,
                    am_stat=am_stat,
                    phones_dict=phones_dict,
                    tones_dict=tones_dict,
                    speaker_dict=speaker_dict,
                    spk_id=spk_id,
                    # vocoder related
                    voc=voc,
                    voc_config=voc_config,
                    voc_ckpt=voc_ckpt,
                    voc_stat=voc_stat,
                    # other
                    lang=lang,
                    device=device,
                    output=output)
                task_results[id_] = res
            except Exception as e:
                has_exceptions = True
                task_results[id_] = f'{e.__class__.__name__}: {e}'

        self.process_task_results(args.input, task_results,
                                  args.job_dump_result)

        if has_exceptions:
            return False
        else:
            return True

    @stats_wrapper
    def __call__(self,
                 text: str,
                 am: str='fastspeech2_csmsc',
                 am_config: Optional[os.PathLike]=None,
                 am_ckpt: Optional[os.PathLike]=None,
                 am_stat: Optional[os.PathLike]=None,
                 spk_id: int=0,
                 phones_dict: Optional[os.PathLike]=None,
                 tones_dict: Optional[os.PathLike]=None,
                 speaker_dict: Optional[os.PathLike]=None,
                 voc: str='pwgan_csmsc',
                 voc_config: Optional[os.PathLike]=None,
                 voc_ckpt: Optional[os.PathLike]=None,
                 voc_stat: Optional[os.PathLike]=None,
                 lang: str='zh',
                 device: str=paddle.get_device(),
                 output: str='output.wav'):
        """
        Python API to call an executor.
        """
        paddle.set_device(device)
        self._init_from_path(
            am=am,
            am_config=am_config,
            am_ckpt=am_ckpt,
            am_stat=am_stat,
            phones_dict=phones_dict,
            tones_dict=tones_dict,
            speaker_dict=speaker_dict,
            voc=voc,
            voc_config=voc_config,
            voc_ckpt=voc_ckpt,
            voc_stat=voc_stat,
            lang=lang)

        self.infer(text=text, lang=lang, am=am, spk_id=spk_id)

        res = self.postprocess(output=output)

        return res
