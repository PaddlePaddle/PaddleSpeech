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
from ..utils import cli_register
from ..utils import download_and_decompress
from ..utils import logger
from ..utils import MODEL_HOME
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
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_vctk_ckpt_0.5.zip',
        'md5':
        '322ca688aec9b127cec2788b65aa3d52',
        'config':
        'pwg_default.yaml',
        'ckpt':
        'pwg_snapshot_iter_1000000.pdz',
        'speech_stats':
        'pwg_stats.npy',
    },
    # mb_melgan
    "mb_melgan_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_baker_finetune_ckpt_0.5.zip',
        'md5':
        'b69322ab4ea766d955bd3d9af7dc5f2d',
        'config':
        'finetune.yaml',
        'ckpt':
        'snapshot_iter_2000000.pdz',
        'speech_stats':
        'feats_stats.npy',
    },
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
    # voc
    "pwgan":
    "paddlespeech.t2s.models.parallel_wavegan:PWGGenerator",
    "pwgan_inference":
    "paddlespeech.t2s.models.parallel_wavegan:PWGInference",
    "mb_melgan":
    "paddlespeech.t2s.models.melgan:MelGANGenerator",
    "mb_melgan_inference":
    "paddlespeech.t2s.models.melgan:MelGANInference",
}


@cli_register(
    name='paddlespeech.tts', description='Text to Speech infer command.')
class TTSExecutor(BaseExecutor):
    def __init__(self):
        super().__init__()

        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.tts', add_help=True)
        self.parser.add_argument(
            '--input', type=str, required=True, help='Input text to generate.')
        # acoustic model
        self.parser.add_argument(
            '--am',
            type=str,
            default='fastspeech2_csmsc',
            choices=[
                'speedyspeech_csmsc', 'fastspeech2_csmsc',
                'fastspeech2_ljspeech', 'fastspeech2_aishell3',
                'fastspeech2_vctk'
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
                'pwgan_csmsc', 'pwgan_ljspeech', 'pwgan_aishell3', 'pwgan_vctk',
                'mb_melgan_csmsc'
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

    def _get_pretrained_path(self, tag: str) -> os.PathLike:
        """
        Download and returns pretrained resources path of current task.
        """
        assert tag in pretrained_models, 'Can not find pretrained resources of {}.'.format(
            tag)

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
        if hasattr(self, 'am') and hasattr(self, 'voc'):
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

        # Enter the path of model root

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

        am.set_state_dict(paddle.load(self.am_ckpt)["main_params"])
        am.eval()
        am_mu, am_std = np.load(self.am_stat)
        am_mu = paddle.to_tensor(am_mu)
        am_std = paddle.to_tensor(am_std)
        am_normalizer = ZScore(am_mu, am_std)
        self.am_inference = am_inference_class(am_normalizer, am)
        print("acoustic model done!")

        # vocoder
        # model: {model_name}_{dataset}
        voc_name = '_'.join(voc.split('_')[:-1])
        voc_class = dynamic_import(voc_name, model_alias)
        voc_inference_class = dynamic_import(voc_name + '_inference',
                                             model_alias)
        voc = voc_class(**self.voc_config["generator_params"])
        voc.set_state_dict(paddle.load(self.voc_ckpt)["generator_params"])
        voc.remove_weight_norm()
        voc.eval()
        voc_mu, voc_std = np.load(self.voc_stat)
        voc_mu = paddle.to_tensor(voc_mu)
        voc_std = paddle.to_tensor(voc_std)
        voc_normalizer = ZScore(voc_mu, voc_std)
        self.voc_inference = voc_inference_class(voc_normalizer, voc)
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
        model_name = am[:am.rindex('_')]
        dataset = am[am.rindex('_') + 1:]
        get_tone_ids = False
        if 'speedyspeech' in model_name:
            get_tone_ids = True
        if lang == 'zh':
            input_ids = self.frontend.get_input_ids(
                text, merge_sentences=True, get_tone_ids=get_tone_ids)
            phone_ids = input_ids["phone_ids"]
            phone_ids = phone_ids[0]
            if get_tone_ids:
                tone_ids = input_ids["tone_ids"]
                tone_ids = tone_ids[0]
        elif lang == 'en':
            input_ids = self.frontend.get_input_ids(text)
            phone_ids = input_ids["phone_ids"]
        else:
            print("lang should in {'zh', 'en'}!")

        # am
        if 'speedyspeech' in model_name:
            mel = self.am_inference(phone_ids, tone_ids)
        # fastspeech2
        else:
            # multi speaker
            if dataset in {"aishell3", "vctk"}:
                mel = self.am_inference(
                    phone_ids, spk_id=paddle.to_tensor(spk_id))

            else:
                mel = self.am_inference(phone_ids)

        # voc
        wav = self.voc_inference(mel)
        self._outputs['wav'] = wav

    def postprocess(self, output: str='output.wav'):
        """
        Output postprocess and return results.
        This method get model output from self._outputs and convert it into human-readable results.

        Returns:
            Union[str, os.PathLike]: Human-readable results such as texts and audio files.
        """
        sf.write(
            output, self._outputs['wav'].numpy(), samplerate=self.am_config.fs)
        return output

    def execute(self, argv: List[str]) -> bool:
        """
        Command line entry.
        """

        args = self.parser.parse_args(argv)

        text = args.input
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
        output = args.output
        spk_id = args.spk_id

        try:
            res = self(
                text=text,
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
            logger.info('TTS Result Saved in: {}'.format(res))
            return True
        except Exception as e:
            logger.exception(e)
            return False

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
                 device: str='gpu',
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
