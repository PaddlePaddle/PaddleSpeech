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
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.modules.normalizer import ZScore
from yacs.config import CfgNode

from ..executor import BaseExecutor
from ..utils import cli_register
from ..utils import download_and_decompress
from ..utils import logger
from ..utils import MODEL_HOME

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
    # vocoder
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
        # acoustic_model
        self.parser.add_argument(
            '--acoustic_model',
            type=str,
            default='fastspeech2_csmsc',
            help='Choose model type of asr task.')
        self.parser.add_argument(
            '--acoustic_model_config',
            type=str,
            default=None,
            help='Config of acoustic model. Use deault config when it is None.')
        self.parser.add_argument(
            '--acoustic_model_ckpt',
            type=str,
            default=None,
            help='Checkpoint file of acoustic model.')
        self.parser.add_argument(
            "--acoustic_model_stat",
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
            '--vocoder',
            type=str,
            default='pwgan_csmsc',
            help='Choose model type of asr task.')

        self.parser.add_argument(
            '--vocoder_config',
            type=str,
            default=None,
            help='Config of vocoder. Use deault config when it is None.')
        self.parser.add_argument(
            '--vocoder_ckpt',
            type=str,
            default=None,
            help='Checkpoint file of vocoder.')
        self.parser.add_argument(
            "--vocoder_stat",
            type=str,
            help="mean and standard deviation used to normalize spectrogram when training vocoder."
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
            acoustic_model: str='fastspeech2_csmsc',
            acoustic_model_config: Optional[os.PathLike]=None,
            acoustic_model_ckpt: Optional[os.PathLike]=None,
            acoustic_model_stat: Optional[os.PathLike]=None,
            phones_dict: Optional[os.PathLike]=None,
            tones_dict: Optional[os.PathLike]=None,
            speaker_dict: Optional[os.PathLike]=None,
            vocoder: str='pwgan_csmsc',
            vocoder_config: Optional[os.PathLike]=None,
            vocoder_ckpt: Optional[os.PathLike]=None,
            vocoder_stat: Optional[os.PathLike]=None,
            lang: str='zh', ):
        """
        Init model and other resources from a specific path.
        """
        if hasattr(self, 'acoustic_model') and hasattr(self, 'vocoder'):
            logger.info('Models had been initialized.')
            return
        # acoustic_model
        acoustic_model_tag = acoustic_model + '-' + lang
        if acoustic_model_ckpt is None or acoustic_model_config is None or acoustic_model_stat is None or phones_dict is None:
            acoustic_model_res_path = self._get_pretrained_path(
                acoustic_model_tag)
            self.acoustic_model_res_path = acoustic_model_res_path
            self.acoustic_model_config = os.path.join(
                acoustic_model_res_path,
                pretrained_models[acoustic_model_tag]['config'])
            self.acoustic_model_ckpt = os.path.join(
                acoustic_model_res_path,
                pretrained_models[acoustic_model_tag]['ckpt'])
            self.acoustic_model_stat = os.path.join(
                acoustic_model_res_path,
                pretrained_models[acoustic_model_tag]['speech_stats'])
            # must have phones_dict in acoustic
            self.phones_dict = os.path.join(
                acoustic_model_res_path,
                pretrained_models[acoustic_model_tag]['phones_dict'])
            print("self.phones_dict:", self.phones_dict)
            logger.info(acoustic_model_res_path)
            logger.info(self.acoustic_model_config)
            logger.info(self.acoustic_model_ckpt)
        else:
            self.acoustic_model_config = os.path.abspath(acoustic_model_config)
            self.acoustic_model_ckpt = os.path.abspath(acoustic_model_ckpt)
            self.acoustic_model_stat = os.path.abspath(acoustic_model_stat)
            self.phones_dict = os.path.abspath(phones_dict)
            self.acoustic_model_res_path = os.path.dirname(
                os.path.abspath(self.acoustic_model_config))
        print("self.phones_dict:", self.phones_dict)

        # for speedyspeech
        self.tones_dict = None
        if 'tones_dict' in pretrained_models[acoustic_model_tag]:
            self.tones_dict = os.path.join(
                acoustic_model_res_path,
                pretrained_models[acoustic_model_tag]['tones_dict'])
            if tones_dict:
                self.tones_dict = tones_dict

        # for multi speaker fastspeech2
        self.speaker_dict = None
        if 'speaker_dict' in pretrained_models[acoustic_model_tag]:
            self.speaker_dict = os.path.join(
                acoustic_model_res_path,
                pretrained_models[acoustic_model_tag]['speaker_dict'])
            if speaker_dict:
                self.speaker_dict = speaker_dict

        # vocoder
        vocoder_tag = vocoder + '-' + lang
        if vocoder_ckpt is None or vocoder_config is None or vocoder_stat is None:
            vocoder_res_path = self._get_pretrained_path(vocoder_tag)
            self.vocoder_res_path = vocoder_res_path
            self.vocoder_config = os.path.join(
                vocoder_res_path, pretrained_models[vocoder_tag]['config'])
            self.vocoder_ckpt = os.path.join(
                vocoder_res_path, pretrained_models[vocoder_tag]['ckpt'])
            self.vocoder_stat = os.path.join(
                vocoder_res_path,
                pretrained_models[vocoder_tag]['speech_stats'])
            logger.info(vocoder_res_path)
            logger.info(self.vocoder_config)
            logger.info(self.vocoder_ckpt)
        else:
            self.vocoder_config = os.path.abspath(vocoder_config)
            self.vocoder_ckpt = os.path.abspath(vocoder_ckpt)
            self.vocoder_stat = os.path.abspath(vocoder_stat)
            self.vocoder_res_path = os.path.dirname(
                os.path.abspath(self.vocoder_config))

        # Init body.
        with open(self.acoustic_model_config) as f:
            self.acoustic_model_config = CfgNode(yaml.safe_load(f))
        with open(self.vocoder_config) as f:
            self.vocoder_config = CfgNode(yaml.safe_load(f))

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

        # acoustic_model
        odim = self.acoustic_model_config.n_mels
        # model: {model_name}_{dataset}
        acoustic_model_name = acoustic_model[:acoustic_model.rindex('_')]

        acoustic_model_class = dynamic_import(acoustic_model_name, model_alias)
        acoustic_model_inference_class = dynamic_import(
            acoustic_model_name + '_inference', model_alias)

        if acoustic_model_name == 'fastspeech2':
            acoustic_model = acoustic_model_class(
                idim=vocab_size,
                odim=odim,
                spk_num=spk_num,
                **self.acoustic_model_config["model"])
        elif acoustic_model_name == 'speedyspeech':
            acoustic_model = acoustic_model_class(
                vocab_size=vocab_size,
                tone_size=tone_size,
                **self.acoustic_model_config["model"])

        acoustic_model.set_state_dict(
            paddle.load(self.acoustic_model_ckpt)["main_params"])
        acoustic_model.eval()
        acoustic_model_mu, acoustic_model_std = np.load(
            self.acoustic_model_stat)
        acoustic_model_mu = paddle.to_tensor(acoustic_model_mu)
        acoustic_model_std = paddle.to_tensor(acoustic_model_std)
        acoustic_model_normalizer = ZScore(acoustic_model_mu,
                                           acoustic_model_std)
        self.acoustic_model_inference = acoustic_model_inference_class(
            acoustic_model_normalizer, acoustic_model)
        print("acoustic model done!")

        # vocoder
        # model: {model_name}_{dataset}
        vocoder_name = '_'.join(vocoder.split('_')[:-1])
        vocoder_class = dynamic_import(vocoder_name, model_alias)
        vocoder_inference_class = dynamic_import(vocoder_name + '_inference',
                                                 model_alias)
        vocoder = vocoder_class(**self.vocoder_config["generator_params"])
        vocoder.set_state_dict(
            paddle.load(self.vocoder_ckpt)["generator_params"])
        vocoder.remove_weight_norm()
        vocoder.eval()
        vocoder_mu, vocoder_std = np.load(self.vocoder_stat)
        vocoder_mu = paddle.to_tensor(vocoder_mu)
        vocoder_std = paddle.to_tensor(vocoder_std)
        vocoder_normalizer = ZScore(vocoder_mu, vocoder_std)
        self.vocoder_inference = vocoder_inference_class(vocoder_normalizer,
                                                         vocoder)
        print("vocoder done!")

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
              acoustic_model: str='fastspeech2_csmsc',
              spk_id: int=0,
              output: str='output.wav'):
        """
        Model inference and result stored in self.output.
        """
        model_name = acoustic_model[:acoustic_model.rindex('_')]
        dataset = acoustic_model[acoustic_model.rindex('_') + 1:]
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

        # acoustic_model
        if 'speedyspeech' in model_name:
            mel = self.acoustic_model_inference(phone_ids, tone_ids)
        # fastspeech2
        else:
            # multi speaker
            if dataset in {"aishell3", "vctk"}:
                mel = self.acoustic_model_inference(
                    phone_ids, spk_id=paddle.to_tensor(spk_id))

            else:
                mel = self.acoustic_model_inference(phone_ids)

        # vocoder
        wav = self.vocoder_inference(mel)
        sf.write(output, wav.numpy(), samplerate=self.acoustic_model_config.fs)
        return output

    def postprocess(self, *args, **kwargs) -> Union[str, os.PathLike]:
        """
        Output postprocess and return results.
        This method get model output from self._outputs and convert it into human-readable results.

        Returns:
            Union[str, os.PathLike]: Human-readable results such as texts and audio files.
        """
        pass

    def execute(self, argv: List[str]) -> bool:
        """
        Command line entry.
        """

        args = self.parser.parse_args(argv)

        text = args.input
        acoustic_model = args.acoustic_model
        acoustic_model_config = args.acoustic_model_config
        acoustic_model_ckpt = args.acoustic_model_ckpt
        acoustic_model_stat = args.acoustic_model_stat
        phones_dict = args.phones_dict
        print("phones_dict:", phones_dict)
        tones_dict = args.tones_dict
        speaker_dict = args.speaker_dict
        vocoder = args.vocoder
        vocoder_config = args.vocoder_config
        vocoder_ckpt = args.vocoder_ckpt
        vocoder_stat = args.vocoder_stat
        lang = args.lang
        device = args.device
        output = args.output
        spk_id = args.spk_id

        try:
            res = self(
                text=text,
                # acoustic_model related
                acoustic_model=acoustic_model,
                acoustic_model_config=acoustic_model_config,
                acoustic_model_ckpt=acoustic_model_ckpt,
                acoustic_model_stat=acoustic_model_stat,
                phones_dict=phones_dict,
                tones_dict=tones_dict,
                speaker_dict=speaker_dict,
                spk_id=spk_id,
                # vocoder related
                vocoder=vocoder,
                vocoder_config=vocoder_config,
                vocoder_ckpt=vocoder_ckpt,
                vocoder_stat=vocoder_stat,
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
                 acoustic_model: str='fastspeech2_csmsc',
                 acoustic_model_config: Optional[os.PathLike]=None,
                 acoustic_model_ckpt: Optional[os.PathLike]=None,
                 acoustic_model_stat: Optional[os.PathLike]=None,
                 spk_id: int=0,
                 phones_dict: Optional[os.PathLike]=None,
                 tones_dict: Optional[os.PathLike]=None,
                 speaker_dict: Optional[os.PathLike]=None,
                 vocoder: str='pwgan_csmsc',
                 vocoder_config: Optional[os.PathLike]=None,
                 vocoder_ckpt: Optional[os.PathLike]=None,
                 vocoder_stat: Optional[os.PathLike]=None,
                 lang: str='zh',
                 device: str='gpu',
                 output: str='output.wav'):
        """
        Python API to call an executor.
        """
        paddle.set_device(device)
        self._init_from_path(
            acoustic_model=acoustic_model,
            acoustic_model_config=acoustic_model_config,
            acoustic_model_ckpt=acoustic_model_ckpt,
            acoustic_model_stat=acoustic_model_stat,
            phones_dict=phones_dict,
            tones_dict=tones_dict,
            speaker_dict=speaker_dict,
            vocoder=vocoder,
            vocoder_config=vocoder_config,
            vocoder_ckpt=vocoder_ckpt,
            vocoder_stat=vocoder_stat,
            lang=lang)
        res = self.infer(
            text=text,
            lang=lang,
            acoustic_model=acoustic_model,
            spk_id=spk_id,
            output=output)

        return res
