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
import base64
import io
import os
from typing import Optional

import librosa
import numpy as np
import paddle
import soundfile as sf
import yaml
from engine.base_engine import BaseEngine
from scipy.io import wavfile

from paddlespeech.cli.log import logger
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.cli.utils import download_and_decompress
from paddlespeech.cli.utils import MODEL_HOME
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from utils.audio_process import change_speed
from utils.errors import ErrorCode
from utils.exception import ServerBaseException
from utils.paddle_predictor import init_predictor
from utils.paddle_predictor import run_model
#from paddle.inference import Config
#from paddle.inference import create_predictor

__all__ = ['TTSEngine']

# Static model applied on paddle inference
pretrained_models = {
    # speedyspeech
    "speedyspeech_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/speedyspeech/speedyspeech_nosil_baker_static_0.5.zip',
        'md5':
        '9a849a74d1be0c758dd5a1b9c8f77f3d',
        'model':
        'speedyspeech_csmsc.pdmodel',
        'params':
        'speedyspeech_csmsc.pdiparams',
        'phones_dict':
        'phone_id_map.txt',
        'tones_dict':
        'tone_id_map.txt',
    },
    # fastspeech2
    "fastspeech2_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_static_0.4.zip',
        'md5':
        '8eb01c2e4bc7e8b59beaa9fa046069cf',
        'model':
        'fastspeech2_csmsc.pdmodel',
        'params':
        'fastspeech2_csmsc.pdiparams',
        'phones_dict':
        'phone_id_map.txt',
    },
    # pwgan
    "pwgan_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_static_0.4.zip',
        'md5':
        'e3504aed9c5a290be12d1347836d2742',
        'model':
        'pwgan_csmsc.pdmodel',
        'params':
        'pwgan_csmsc.pdiparams',
    },
    # mb_melgan
    "mb_melgan_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_static_0.1.1.zip',
        'md5':
        'ac6eee94ba483421d750433f4c3b8d36',
        'model':
        'mb_melgan_csmsc.pdmodel',
        'params':
        'mb_melgan_csmsc.pdiparams',
    },
    # hifigan
    "hifigan_csmsc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_static_0.1.1.zip',
        'md5':
        '7edd8c436b3a5546b3a7cb8cff9d5a0c',
        'model':
        'hifigan_csmsc.pdmodel',
        'params':
        'hifigan_csmsc.pdiparams',
    },
}


class TTSServerExecutor(TTSExecutor):
    def __init__(self):
        super().__init__()

        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.tts', add_help=True)
        self.parser.add_argument(
            '--conf',
            type=str,
            default='./conf/tts/tts_pd.yaml',
            help='Configuration parameters.')

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
            am_model: Optional[os.PathLike]=None,
            am_params: Optional[os.PathLike]=None,
            phones_dict: Optional[os.PathLike]=None,
            tones_dict: Optional[os.PathLike]=None,
            speaker_dict: Optional[os.PathLike]=None,
            voc: str='pwgan_csmsc',
            voc_model: Optional[os.PathLike]=None,
            voc_params: Optional[os.PathLike]=None,
            lang: str='zh',
            am_predictor_conf: dict=None,
            voc_predictor_conf: dict=None, ):
        """
        Init model and other resources from a specific path.
        """
        if hasattr(self, 'am') and hasattr(self, 'voc'):
            logger.info('Models had been initialized.')
            return
        # am
        am_tag = am + '-' + lang
        if phones_dict is None:
            print("please input phones_dict!")
            ### 后续下载的模型里加上 phone 和 tone的 dict 就不用这个了
        #if am_model is None or am_params is None or phones_dict is None:
        if am_model is None or am_params is None:
            am_res_path = self._get_pretrained_path(am_tag)
            self.am_res_path = am_res_path
            self.am_model = os.path.join(am_res_path,
                                         pretrained_models[am_tag]['model'])
            self.am_params = os.path.join(am_res_path,
                                          pretrained_models[am_tag]['params'])
            # must have phones_dict in acoustic
            #self.phones_dict = os.path.join(
            #am_res_path, pretrained_models[am_tag]['phones_dict'])

            self.phones_dict = os.path.abspath(phones_dict)
            logger.info(am_res_path)
            logger.info(self.am_model)
            logger.info(self.am_params)
        else:
            self.am_model = os.path.abspath(am_model)
            self.am_params = os.path.abspath(am_params)
            self.phones_dict = os.path.abspath(phones_dict)
            self.am_res_path = os.path.dirname(os.path.abspath(self.am_model))
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
        if voc_model is None or voc_params is None:
            voc_res_path = self._get_pretrained_path(voc_tag)
            self.voc_res_path = voc_res_path
            self.voc_model = os.path.join(voc_res_path,
                                          pretrained_models[voc_tag]['model'])
            self.voc_params = os.path.join(voc_res_path,
                                           pretrained_models[voc_tag]['params'])
            logger.info(voc_res_path)
            logger.info(self.voc_model)
            logger.info(self.voc_params)
        else:
            self.voc_model = os.path.abspath(voc_model)
            self.voc_params = os.path.abspath(voc_params)
            self.voc_res_path = os.path.dirname(os.path.abspath(self.voc_model))

        # Init body.
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

        # am predictor
        self.am_predictor_conf = am_predictor_conf
        self.am_predictor = init_predictor(
            model_file=self.am_model,
            params_file=self.am_params,
            predictor_conf=self.am_predictor_conf)

        # voc predictor
        self.voc_predictor_conf = voc_predictor_conf
        self.voc_predictor = init_predictor(
            model_file=self.voc_model,
            params_file=self.voc_params,
            predictor_conf=self.voc_predictor_conf)

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

        flags = 0
        for i in range(len(phone_ids)):
            part_phone_ids = phone_ids[i]
            # am
            if am_name == 'speedyspeech':
                part_tone_ids = tone_ids[i]
                am_result = run_model(
                    self.am_predictor,
                    [part_phone_ids.numpy(), part_tone_ids.numpy()])
                mel = am_result[0]

            # fastspeech2
            else:
                # multi speaker  do not have static model
                if am_dataset in {"aishell3", "vctk"}:
                    pass
                else:
                    am_result = run_model(self.am_predictor,
                                          [part_phone_ids.numpy()])
                    mel = am_result[0]
            # voc
            voc_result = run_model(self.voc_predictor, [mel])
            wav = voc_result[0]
            wav = paddle.to_tensor(wav)

            if flags == 0:
                wav_all = wav
                flags = 1
            else:
                wav_all = paddle.concat([wav_all, wav])
        self._outputs['wav'] = wav_all


class TTSEngine(BaseEngine):
    """TTS server engine

    Args:
        metaclass: Defaults to Singleton.
    """

    def __init__(self, name=None):
        """Initialize TTS server engine
        """
        super(TTSEngine, self).__init__()
        self.executor = TTSServerExecutor()

        config_path = self.executor.parser.parse_args().conf
        with open(config_path, 'rt') as f:
            self.conf_dict = yaml.safe_load(f)

        self.executor._init_from_path(
            am=self.conf_dict["am"],
            am_model=self.conf_dict["am_model"],
            am_params=self.conf_dict["am_params"],
            phones_dict=self.conf_dict["phones_dict"],
            tones_dict=self.conf_dict["tones_dict"],
            speaker_dict=self.conf_dict["speaker_dict"],
            voc=self.conf_dict["voc"],
            voc_model=self.conf_dict["voc_model"],
            voc_params=self.conf_dict["voc_params"],
            lang=self.conf_dict["lang"],
            am_predictor_conf=self.conf_dict["am_predictor_conf"],
            voc_predictor_conf=self.conf_dict["voc_predictor_conf"], )

        logger.info("Initialize TTS server engine successfully.")

    def postprocess(self,
                    wav,
                    original_fs: int,
                    target_fs: int=16000,
                    volume: float=1.0,
                    speed: float=1.0,
                    audio_path: str=None):
        """Post-processing operations, including speech, volume, sample rate, save audio file

        Args:
            wav (numpy(float)): Synthesized audio sample points
            original_fs (int): original audio sample rate
            target_fs (int): target audio sample rate
            volume (float): target volume
            speed (float): target speed
        """

        # transform sample_rate
        if target_fs == 0 or target_fs > original_fs:
            target_fs = original_fs
            wav_tar_fs = wav
        else:
            wav_tar_fs = librosa.resample(
                np.squeeze(wav), original_fs, target_fs)

        # transform volume
        wav_vol = wav_tar_fs * volume

        # transform speed
        try:  # windows not support soxbindings
            wav_speed = change_speed(wav_vol, speed, target_fs)
        except:
            raise ServerBaseException(
                ErrorCode.SERVER_INTERNAL_ERR,
                "Can not install soxbindings on your system.")

        # wav to base64
        buf = io.BytesIO()
        wavfile.write(buf, target_fs, wav_speed)
        base64_bytes = base64.b64encode(buf.read())
        wav_base64 = base64_bytes.decode('utf-8')

        # save audio
        if audio_path is not None and audio_path.endswith(".wav"):
            sf.write(audio_path, wav_speed, target_fs)
        elif audio_path is not None and audio_path.endswith(".pcm"):
            wav_norm = wav_speed * (32767 / max(0.001,
                                                np.max(np.abs(wav_speed))))
            with open(audio_path, "wb") as f:
                f.write(wav_norm.astype(np.int16))

        return target_fs, wav_base64

    def run(self,
            sentence: str,
            spk_id: int=0,
            speed: float=1.0,
            volume: float=1.0,
            sample_rate: int=0,
            save_path: str=None):
        """get the result of the server response

        Args:
            sentence (str): sentence to be synthesized
            spk_id (int, optional): speaker id. Defaults to 0.
            speed (float, optional): audio speed, 0 < speed <=3.0. Defaults to 1.0.
            volume (float, optional): The volume relative to the audio synthesized by the model, 
            0 < volume <=3.0. Defaults to 1.0.
            sample_rate (int, optional): Set the sample rate of the synthesized audio. 
            0 represents the sample rate for model synthesis. Defaults to 0.
            save_path (str, optional): The save path of the synthesized audio. Defaults to None.

        Raises:
            ServerBaseException: Exception
            ServerBaseException: Exception

        Returns:
            lang, target_sample_rate, wav_base64
        """

        lang = self.conf_dict["lang"]

        try:
            self.executor.infer(
                text=sentence,
                lang=lang,
                am=self.conf_dict["am"],
                spk_id=spk_id)
        except:
            raise ServerBaseException(ErrorCode.SERVER_INTERNAL_ERR,
                                      "tts infer failed.")

        try:
            target_sample_rate, wav_base64 = self.postprocess(
                wav=self.executor._outputs['wav'].numpy(),
                #original_fs=self.executor.am_config.fs,
                original_fs=24000,  # TODO get sample rate from model
                target_fs=sample_rate,
                volume=volume,
                speed=speed,
                audio_path=save_path)
        except:
            raise ServerBaseException(ErrorCode.SERVER_INTERNAL_ERR,
                                      "tts postprocess failed.")

        return lang, target_sample_rate, wav_base64
