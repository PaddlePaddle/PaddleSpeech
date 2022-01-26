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

import librosa
import numpy as np
import soundfile as sf
import yaml
from engine.base_engine import BaseEngine

from paddlespeech.cli.log import logger
from paddlespeech.cli.tts.infer import TTSExecutor
from utils.errors import ErrorCode
from utils.exception import ServerBaseException

__all__ = ['TTSEngine']


class TTSServerExecutor(TTSExecutor):
    def __init__(self):
        super().__init__()

        self.parser = argparse.ArgumentParser(
            prog='paddlespeech.tts', add_help=True)
        self.parser.add_argument(
            '--conf',
            type=str,
            default='./conf/tts/tts.yaml',
            help='Configuration parameters.')


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
            am_config=self.conf_dict["am_config"],
            am_ckpt=self.conf_dict["am_ckpt"],
            am_stat=self.conf_dict["am_stat"],
            phones_dict=self.conf_dict["phones_dict"],
            tones_dict=self.conf_dict["tones_dict"],
            speaker_dict=self.conf_dict["speaker_dict"],
            voc=self.conf_dict["voc"],
            voc_config=self.conf_dict["voc_config"],
            voc_ckpt=self.conf_dict["voc_ckpt"],
            voc_stat=self.conf_dict["voc_stat"],
            lang=self.conf_dict["lang"])

        logger.info("Initialize TTS server engine successfully.")

    def postprocess(self,
                    wav,
                    original_fs: int,
                    target_fs: int=16000,
                    volume: float=1.0,
                    speed: float=1.0,
                    audio_path: str=None,
                    audio_format: str="wav"):
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
        # TODO
        target_wav = wav_vol.reshape(-1, 1)

        # save audio
        if audio_path is not None:
            sf.write(audio_path, target_wav, target_fs)
            logger.info('Wave file has been generated: {}'.format(audio_path))

        # wav to base64
        base64_bytes = base64.b64encode(target_wav)
        base64_string = base64_bytes.decode('utf-8')
        wav_base64 = base64_string

        return target_fs, wav_base64

    def run(self,
            sentence: str,
            spk_id: int=0,
            speed: float=1.0,
            volume: float=1.0,
            sample_rate: int=0,
            save_path: str=None,
            audio_format: str="wav"):

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
                original_fs=self.executor.am_config.fs,
                target_fs=sample_rate,
                volume=volume,
                speed=speed,
                audio_path=save_path,
                audio_format=audio_format)
        except:
            raise ServerBaseException(ErrorCode.SERVER_INTERNAL_ERR,
                                      "tts postprocess failed.")

        return lang, target_sample_rate, wav_base64
