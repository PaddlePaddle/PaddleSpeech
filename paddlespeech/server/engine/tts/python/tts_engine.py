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
import base64
import io
import sys
import time

import librosa
import numpy as np
import paddle
import soundfile as sf
from scipy.io import wavfile

from paddlespeech.cli.log import logger
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.audio_process import change_speed
from paddlespeech.server.utils.errors import ErrorCode
from paddlespeech.server.utils.exception import ServerBaseException

__all__ = ['TTSEngine', 'PaddleTTSConnectionHandler']


class TTSServerExecutor(TTSExecutor):
    def __init__(self):
        super().__init__()
        pass


class TTSEngine(BaseEngine):
    """TTS server engine

    Args:
        metaclass: Defaults to Singleton.
    """

    def __init__(self, name=None):
        """Initialize TTS server engine
        """
        super(TTSEngine, self).__init__()

    def init(self, config: dict) -> bool:
        self.executor = TTSServerExecutor()
        self.config = config
        self.lang = self.config.lang
        self.engine_type = "python"

        try:
            if self.config.device is not None:
                self.device = self.config.device
            else:
                self.device = paddle.get_device()
            paddle.set_device(self.device)
        except Exception as e:
            logger.error(
                "Set device failed, please check if device is already used and the parameter 'device' in the yaml file"
            )
            logger.error("Initialize TTS server engine Failed on device: %s." %
                         (self.device))
            logger.error(e)
            return False

        try:
            self.executor._init_from_path(
                am=self.config.am,
                am_config=self.config.am_config,
                am_ckpt=self.config.am_ckpt,
                am_stat=self.config.am_stat,
                phones_dict=self.config.phones_dict,
                tones_dict=self.config.tones_dict,
                speaker_dict=self.config.speaker_dict,
                voc=self.config.voc,
                voc_config=self.config.voc_config,
                voc_ckpt=self.config.voc_ckpt,
                voc_stat=self.config.voc_stat,
                lang=self.config.lang)
        except Exception as e:
            logger.error("Failed to get model related files.")
            logger.error("Initialize TTS server engine Failed on device: %s." %
                         (self.device))
            logger.error(e)
            return False

        logger.info("Initialize TTS server engine successfully on device: %s." %
                    (self.device))
        return True


class PaddleTTSConnectionHandler(TTSServerExecutor):
    def __init__(self, tts_engine):
        """The PaddleSpeech TTS Server Connection Handler
           This connection process every tts server request
        Args:
            tts_engine (TTSEngine): The TTS engine
        """
        super().__init__()
        logger.debug(
            "Create PaddleTTSConnectionHandler to process the tts request")

        self.tts_engine = tts_engine
        self.executor = self.tts_engine.executor
        self.config = self.tts_engine.config
        self.frontend = self.executor.frontend
        self.am_inference = self.executor.am_inference
        self.voc_inference = self.executor.voc_inference

    def postprocess(self,
                    wav,
                    original_fs: int,
                    target_fs: int=0,
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

        Raises:
            ServerBaseException: Throws an exception if the change speed unsuccessfully.

        Returns:
            target_fs: target sample rate for synthesized audio.
            wav_base64: The base64 format of the synthesized audio.
        """

        # transform sample_rate
        if target_fs == 0 or target_fs > original_fs:
            target_fs = original_fs
            wav_tar_fs = wav
            logger.debug(
                "The sample rate of synthesized audio is the same as model, which is {}Hz".
                format(original_fs))
        else:
            wav_tar_fs = librosa.resample(
                np.squeeze(wav), original_fs, target_fs)
            logger.debug(
                "The sample rate of model is {}Hz and the target sample rate is {}Hz. Converting the sample rate of the synthesized audio successfully.".
                format(original_fs, target_fs))
        # transform volume
        wav_vol = wav_tar_fs * volume
        logger.debug("Transform the volume of the audio successfully.")

        # transform speed
        try:  # windows not support soxbindings
            wav_speed = change_speed(wav_vol, speed, target_fs)
            logger.debug("Transform the speed of the audio successfully.")
        except ServerBaseException:
            raise ServerBaseException(
                ErrorCode.SERVER_INTERNAL_ERR,
                "Failed to transform speed. Can not install soxbindings on your system. \
                 You need to set speed value 1.0.")
            sys.exit(-1)
        except Exception as e:
            logger.error("Failed to transform speed.")
            logger.error(e)
            sys.exit(-1)

        # wav to base64
        buf = io.BytesIO()
        wavfile.write(buf, target_fs, wav_speed)
        base64_bytes = base64.b64encode(buf.read())
        wav_base64 = base64_bytes.decode('utf-8')
        logger.debug("Audio to string successfully.")

        # save audio
        if audio_path is not None:
            if audio_path.endswith(".wav"):
                sf.write(audio_path, wav_speed, target_fs)
            elif audio_path.endswith(".pcm"):
                wav_norm = wav_speed * (32767 / max(0.001,
                                                    np.max(np.abs(wav_speed))))
                with open(audio_path, "wb") as f:
                    f.write(wav_norm.astype(np.int16))
            logger.info("Save audio to {} successfully.".format(audio_path))
        else:
            logger.info("There is no need to save audio.")

        return target_fs, wav_base64

    def run(self,
            sentence: str,
            spk_id: int=0,
            speed: float=1.0,
            volume: float=1.0,
            sample_rate: int=0,
            save_path: str=None):
        """ run include inference and postprocess.

        Args:
            sentence (str): text to be synthesized
            spk_id (int, optional): speaker id for multi-speaker speech synthesis. Defaults to 0.
            speed (float, optional): speed. Defaults to 1.0.
            volume (float, optional): volume. Defaults to 1.0.
            sample_rate (int, optional): target sample rate for synthesized audio, 
            0 means the same as the model sampling rate. Defaults to 0.
            save_path (str, optional): The save path of the synthesized audio. 
            None means do not save audio. Defaults to None.

        Raises:
            ServerBaseException: Throws an exception if tts inference unsuccessfully.
            ServerBaseException: Throws an exception if postprocess unsuccessfully.

        Returns:
            lang: model language 
            target_sample_rate: target sample rate for synthesized audio.
            wav_base64: The base64 format of the synthesized audio.
        """

        lang = self.config.lang

        try:
            infer_st = time.time()
            self.infer(
                text=sentence, lang=lang, am=self.config.am, spk_id=spk_id)
            infer_et = time.time()
            infer_time = infer_et - infer_st
            duration = len(
                self._outputs["wav"].numpy()) / self.executor.am_config.fs
            rtf = infer_time / duration

        except ServerBaseException:
            raise ServerBaseException(ErrorCode.SERVER_INTERNAL_ERR,
                                      "tts infer failed.")
            sys.exit(-1)
        except Exception as e:
            logger.error("tts infer failed.")
            logger.error(e)
            sys.exit(-1)

        try:
            postprocess_st = time.time()
            target_sample_rate, wav_base64 = self.postprocess(
                wav=self._outputs["wav"].numpy(),
                original_fs=self.executor.am_config.fs,
                target_fs=sample_rate,
                volume=volume,
                speed=speed,
                audio_path=save_path)
            postprocess_et = time.time()
            postprocess_time = postprocess_et - postprocess_st

        except ServerBaseException:
            raise ServerBaseException(ErrorCode.SERVER_INTERNAL_ERR,
                                      "tts postprocess failed.")
            sys.exit(-1)
        except Exception as e:
            logger.error("tts postprocess failed.")
            logger.error(e)
            sys.exit(-1)

        logger.debug("AM model: {}".format(self.config.am))
        logger.debug("Vocoder model: {}".format(self.config.voc))
        logger.debug("Language: {}".format(lang))
        logger.info("tts engine type: python")

        logger.info("audio duration: {}".format(duration))
        logger.debug("frontend inference time: {}".format(self.frontend_time))
        logger.debug("AM inference time: {}".format(self.am_time))
        logger.debug("Vocoder inference time: {}".format(self.voc_time))
        logger.info("total inference time: {}".format(infer_time))
        logger.info(
            "postprocess (change speed, volume, target sample rate) time: {}".
            format(postprocess_time))
        logger.info("total generate audio time: {}".format(infer_time +
                                                           postprocess_time))
        logger.info("RTF: {}".format(rtf))
        logger.debug("device: {}".format(self.tts_engine.device))

        return lang, target_sample_rate, duration, wav_base64
