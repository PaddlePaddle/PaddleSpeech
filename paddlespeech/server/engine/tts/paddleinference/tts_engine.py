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
import os
import sys
import time
from typing import Optional

import librosa
import numpy as np
import paddle
import soundfile as sf
from scipy.io import wavfile

from paddlespeech.cli.log import logger
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.resource import CommonTaskResource
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.audio_process import change_speed
from paddlespeech.server.utils.errors import ErrorCode
from paddlespeech.server.utils.exception import ServerBaseException
from paddlespeech.server.utils.paddle_predictor import init_predictor
from paddlespeech.server.utils.paddle_predictor import run_model
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend

__all__ = ['TTSEngine', 'PaddleTTSConnectionHandler']


class TTSServerExecutor(TTSExecutor):
    def __init__(self):
        super().__init__()
        self.task_resource = CommonTaskResource(
            task='tts', model_format='static')

    def _init_from_path(
            self,
            am: str='fastspeech2_csmsc',
            am_model: Optional[os.PathLike]=None,
            am_params: Optional[os.PathLike]=None,
            am_sample_rate: int=24000,
            phones_dict: Optional[os.PathLike]=None,
            tones_dict: Optional[os.PathLike]=None,
            speaker_dict: Optional[os.PathLike]=None,
            voc: str='pwgan_csmsc',
            voc_model: Optional[os.PathLike]=None,
            voc_params: Optional[os.PathLike]=None,
            voc_sample_rate: int=24000,
            lang: str='zh',
            am_predictor_conf: dict=None,
            voc_predictor_conf: dict=None, ):
        """
        Init model and other resources from a specific path.
        """
        if hasattr(self, 'am_predictor') and hasattr(self, 'voc_predictor'):
            logger.debug('Models had been initialized.')
            return
        # am
        if am_model is None or am_params is None or phones_dict is None:
            use_pretrained_am = True
        else:
            use_pretrained_am = False

        am_tag = am + '-' + lang
        self.task_resource.set_task_model(
            model_tag=am_tag,
            model_type=0,  # am
            skip_download=not use_pretrained_am,
            version=None,  # default version
        )
        if use_pretrained_am:
            self.am_res_path = self.task_resource.res_dir
            self.am_model = os.path.join(self.am_res_path,
                                         self.task_resource.res_dict['model'])
            self.am_params = os.path.join(self.am_res_path,
                                          self.task_resource.res_dict['params'])
            # must have phones_dict in acoustic
            self.phones_dict = os.path.join(
                self.am_res_path, self.task_resource.res_dict['phones_dict'])
            self.am_sample_rate = self.task_resource.res_dict['sample_rate']

            logger.debug(self.am_res_path)
            logger.debug(self.am_model)
            logger.debug(self.am_params)
        else:
            self.am_model = os.path.abspath(am_model)
            self.am_params = os.path.abspath(am_params)
            self.phones_dict = os.path.abspath(phones_dict)
            self.am_sample_rate = am_sample_rate
            self.am_res_path = os.path.dirname(os.path.abspath(self.am_model))
        logger.debug("self.phones_dict: {}".format(self.phones_dict))

        # for speedyspeech
        self.tones_dict = None
        if 'tones_dict' in self.task_resource.res_dict:
            self.tones_dict = os.path.join(
                self.am_res_path, self.task_resource.res_dict['tones_dict'])
            if tones_dict:
                self.tones_dict = tones_dict

        # for multi speaker fastspeech2
        self.speaker_dict = None
        if 'speaker_dict' in self.task_resource.res_dict:
            self.speaker_dict = os.path.join(
                self.am_res_path, self.task_resource.res_dict['speaker_dict'])
            if speaker_dict:
                self.speaker_dict = speaker_dict

        # voc
        if voc_model is None or voc_params is None:
            use_pretrained_voc = True
        else:
            use_pretrained_voc = False

        voc_tag = voc + '-' + lang
        self.task_resource.set_task_model(
            model_tag=voc_tag,
            model_type=1,  # vocoder
            skip_download=not use_pretrained_voc,
            version=None,  # default version
        )
        if use_pretrained_voc:
            self.voc_res_path = self.task_resource.voc_res_dir
            self.voc_model = os.path.join(
                self.voc_res_path, self.task_resource.voc_res_dict['model'])
            self.voc_params = os.path.join(
                self.voc_res_path, self.task_resource.voc_res_dict['params'])
            self.voc_sample_rate = self.task_resource.voc_res_dict[
                'sample_rate']
            logger.debug(self.voc_res_path)
            logger.debug(self.voc_model)
            logger.debug(self.voc_params)
        else:
            self.voc_model = os.path.abspath(voc_model)
            self.voc_params = os.path.abspath(voc_params)
            self.voc_sample_rate = voc_sample_rate
            self.voc_res_path = os.path.dirname(os.path.abspath(self.voc_model))

        assert (
            self.voc_sample_rate == self.am_sample_rate
        ), "The sample rate of AM and Vocoder model are different, please check model."

        # Init body.
        with open(self.phones_dict, "r") as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        vocab_size = len(phn_id)
        logger.debug("vocab_size: {}".format(vocab_size))

        tone_size = None
        if self.tones_dict:
            with open(self.tones_dict, "r") as f:
                tone_id = [line.strip().split() for line in f.readlines()]
            tone_size = len(tone_id)
            logger.debug("tone_size: {}".format(tone_size))

        spk_num = None
        if self.speaker_dict:
            with open(self.speaker_dict, 'rt') as f:
                spk_id = [line.strip().split() for line in f.readlines()]
            spk_num = len(spk_id)
            logger.debug("spk_num: {}".format(spk_num))

        # frontend
        if lang == 'zh':
            self.frontend = Frontend(
                phone_vocab_path=self.phones_dict,
                tone_vocab_path=self.tones_dict)

        elif lang == 'en':
            self.frontend = English(phone_vocab_path=self.phones_dict)
        logger.debug("frontend done!")

        # Create am predictor
        self.am_predictor_conf = am_predictor_conf
        self.am_predictor = init_predictor(
            model_file=self.am_model,
            params_file=self.am_params,
            predictor_conf=self.am_predictor_conf)
        logger.debug("Create AM predictor successfully.")

        # Create voc predictor
        self.voc_predictor_conf = voc_predictor_conf
        self.voc_predictor = init_predictor(
            model_file=self.voc_model,
            params_file=self.voc_params,
            predictor_conf=self.voc_predictor_conf)
        logger.debug("Create Vocoder predictor successfully.")

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
            logger.error("lang should in {'zh', 'en'}!")
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
            self.am_time += (time.time() - am_st)

            # voc
            voc_st = time.time()
            voc_result = run_model(self.voc_predictor, [mel])
            wav = voc_result[0]
            wav = paddle.to_tensor(wav)

            if flags == 0:
                wav_all = wav
                flags = 1
            else:
                wav_all = paddle.concat([wav_all, wav])
            self.voc_time += (time.time() - voc_st)
        self._outputs["wav"] = wav_all


class TTSEngine(BaseEngine):
    """TTS server engine

    Args:
        metaclass: Defaults to Singleton.
    """

    def __init__(self):
        """Initialize TTS server engine
        """
        super(TTSEngine, self).__init__()

    def init(self, config: dict) -> bool:
        self.executor = TTSServerExecutor()
        self.config = config
        self.lang = self.config.lang
        self.engine_type = "inference"

        try:
            if self.config.am_predictor_conf.device is not None:
                self.device = self.config.am_predictor_conf.device
            elif self.config.voc_predictor_conf.device is not None:
                self.device = self.config.voc_predictor_conf.device
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
                am_model=self.config.am_model,
                am_params=self.config.am_params,
                am_sample_rate=self.config.am_sample_rate,
                phones_dict=self.config.phones_dict,
                tones_dict=self.config.tones_dict,
                speaker_dict=self.config.speaker_dict,
                voc=self.config.voc,
                voc_model=self.config.voc_model,
                voc_params=self.config.voc_params,
                voc_sample_rate=self.config.voc_sample_rate,
                lang=self.config.lang,
                am_predictor_conf=self.config.am_predictor_conf,
                voc_predictor_conf=self.config.voc_predictor_conf, )
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
        self.am_predictor = self.executor.am_predictor
        self.voc_predictor = self.executor.voc_predictor

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
        # wavfile.write(buf, target_fs, wav_speed)
        sf.write(buf, wav_speed, target_fs, format="wav")
        buf.seek(0)
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
                original_fs=self.executor.am_sample_rate,
                target_fs=sample_rate,
                volume=volume,
                speed=speed,
                audio_path=save_path)
            postprocess_et = time.time()
            postprocess_time = postprocess_et - postprocess_st
            duration = len(
                self._outputs["wav"].numpy()) / self.executor.am_sample_rate
            rtf = infer_time / duration

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
