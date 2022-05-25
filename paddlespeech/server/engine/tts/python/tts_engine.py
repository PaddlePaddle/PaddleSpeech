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
import time
from typing import Optional

import librosa
import numpy as np
import paddle
import soundfile as sf
import yaml
from scipy.io import wavfile
from yacs.config import CfgNode

from .pretrained_models import model_alias
from .pretrained_models import pretrained_models
from paddlespeech.cli.log import logger
from paddlespeech.cli.utils import download_and_decompress
from paddlespeech.cli.utils import MODEL_HOME
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.audio_process import change_speed
from paddlespeech.server.utils.errors import ErrorCode
from paddlespeech.server.utils.exception import ServerBaseException
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.modules.normalizer import ZScore

__all__ = ['TTSEngine', 'TTSHandler']


class TTSEngine(BaseEngine):
    """TTS server engine for model setting

    Args:
        metaclass: Defaults to Singleton.
    """

    def __init__(self, name=None):
        """Initialize TTS server engine
        """
        super(TTSEngine, self).__init__()
        self.model_alias = model_alias
        self.pretrained_models = pretrained_models
        self.engine_type = "python"

    def init(self, config: dict) -> bool:
        self.config = config
        self.lang = self.config.lang

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
            self._init_from_path(
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
            logger.info(
                "Initialize TTS server engine successfully on device: %s." %
                (self.device))
        except Exception as e:
            logger.error("Failed to get model related files.")
            logger.error("Initialize TTS server engine Failed on device: %s." %
                         (self.device))
            logger.error(e)
            return False

        return True

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
            self.am_config = os.path.join(
                am_res_path, self.pretrained_models[am_tag]['config'])
            self.am_ckpt = os.path.join(am_res_path,
                                        self.pretrained_models[am_tag]['ckpt'])
            self.am_stat = os.path.join(
                am_res_path, self.pretrained_models[am_tag]['speech_stats'])
            # must have phones_dict in acoustic
            self.phones_dict = os.path.join(
                am_res_path, self.pretrained_models[am_tag]['phones_dict'])
            logger.info(am_res_path)
            logger.info(self.am_config)
            logger.info(self.am_ckpt)
        else:
            self.am_config = os.path.abspath(am_config)
            self.am_ckpt = os.path.abspath(am_ckpt)
            self.am_stat = os.path.abspath(am_stat)
            self.phones_dict = os.path.abspath(phones_dict)
            self.am_res_path = os.path.dirname(os.path.abspath(self.am_config))

        # for speedyspeech
        self.tones_dict = None
        if 'tones_dict' in self.pretrained_models[am_tag]:
            self.tones_dict = os.path.join(
                am_res_path, self.pretrained_models[am_tag]['tones_dict'])
            if tones_dict:
                self.tones_dict = tones_dict

        # for multi speaker fastspeech2
        self.speaker_dict = None
        if 'speaker_dict' in self.pretrained_models[am_tag]:
            self.speaker_dict = os.path.join(
                am_res_path, self.pretrained_models[am_tag]['speaker_dict'])
            if speaker_dict:
                self.speaker_dict = speaker_dict

        # voc
        voc_tag = voc + '-' + lang
        if voc_ckpt is None or voc_config is None or voc_stat is None:
            voc_res_path = self._get_pretrained_path(voc_tag)
            self.voc_res_path = voc_res_path
            self.voc_config = os.path.join(
                voc_res_path, self.pretrained_models[voc_tag]['config'])
            self.voc_ckpt = os.path.join(
                voc_res_path, self.pretrained_models[voc_tag]['ckpt'])
            self.voc_stat = os.path.join(
                voc_res_path, self.pretrained_models[voc_tag]['speech_stats'])
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
        logger.info(f"vocab_size: {vocab_size}")

        tone_size = None
        if self.tones_dict:
            with open(self.tones_dict, "r") as f:
                tone_id = [line.strip().split() for line in f.readlines()]
            tone_size = len(tone_id)
            logger.info(f"tone_size: {tone_size}")

        spk_num = None
        if self.speaker_dict:
            with open(self.speaker_dict, 'rt') as f:
                spk_id = [line.strip().split() for line in f.readlines()]
            spk_num = len(spk_id)
            logger.info(f"spk_num: {spk_num}")

        # frontend
        if lang == 'zh':
            self.frontend = Frontend(
                phone_vocab_path=self.phones_dict,
                tone_vocab_path=self.tones_dict)

        elif lang == 'en':
            self.frontend = English(phone_vocab_path=self.phones_dict)
        logger.info("frontend done!")

        # acoustic model
        odim = self.am_config.n_mels
        # model: {model_name}_{dataset}
        am_name = am[:am.rindex('_')]

        am_class = dynamic_import(am_name, self.model_alias)
        am_inference_class = dynamic_import(am_name + '_inference',
                                            self.model_alias)

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
        logger.info("acoustic model done!")

        # vocoder
        # model: {model_name}_{dataset}
        voc_name = voc[:voc.rindex('_')]
        voc_class = dynamic_import(voc_name, self.model_alias)
        voc_inference_class = dynamic_import(voc_name + '_inference',
                                             self.model_alias)
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
        logger.info("voc done!")

    def _get_pretrained_path(self, tag: str) -> os.PathLike:
        """
        Download and returns pretrained resources path of current task.
        """
        support_models = list(self.pretrained_models.keys())
        assert tag in self.pretrained_models, 'The model "{}" you want to use has not been supported, please choose other models.\nThe support models includes:\n\t\t{}\n'.format(
            tag, '\n\t\t'.join(support_models))

        res_path = os.path.join(MODEL_HOME, tag)
        decompressed_path = download_and_decompress(self.pretrained_models[tag],
                                                    res_path)
        decompressed_path = os.path.abspath(decompressed_path)
        logger.info(
            'Use pretrained model stored in: {}'.format(decompressed_path))

        return decompressed_path


class TTSHandler():
    def __init__(self, tts_engine):
        """Initialize TTS server engine
        """
        super(TTSHandler, self).__init__()
        self.tts_engine = tts_engine

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
            input_ids = self.tts_engine.frontend.get_input_ids(
                text,
                merge_sentences=merge_sentences,
                get_tone_ids=get_tone_ids)
            phone_ids = input_ids["phone_ids"]
            if get_tone_ids:
                tone_ids = input_ids["tone_ids"]
        elif lang == 'en':
            input_ids = self.tts_engine.frontend.get_input_ids(
                text, merge_sentences=merge_sentences)
            phone_ids = input_ids["phone_ids"]
        else:
            logger.info("lang should in {'zh', 'en'}!")
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
                mel = self.tts_engine.am_inference(part_phone_ids,
                                                   part_tone_ids)
            # fastspeech2
            else:
                # multi speaker
                if am_dataset in {"aishell3", "vctk"}:
                    mel = self.tts_engine.am_inference(
                        part_phone_ids, spk_id=paddle.to_tensor(spk_id))
                else:
                    mel = self.tts_engine.am_inference(part_phone_ids)
            self.am_time += (time.time() - am_st)
            # voc
            voc_st = time.time()
            wav = self.tts_engine.voc_inference(mel)
            if flags == 0:
                wav_all = wav
                flags = 1
            else:
                wav_all = paddle.concat([wav_all, wav])
            self.voc_time += (time.time() - voc_st)
        self.output = wav_all

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
            logger.info(
                "The sample rate of synthesized audio is the same as model, which is {}Hz".
                format(original_fs))
        else:
            wav_tar_fs = librosa.resample(
                np.squeeze(wav), original_fs, target_fs)
            logger.info(
                "The sample rate of model is {}Hz and the target sample rate is {}Hz. Converting the sample rate of the synthesized audio successfully.".
                format(original_fs, target_fs))
        # transform volume
        wav_vol = wav_tar_fs * volume
        logger.info("Transform the volume of the audio successfully.")

        # transform speed
        try:  # windows not support soxbindings
            wav_speed = change_speed(wav_vol, speed, target_fs)
            logger.info("Transform the speed of the audio successfully.")
        except ServerBaseException:
            raise ServerBaseException(
                ErrorCode.SERVER_INTERNAL_ERR,
                "Failed to transform speed. Can not install soxbindings on your system. \
                 You need to set speed value 1.0.")
        except Exception as e:
            logger.error("Failed to transform speed.")
            logger.error(e)

        # wav to base64
        buf = io.BytesIO()
        wavfile.write(buf, target_fs, wav_speed)
        base64_bytes = base64.b64encode(buf.read())
        wav_base64 = base64_bytes.decode('utf-8')
        logger.info("Audio to string successfully.")

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

        lang = self.tts_engine.config.lang

        try:
            infer_st = time.time()
            self.infer(
                text=sentence,
                lang=self.tts_engine.config.lang,
                am=self.tts_engine.config.am,
                spk_id=spk_id)
            infer_et = time.time()
            infer_time = infer_et - infer_st
            duration = len(self.output.numpy()) / self.tts_engine.am_config.fs
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
                wav=self.output.numpy(),
                original_fs=self.tts_engine.am_config.fs,
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

        logger.info("AM model: {}".format(self.tts_engine.config.am))
        logger.info("Vocoder model: {}".format(self.tts_engine.config.voc))
        logger.info("Language: {}".format(lang))
        logger.info("tts engine type: python")

        logger.info("audio duration: {}".format(duration))
        logger.info("frontend inference time: {}".format(self.frontend_time))
        logger.info("AM inference time: {}".format(self.am_time))
        logger.info("Vocoder inference time: {}".format(self.voc_time))
        logger.info("total inference time: {}".format(infer_time))
        logger.info(
            "postprocess (change speed, volume, target sample rate) time: {}".
            format(postprocess_time))
        logger.info("total generate audio time: {}".format(infer_time +
                                                           postprocess_time))
        logger.info("RTF: {}".format(rtf))
        logger.info("device: {}".format(self.tts_engine.device))

        return lang, target_sample_rate, duration, wav_base64
