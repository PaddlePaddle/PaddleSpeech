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
import math
import os
import time
from typing import Optional

import numpy as np
import paddle
import yaml
from yacs.config import CfgNode

from paddlespeech.cli.log import logger
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.resource import CommonTaskResource
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.audio_process import float2pcm
from paddlespeech.server.utils.util import denorm
from paddlespeech.server.utils.util import get_chunks
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.modules.normalizer import ZScore

__all__ = ['TTSEngine', 'PaddleTTSConnectionHandler']


class TTSServerExecutor(TTSExecutor):
    def __init__(self):
        super().__init__()
        self.task_resource = CommonTaskResource(
            task='tts', model_format='dynamic', inference_mode='online')

    def get_model_info(self,
                       field: str,
                       model_name: str,
                       ckpt: Optional[os.PathLike],
                       stat: Optional[os.PathLike]):
        """get model information

        Args:
            field (str): am or voc
            model_name (str): model type, support fastspeech2, higigan, mb_melgan
            ckpt (Optional[os.PathLike]): ckpt file
            stat (Optional[os.PathLike]): stat file, including mean and standard deviation

        Returns:
            [module]: model module
            [Tensor]: mean
            [Tensor]: standard deviation
        """

        model_class = self.task_resource.get_model_class(model_name)

        if field == "am":
            odim = self.am_config.n_mels
            model = model_class(
                idim=self.vocab_size, odim=odim, **self.am_config["model"])
            model.set_state_dict(paddle.load(ckpt)["main_params"])

        elif field == "voc":
            model = model_class(**self.voc_config["generator_params"])
            model.set_state_dict(paddle.load(ckpt)["generator_params"])
            model.remove_weight_norm()

        else:
            logger.error("Please set correct field, am or voc")

        model.eval()
        model_mu, model_std = np.load(stat)
        model_mu = paddle.to_tensor(model_mu)
        model_std = paddle.to_tensor(model_std)

        return model, model_mu, model_std

    def _init_from_path(
            self,
            am: str='fastspeech2_csmsc',
            am_config: Optional[os.PathLike]=None,
            am_ckpt: Optional[os.PathLike]=None,
            am_stat: Optional[os.PathLike]=None,
            phones_dict: Optional[os.PathLike]=None,
            tones_dict: Optional[os.PathLike]=None,
            speaker_dict: Optional[os.PathLike]=None,
            voc: str='mb_melgan_csmsc',
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
        # am model info
        if am_ckpt is None or am_config is None or am_stat is None or phones_dict is None:
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
            self.am_config = os.path.join(self.am_res_path,
                                          self.task_resource.res_dict['config'])
            self.am_ckpt = os.path.join(self.am_res_path,
                                        self.task_resource.res_dict['ckpt'])
            self.am_stat = os.path.join(
                self.am_res_path, self.task_resource.res_dict['speech_stats'])
            # must have phones_dict in acoustic
            self.phones_dict = os.path.join(
                self.am_res_path, self.task_resource.res_dict['phones_dict'])
            print("self.phones_dict:", self.phones_dict)
            logger.info(self.am_res_path)
            logger.info(self.am_config)
            logger.info(self.am_ckpt)
        else:
            self.am_config = os.path.abspath(am_config)
            self.am_ckpt = os.path.abspath(am_ckpt)
            self.am_stat = os.path.abspath(am_stat)
            self.phones_dict = os.path.abspath(phones_dict)
            self.am_res_path = os.path.dirname(os.path.abspath(self.am_config))
        print("self.phones_dict:", self.phones_dict)

        self.tones_dict = None
        self.speaker_dict = None

        # voc model info
        if voc_ckpt is None or voc_config is None or voc_stat is None:
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
            self.voc_config = os.path.join(
                self.voc_res_path, self.task_resource.voc_res_dict['config'])
            self.voc_ckpt = os.path.join(
                self.voc_res_path, self.task_resource.voc_res_dict['ckpt'])
            self.voc_stat = os.path.join(
                self.voc_res_path,
                self.task_resource.voc_res_dict['speech_stats'])
            logger.info(self.voc_res_path)
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
        self.vocab_size = len(phn_id)
        print("vocab_size:", self.vocab_size)

        # frontend
        if lang == 'zh':
            self.frontend = Frontend(
                phone_vocab_path=self.phones_dict,
                tone_vocab_path=self.tones_dict)

        elif lang == 'en':
            self.frontend = English(phone_vocab_path=self.phones_dict)
        print("frontend done!")

        # am infer info
        self.am_name = am[:am.rindex('_')]
        if self.am_name == "fastspeech2_cnndecoder":
            self.am_inference, self.am_mu, self.am_std = self.get_model_info(
                "am", "fastspeech2", self.am_ckpt, self.am_stat)
        else:
            am, am_mu, am_std = self.get_model_info("am", self.am_name,
                                                    self.am_ckpt, self.am_stat)
            am_normalizer = ZScore(am_mu, am_std)
            am_inference_class = self.task_resource.get_model_class(
                self.am_name + '_inference')
            self.am_inference = am_inference_class(am_normalizer, am)
            self.am_inference.eval()
        print("acoustic model done!")

        # voc infer info
        self.voc_name = voc[:voc.rindex('_')]
        voc, voc_mu, voc_std = self.get_model_info("voc", self.voc_name,
                                                   self.voc_ckpt, self.voc_stat)
        voc_normalizer = ZScore(voc_mu, voc_std)
        voc_inference_class = self.task_resource.get_model_class(self.voc_name +
                                                                 '_inference')
        self.voc_inference = voc_inference_class(voc_normalizer, voc)
        self.voc_inference.eval()
        print("voc done!")


class TTSEngine(BaseEngine):
    """TTS server engine

    Args:
        metaclass: Defaults to Singleton.
    """

    def __init__(self, name=None):
        """Initialize TTS server engine
        """
        super().__init__()

    def init(self, config: dict) -> bool:
        self.executor = TTSServerExecutor()
        self.config = config
        self.lang = self.config.lang
        self.engine_type = "online"

        assert (
            config.am == "fastspeech2_csmsc" or
            config.am == "fastspeech2_cnndecoder_csmsc"
        ) and (
            config.voc == "hifigan_csmsc" or config.voc == "mb_melgan_csmsc"
        ), 'Please check config, am support: fastspeech2, voc support: hifigan_csmsc-zh or mb_melgan_csmsc.'

        assert (
            config.voc_block > 0 and config.voc_pad > 0
        ), "Please set correct voc_block and voc_pad, they should be more than 0."

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

        self.am_block = self.config.am_block
        self.am_pad = self.config.am_pad
        self.voc_block = self.config.voc_block
        self.voc_pad = self.config.voc_pad
        self.am_upsample = 1
        self.voc_upsample = self.executor.voc_config.n_shift

        logger.info("Initialize TTS server engine successfully on device: %s." %
                    (self.device))

        return True


class PaddleTTSConnectionHandler:
    def __init__(self, tts_engine):
        """The PaddleSpeech TTS Server Connection Handler
           This connection process every tts server request
        Args:
            tts_engine (TTSEngine): The TTS engine
        """
        super().__init__()
        logger.info(
            "Create PaddleTTSConnectionHandler to process the tts request")

        self.tts_engine = tts_engine
        self.executor = self.tts_engine.executor
        self.config = self.tts_engine.config
        self.am_block = self.tts_engine.am_block
        self.am_pad = self.tts_engine.am_pad
        self.voc_block = self.tts_engine.voc_block
        self.voc_pad = self.tts_engine.voc_pad
        self.am_upsample = self.tts_engine.am_upsample
        self.voc_upsample = self.tts_engine.voc_upsample

    def depadding(self, data, chunk_num, chunk_id, block, pad, upsample):
        """ 
        Streaming inference removes the result of pad inference
        """
        front_pad = min(chunk_id * block, pad)
        # first chunk
        if chunk_id == 0:
            data = data[:block * upsample]
        # last chunk
        elif chunk_id == chunk_num - 1:
            data = data[front_pad * upsample:]
        # middle chunk
        else:
            data = data[front_pad * upsample:(front_pad + block) * upsample]

        return data

    @paddle.no_grad()
    def infer(
            self,
            text: str,
            lang: str='zh',
            am: str='fastspeech2_csmsc',
            spk_id: int=0, ):
        """
        Model inference and result stored in self.output.
        """

        # first_flag 用于标记首包
        first_flag = 1

        get_tone_ids = False
        merge_sentences = False
        frontend_st = time.time()
        if lang == 'zh':
            input_ids = self.executor.frontend.get_input_ids(
                text,
                merge_sentences=merge_sentences,
                get_tone_ids=get_tone_ids)
            phone_ids = input_ids["phone_ids"]
            if get_tone_ids:
                tone_ids = input_ids["tone_ids"]
        elif lang == 'en':
            input_ids = self.executor.frontend.get_input_ids(
                text, merge_sentences=merge_sentences)
            phone_ids = input_ids["phone_ids"]
        else:
            print("lang should in {'zh', 'en'}!")
        frontend_et = time.time()
        self.frontend_time = frontend_et - frontend_st

        for i in range(len(phone_ids)):
            part_phone_ids = phone_ids[i]
            voc_chunk_id = 0

            # fastspeech2_csmsc
            if am == "fastspeech2_csmsc":
                # am 
                mel = self.executor.am_inference(part_phone_ids)
                if first_flag == 1:
                    first_am_et = time.time()
                    self.first_am_infer = first_am_et - frontend_et

                # voc streaming
                mel_chunks = get_chunks(mel, self.voc_block, self.voc_pad,
                                        "voc")
                voc_chunk_num = len(mel_chunks)
                voc_st = time.time()
                for i, mel_chunk in enumerate(mel_chunks):
                    sub_wav = self.executor.voc_inference(mel_chunk)
                    sub_wav = self.depadding(sub_wav, voc_chunk_num, i,
                                             self.voc_block, self.voc_pad,
                                             self.voc_upsample)
                    if first_flag == 1:
                        first_voc_et = time.time()
                        self.first_voc_infer = first_voc_et - first_am_et
                        self.first_response_time = first_voc_et - frontend_st
                        first_flag = 0

                    yield sub_wav

            # fastspeech2_cnndecoder_csmsc 
            elif am == "fastspeech2_cnndecoder_csmsc":
                # am 
                orig_hs = self.executor.am_inference.encoder_infer(
                    part_phone_ids)

                # streaming voc chunk info
                mel_len = orig_hs.shape[1]
                voc_chunk_num = math.ceil(mel_len / self.voc_block)
                start = 0
                end = min(self.voc_block + self.voc_pad, mel_len)

                # streaming am
                hss = get_chunks(orig_hs, self.am_block, self.am_pad, "am")
                am_chunk_num = len(hss)
                for i, hs in enumerate(hss):
                    before_outs = self.executor.am_inference.decoder(hs)
                    after_outs = before_outs + self.executor.am_inference.postnet(
                        before_outs.transpose((0, 2, 1))).transpose((0, 2, 1))
                    normalized_mel = after_outs[0]
                    sub_mel = denorm(normalized_mel, self.executor.am_mu,
                                     self.executor.am_std)
                    sub_mel = self.depadding(sub_mel, am_chunk_num, i,
                                             self.am_block, self.am_pad,
                                             self.am_upsample)

                    if i == 0:
                        mel_streaming = sub_mel
                    else:
                        mel_streaming = np.concatenate(
                            (mel_streaming, sub_mel), axis=0)

                    # streaming voc
                    # 当流式AM推理的mel帧数大于流式voc推理的chunk size，开始进行流式voc 推理
                    while (mel_streaming.shape[0] >= end and
                           voc_chunk_id < voc_chunk_num):
                        if first_flag == 1:
                            first_am_et = time.time()
                            self.first_am_infer = first_am_et - frontend_et
                        voc_chunk = mel_streaming[start:end, :]
                        voc_chunk = paddle.to_tensor(voc_chunk)
                        sub_wav = self.executor.voc_inference(voc_chunk)

                        sub_wav = self.depadding(
                            sub_wav, voc_chunk_num, voc_chunk_id,
                            self.voc_block, self.voc_pad, self.voc_upsample)
                        if first_flag == 1:
                            first_voc_et = time.time()
                            self.first_voc_infer = first_voc_et - first_am_et
                            self.first_response_time = first_voc_et - frontend_st
                            first_flag = 0

                        yield sub_wav

                        voc_chunk_id += 1
                        start = max(
                            0, voc_chunk_id * self.voc_block - self.voc_pad)
                        end = min(
                            (voc_chunk_id + 1) * self.voc_block + self.voc_pad,
                            mel_len)

            else:
                logger.error(
                    "Only support fastspeech2_csmsc or fastspeech2_cnndecoder_csmsc on streaming tts."
                )

        self.final_response_time = time.time() - frontend_st

    def preprocess(self, text_bese64: str=None, text_bytes: bytes=None):
        # Convert byte to text
        if text_bese64:
            text_bytes = base64.b64decode(text_bese64)  # base64 to bytes
        text = text_bytes.decode('utf-8')  # bytes to text

        return text

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

        Returns:
            wav_base64: The base64 format of the synthesized audio.
        """

        wav_list = []

        for wav in self.infer(
                text=sentence,
                lang=self.config.lang,
                am=self.config.am,
                spk_id=spk_id, ):

            # wav type: <class 'numpy.ndarray'>  float32, convert to pcm (base64)
            wav = float2pcm(wav)  # float32 to int16
            wav_bytes = wav.tobytes()  # to bytes
            wav_base64 = base64.b64encode(wav_bytes).decode('utf8')  # to base64
            wav_list.append(wav)

            yield wav_base64

        wav_all = np.concatenate(wav_list, axis=0)
        duration = len(wav_all) / self.executor.am_config.fs

        logger.info(f"sentence: {sentence}")
        logger.info(f"The durations of audio is: {duration} s")
        logger.info(f"first response time: {self.first_response_time} s")
        logger.info(f"final response time: {self.final_response_time} s")
        logger.info(f"RTF: {self.final_response_time / duration}")
        logger.info(
            f"Other info: front time: {self.frontend_time} s, first am infer time: {self.first_am_infer} s, first voc infer time: {self.first_voc_infer} s,"
        )
