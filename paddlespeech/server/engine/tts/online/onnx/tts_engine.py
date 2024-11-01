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

from paddlespeech.cli.log import logger
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.resource import CommonTaskResource
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.audio_process import float2pcm
from paddlespeech.server.utils.onnx_infer import get_sess
from paddlespeech.server.utils.util import denorm
from paddlespeech.server.utils.util import get_chunks
from paddlespeech.t2s.frontend.en_frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend

__all__ = ['TTSEngine', 'PaddleTTSConnectionHandler']


class TTSServerExecutor(TTSExecutor):
    def __init__(self):
        super().__init__()
        self.task_resource = CommonTaskResource(task='tts', model_format='onnx')

    def _init_from_path(
            self,
            am: str='fastspeech2_csmsc_onnx',
            am_ckpt: Optional[list]=None,
            am_stat: Optional[os.PathLike]=None,
            phones_dict: Optional[os.PathLike]=None,
            tones_dict: Optional[os.PathLike]=None,
            speaker_dict: Optional[os.PathLike]=None,
            am_sample_rate: int=24000,
            am_sess_conf: dict=None,
            voc: str='mb_melgan_csmsc_onnx',
            voc_ckpt: Optional[os.PathLike]=None,
            voc_sample_rate: int=24000,
            voc_sess_conf: dict=None,
            lang: str='zh', ):
        """
        Init model and other resources from a specific path.
        """

        if (hasattr(self, 'am_sess') or
            (hasattr(self, 'am_encoder_infer_sess') and
             hasattr(self, 'am_decoder_sess') and hasattr(
                 self, 'am_postnet_sess'))) and hasattr(self, 'voc_inference'):
            logger.debug('Models had been initialized.')
            return

        # am
        am_tag = am + '-' + lang
        if am == "fastspeech2_csmsc_onnx":
            # get model info
            if am_ckpt is None or phones_dict is None:
                self.task_resource.set_task_model(
                    model_tag=am_tag,
                    model_type=0,  # am
                    version=None,  # default version
                )
                self.am_res_path = self.task_resource.res_dir
                self.am_ckpt = os.path.join(self.am_res_path,
                                            self.task_resource.res_dict['ckpt'])
                # must have phones_dict in acoustic
                self.phones_dict = os.path.join(
                    self.am_res_path,
                    self.task_resource.res_dict['phones_dict'])

            else:
                self.am_ckpt = os.path.abspath(am_ckpt[0])
                self.phones_dict = os.path.abspath(phones_dict)
                self.am_res_path = os.path.dirname(os.path.abspath(am_ckpt))

            # create am sess
            self.am_sess = get_sess(self.am_ckpt, am_sess_conf)

        elif am == "fastspeech2_cnndecoder_csmsc_onnx":
            if am_ckpt is None or am_stat is None or phones_dict is None:
                self.task_resource.set_task_model(
                    model_tag=am_tag,
                    model_type=0,  # am
                    version=None,  # default version
                )
                self.am_res_path = self.task_resource.res_dir
                self.am_encoder_infer = os.path.join(
                    self.am_res_path, self.task_resource.res_dict['ckpt'][0])
                self.am_decoder = os.path.join(
                    self.am_res_path, self.task_resource.res_dict['ckpt'][1])
                self.am_postnet = os.path.join(
                    self.am_res_path, self.task_resource.res_dict['ckpt'][2])
                # must have phones_dict in acoustic
                self.phones_dict = os.path.join(
                    self.am_res_path,
                    self.task_resource.res_dict['phones_dict'])
                self.am_stat = os.path.join(
                    self.am_res_path,
                    self.task_resource.res_dict['speech_stats'])

            else:
                self.am_encoder_infer = os.path.abspath(am_ckpt[0])
                self.am_decoder = os.path.abspath(am_ckpt[1])
                self.am_postnet = os.path.abspath(am_ckpt[2])
                self.phones_dict = os.path.abspath(phones_dict)
                self.am_stat = os.path.abspath(am_stat)
                self.am_res_path = os.path.dirname(os.path.abspath(am_ckpt[0]))

            # create am sess
            self.am_encoder_infer_sess = get_sess(self.am_encoder_infer,
                                                  am_sess_conf)
            self.am_decoder_sess = get_sess(self.am_decoder, am_sess_conf)
            self.am_postnet_sess = get_sess(self.am_postnet, am_sess_conf)

            self.am_mu, self.am_std = np.load(self.am_stat)

        logger.debug(f"self.phones_dict: {self.phones_dict}")
        logger.debug(f"am model dir: {self.am_res_path}")
        logger.debug("Create am sess successfully.")

        # voc model info
        voc_tag = voc + '-' + lang

        if voc_ckpt is None:
            self.task_resource.set_task_model(
                model_tag=voc_tag,
                model_type=1,  # vocoder
                version=None,  # default version
            )
            self.voc_res_path = self.task_resource.voc_res_dir
            self.voc_ckpt = os.path.join(
                self.voc_res_path, self.task_resource.voc_res_dict['ckpt'])
        else:
            self.voc_ckpt = os.path.abspath(voc_ckpt)
            self.voc_res_path = os.path.dirname(os.path.abspath(self.voc_ckpt))
        logger.debug(self.voc_res_path)

        # create voc sess
        self.voc_sess = get_sess(self.voc_ckpt, voc_sess_conf)
        logger.debug("Create voc sess successfully.")

        with open(self.phones_dict, "r", encoding='utf-8') as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        self.vocab_size = len(phn_id)
        logger.debug(f"vocab_size: {self.vocab_size}")

        # frontend
        self.tones_dict = None
        if lang == 'zh':
            self.frontend = Frontend(
                phone_vocab_path=self.phones_dict,
                tone_vocab_path=self.tones_dict)

        elif lang == 'en':
            self.frontend = English(phone_vocab_path=self.phones_dict)
        logger.debug("frontend done!")


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
        self.engine_type = "online-onnx"

        self.am_block = self.config.am_block
        self.am_pad = self.config.am_pad
        self.voc_block = self.config.voc_block
        self.voc_pad = self.config.voc_pad
        self.am_upsample = 1
        self.voc_upsample = self.config.voc_upsample

        assert (
            self.config.am == "fastspeech2_csmsc_onnx" or
            self.config.am == "fastspeech2_cnndecoder_csmsc_onnx"
        ) and (
            self.config.voc == "hifigan_csmsc_onnx" or
            self.config.voc == "mb_melgan_csmsc_onnx"
        ), 'Please check config, am support: fastspeech2, voc support: hifigan_csmsc-zh or mb_melgan_csmsc.'

        assert (
            self.config.voc_block > 0 and self.config.voc_pad > 0
        ), "Please set correct voc_block and voc_pad, they should be more than 0."

        assert (
            self.config.voc_sample_rate == self.config.am_sample_rate
        ), "The sample rate of AM and Vocoder model are different, please check model."

        self.sample_rate = self.config.voc_sample_rate

        try:
            if self.config.am_sess_conf.device is not None:
                self.device = self.config.am_sess_conf.device
            elif self.config.voc_sess_conf.device is not None:
                self.device = self.config.voc_sess_conf.device
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
                am_ckpt=self.config.am_ckpt,
                am_stat=self.config.am_stat,
                phones_dict=self.config.phones_dict,
                tones_dict=self.config.tones_dict,
                speaker_dict=self.config.speaker_dict,
                am_sample_rate=self.config.am_sample_rate,
                am_sess_conf=self.config.am_sess_conf,
                voc=self.config.voc,
                voc_ckpt=self.config.voc_ckpt,
                voc_sample_rate=self.config.voc_sample_rate,
                voc_sess_conf=self.config.voc_sess_conf,
                lang=self.config.lang)

        except Exception as e:
            logger.error("Failed to get model related files.")
            logger.error("Initialize TTS server engine Failed on device: %s." %
                         (self.config.voc_sess_conf.device))
            logger(e)
            return False

        logger.info("Initialize TTS server engine successfully on device: %s." %
                    (self.config.voc_sess_conf.device))

        return True


class PaddleTTSConnectionHandler:
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
            am: str='fastspeech2_csmsc_onnx',
            spk_id: int=0, ):
        """
        Model inference and result stored in self.output.
        """

        # first_flag 用于标记首包
        first_flag = 1
        get_tone_ids = False
        merge_sentences = False

        # front 
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
            logger.error("lang should in {'zh', 'en'}!")
        frontend_et = time.time()
        self.frontend_time = frontend_et - frontend_st

        for i in range(len(phone_ids)):
            part_phone_ids = phone_ids[i].numpy()
            voc_chunk_id = 0

            # fastspeech2_csmsc
            if am == "fastspeech2_csmsc_onnx":
                # am 
                mel = self.executor.am_sess.run(
                    output_names=None, input_feed={'text': part_phone_ids})
                mel = mel[0]
                if first_flag == 1:
                    first_am_et = time.time()
                    self.first_am_infer = first_am_et - frontend_et

                # voc streaming
                mel_chunks = get_chunks(mel, self.voc_block, self.voc_pad,
                                        "voc")
                voc_chunk_num = len(mel_chunks)
                voc_st = time.time()
                for i, mel_chunk in enumerate(mel_chunks):
                    sub_wav = self.executor.voc_sess.run(
                        output_names=None, input_feed={'logmel': mel_chunk})
                    sub_wav = self.depadding(sub_wav[0], voc_chunk_num, i,
                                             self.voc_block, self.voc_pad,
                                             self.voc_upsample)
                    if first_flag == 1:
                        first_voc_et = time.time()
                        self.first_voc_infer = first_voc_et - first_am_et
                        self.first_response_time = first_voc_et - frontend_st
                        first_flag = 0

                    yield sub_wav

            # fastspeech2_cnndecoder_csmsc 
            elif am == "fastspeech2_cnndecoder_csmsc_onnx":
                # am 
                orig_hs = self.executor.am_encoder_infer_sess.run(
                    None, input_feed={'text': part_phone_ids})
                orig_hs = orig_hs[0]

                # streaming voc chunk info
                mel_len = orig_hs.shape[1]
                voc_chunk_num = math.ceil(mel_len / self.voc_block)
                start = 0
                end = min(self.voc_block + self.voc_pad, mel_len)

                # streaming am
                hss = get_chunks(orig_hs, self.am_block, self.am_pad, "am")
                am_chunk_num = len(hss)
                for i, hs in enumerate(hss):
                    am_decoder_output = self.executor.am_decoder_sess.run(
                        None, input_feed={'xs': hs})
                    am_postnet_output = self.executor.am_postnet_sess.run(
                        None,
                        input_feed={
                            'xs': np.transpose(am_decoder_output[0], (0, 2, 1))
                        })
                    am_output_data = am_decoder_output + np.transpose(
                        am_postnet_output[0], (0, 2, 1))
                    normalized_mel = am_output_data[0][0]

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

                        sub_wav = self.executor.voc_sess.run(
                            output_names=None, input_feed={'logmel': voc_chunk})
                        sub_wav = self.depadding(
                            sub_wav[0], voc_chunk_num, voc_chunk_id,
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

    def run(self, sentence: str, spk_id: int=0):
        """ run include inference and postprocess.

        Args:
            sentence (str): text to be synthesized
            spk_id (int, optional): speaker id for multi-speaker speech synthesis. Defaults to 0.
            
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
        duration = len(wav_all) / self.tts_engine.sample_rate
        logger.info(f"sentence: {sentence}")
        logger.info(f"The durations of audio is: {duration} s")
        logger.info(f"first response time: {self.first_response_time} s")
        logger.info(f"final response time: {self.final_response_time} s")
        logger.info(f"RTF: {self.final_response_time / duration}")
        logger.info(
            f"Other info: front time: {self.frontend_time} s, first am infer time: {self.first_am_infer} s, first voc infer time: {self.first_voc_infer} s,"
        )
