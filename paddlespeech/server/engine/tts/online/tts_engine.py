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
import time

import numpy as np
import paddle

from paddlespeech.cli.log import logger
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.audio_process import float2pcm
from paddlespeech.server.utils.util import get_chunks

__all__ = ['TTSEngine']


class TTSServerExecutor(TTSExecutor):
    def __init__(self):
        super().__init__()
        pass

    @paddle.no_grad()
    def infer(
            self,
            text: str,
            lang: str='zh',
            am: str='fastspeech2_csmsc',
            spk_id: int=0,
            am_block: int=42,
            am_pad: int=12,
            voc_block: int=14,
            voc_pad: int=14, ):
        """
        Model inference and result stored in self.output.
        """
        am_name = am[:am.rindex('_')]
        am_dataset = am[am.rindex('_') + 1:]
        get_tone_ids = False
        merge_sentences = False
        frontend_st = time.time()
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
            am_et = time.time()

            # voc streaming
            voc_upsample = self.voc_config.n_shift
            mel_chunks = get_chunks(mel, voc_block, voc_pad, "voc")
            chunk_num = len(mel_chunks)
            voc_st = time.time()
            for i, mel_chunk in enumerate(mel_chunks):
                sub_wav = self.voc_inference(mel_chunk)
                front_pad = min(i * voc_block, voc_pad)

                if i == 0:
                    sub_wav = sub_wav[:voc_block * voc_upsample]
                elif i == chunk_num - 1:
                    sub_wav = sub_wav[front_pad * voc_upsample:]
                else:
                    sub_wav = sub_wav[front_pad * voc_upsample:(
                        front_pad + voc_block) * voc_upsample]

                yield sub_wav


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
        assert "fastspeech2_csmsc" in config.am and (
            config.voc == "hifigan_csmsc-zh" or config.voc == "mb_melgan_csmsc"
        ), 'Please check config, am support: fastspeech2, voc support: hifigan_csmsc-zh or mb_melgan_csmsc.'
        try:
            if self.config.device:
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
            return False

        self.am_block = self.config.am_block
        self.am_pad = self.config.am_pad
        self.voc_block = self.config.voc_block
        self.voc_pad = self.config.voc_pad

        logger.info("Initialize TTS server engine successfully on device: %s." %
                    (self.device))
        return True

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

        lang = self.config.lang
        wav_list = []

        for wav in self.executor.infer(
                text=sentence,
                lang=lang,
                am=self.config.am,
                spk_id=spk_id,
                am_block=self.am_block,
                am_pad=self.am_pad,
                voc_block=self.voc_block,
                voc_pad=self.voc_pad):
            # wav type: <class 'numpy.ndarray'>  float32, convert to pcm (base64)
            wav = float2pcm(wav)  # float32 to int16
            wav_bytes = wav.tobytes()  # to bytes
            wav_base64 = base64.b64encode(wav_bytes).decode('utf8')  # to base64
            wav_list.append(wav)

            yield wav_base64

        wav_all = np.concatenate(wav_list, axis=0)
        logger.info("The durations of audio is: {} s".format(
            len(wav_all) / self.executor.am_config.fs))
