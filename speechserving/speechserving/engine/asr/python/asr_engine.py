# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import io
import os
from typing import List
from typing import Optional
from typing import Union

import librosa
import paddle
import soundfile
from engine.base_engine import BaseEngine

from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.log import logger
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.transform.transformation import Transformation
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.utility import UpdateConfig
from utils.config import get_config

__all__ = ['ASREngine']


class ASRServerExecutor(ASRExecutor):
    def __init__(self):
        super().__init__()
        pass

    def _check(self, audio_file: str, sample_rate: int, force_yes: bool):
        self.sample_rate = sample_rate
        if self.sample_rate != 16000 and self.sample_rate != 8000:
            logger.error("please input --sr 8000 or --sr 16000")
            return False

        logger.info("checking the audio file format......")
        try:
            audio, audio_sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)
        except Exception as e:
            logger.exception(e)
            logger.error(
                "can not open the audio file, please check the audio file format is 'wav'. \n \
                 you can try to use sox to change the file format.\n \
                 For example: \n \
                 sample rate: 16k \n \
                 sox input_audio.xx --rate 16k --bits 16 --channels 1 output_audio.wav \n \
                 sample rate: 8k \n \
                 sox input_audio.xx --rate 8k --bits 16 --channels 1 output_audio.wav \n \
                 ")

        logger.info("The sample rate is %d" % audio_sample_rate)
        if audio_sample_rate != self.sample_rate:
            logger.warning("The sample rate of the input file is not {}.\n \
                            The program will resample the wav file to {}.\n \
                            If the result does not meet your expectations，\n \
                            Please input the 16k 16 bit 1 channel wav file. \
                        ".format(self.sample_rate, self.sample_rate))
            self.change_format = True
        else:
            logger.info("The audio file format is right")
            self.change_format = False

        return True

    def preprocess(self, model_type: str, input: Union[str, os.PathLike]):
        """
        Input preprocess and return paddle.Tensor stored in self.input.
        Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """

        audio_file = input

        # Get the object for feature extraction
        if "deepspeech2online" in model_type or "deepspeech2offline" in model_type:
            audio, _ = self.collate_fn_test.process_utterance(
                audio_file=audio_file, transcript=" ")
            audio_len = audio.shape[0]
            audio = paddle.to_tensor(audio, dtype='float32')
            audio_len = paddle.to_tensor(audio_len)
            audio = paddle.unsqueeze(audio, axis=0)
            # vocab_list = collate_fn_test.vocab_list
            self._inputs["audio"] = audio
            self._inputs["audio_len"] = audio_len
            logger.info(f"audio feat shape: {audio.shape}")

        elif "conformer" in model_type or "transformer" in model_type or "wenetspeech" in model_type:
            logger.info("get the preprocess conf")
            preprocess_conf = self.config.preprocess_config
            preprocess_args = {"train": False}
            preprocessing = Transformation(preprocess_conf)
            logger.info("read the audio file")
            audio, audio_sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)

            if self.change_format:
                if audio.shape[1] >= 2:
                    audio = audio.mean(axis=1, dtype=np.int16)
                else:
                    audio = audio[:, 0]
                # pcm16 -> pcm 32
                audio = self._pcm16to32(audio)
                audio = librosa.resample(audio, audio_sample_rate,
                                         self.sample_rate)
                audio_sample_rate = self.sample_rate
                # pcm32 -> pcm 16
                audio = self._pcm32to16(audio)
            else:
                audio = audio[:, 0]

            logger.info(f"audio shape: {audio.shape}")
            # fbank
            audio = preprocessing(audio, **preprocess_args)

            audio_len = paddle.to_tensor(audio.shape[0])
            audio = paddle.to_tensor(audio, dtype='float32').unsqueeze(axis=0)

            self._inputs["audio"] = audio
            self._inputs["audio_len"] = audio_len
            logger.info(f"audio feat shape: {audio.shape}")

        else:
            raise Exception("wrong type")


class ASREngine(BaseEngine):
    """ASR server engine

    Args:
        metaclass: Defaults to Singleton.
    """

    def __init__(self):
        super(ASREngine, self).__init__()

    def init(self, config_file: str) -> bool:
        """init engine resource

        Args:
            config_file (str): config file

        Returns:
            bool: init failed or success
        """
        self.input = None
        self.output = None
        self.executor = ASRServerExecutor()

        try:
            self.config = get_config(config_file)
            paddle.set_device(paddle.get_device())
            self.executor._init_from_path(
                self.config.model, self.config.lang, self.config.sample_rate,
                self.config.cfg_path, self.config.decode_method,
                self.config.ckpt_path)
        except:
            logger.info("Initialize ASR server engine Failed.")
            return False

        logger.info("Initialize ASR server engine successfully.")
        return True

    def run(self, audio_data):
        """engine run 

        Args:
            audio_data (bytes): base64.b64decode
        """
        if self.executor._check(
                io.BytesIO(audio_data), self.config.sample_rate,
                self.config.force_yes):
            logger.info("start run asr engine")
            self.executor.preprocess(self.config.model, io.BytesIO(audio_data))
            self.executor.infer(self.config.model)
            self.output = self.executor.postprocess()  # Retrieve result of asr.
        else:
            logger.info("file check failed!")
            self.output = None

    def postprocess(self):
        """postprocess
        """
        return self.output
