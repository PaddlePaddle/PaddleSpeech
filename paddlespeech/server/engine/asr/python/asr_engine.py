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
import sys
import time

import paddle

from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.log import logger
from paddlespeech.server.engine.base_engine import BaseEngine

__all__ = ['ASREngine', 'PaddleASRConnectionHandler']


class ASRServerExecutor(ASRExecutor):
    def __init__(self):
        super().__init__()
        pass


class ASREngine(BaseEngine):
    """ASR server engine

    Args:
        metaclass: Defaults to Singleton.
    """

    def __init__(self):
        super(ASREngine, self).__init__()

    def init(self, config: dict) -> bool:
        """init engine resource

        Args:
            config_file (str): config file

        Returns:
            bool: init failed or success
        """
        self.executor = ASRServerExecutor()
        self.config = config
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
            logger.error(e)
            return False
        
        self.executor._init_from_path(
            self.config.model, self.config.lang, self.config.sample_rate,
            self.config.cfg_path, self.config.decode_method, self.config.num_decoding_left_chunks,
            self.config.ckpt_path)


        logger.info("Initialize ASR server engine successfully on device: %s." %
                    (self.device))
        return True


class PaddleASRConnectionHandler(ASRServerExecutor):
    def __init__(self, asr_engine):
        """The PaddleSpeech ASR Server Connection Handler
           This connection process every asr server request
        Args:
            asr_engine (ASREngine): The ASR engine
        """
        super().__init__()
        self.input = None
        self.output = None
        self.asr_engine = asr_engine
        self.executor = self.asr_engine.executor
        self.max_len = self.executor.max_len
        self.text_feature = self.executor.text_feature
        self.model = self.executor.model
        self.config = self.executor.config

    def run(self, audio_data):
        """engine run 

        Args:
            audio_data (bytes): base64.b64decode
        """
        try:
            if self._check(
                    io.BytesIO(audio_data), self.asr_engine.config.sample_rate,
                    self.asr_engine.config.force_yes):
                logger.debug("start run asr engine")
                self.preprocess(self.asr_engine.config.model,
                                io.BytesIO(audio_data))
                st = time.time()
                self.infer(self.asr_engine.config.model)
                infer_time = time.time() - st
                self.output = self.postprocess()  # Retrieve result of asr.
            else:
                logger.error("file check failed!")
                self.output = None

            logger.info("inference time: {}".format(infer_time))
            logger.info("asr engine type: python")
        except Exception as e:
            logger.info(e)
            sys.exit(-1)
