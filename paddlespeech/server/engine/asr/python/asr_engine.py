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

import paddle

from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.log import logger
from paddlespeech.server.engine.base_engine import BaseEngine
from paddlespeech.server.utils.config import get_config

__all__ = ['ASREngine']


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

        self.config = get_config(config_file)
        if self.config.device is None:
            paddle.set_device(paddle.get_device())
        else:
            paddle.set_device(self.config.device)
        self.executor._init_from_path(
            self.config.model, self.config.lang, self.config.sample_rate,
            self.config.cfg_path, self.config.decode_method,
            self.config.ckpt_path)

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
