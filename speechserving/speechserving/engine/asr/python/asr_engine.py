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
from engine.base_engine import BaseEngine

from utils.log import logger
from utils.config import get_config

__all__ = ['ASREngine']


class ASREngine(BaseEngine):
    def __init__(self):
        super(ASREngine, self).__init__()

    def init(self, config_file: str):
        self.config_file = config_file
        self.executor = None
        self.input = None
        self.output = None
        config = get_config(self.config_file)
        pass

    def postprocess(self):
        pass

    def run(self):
        logger.info("start run asr engine")
        return "hello world"
