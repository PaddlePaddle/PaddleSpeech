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

__all__ = ['ASREngine']


class ASREngine(BaseEngine):
    def __init__(self, name=None):
        super(ASREngine, self).__init__()
        self.executor = name
        self.input = None
        self.output = None

    def init(self):
        pass

    def postprocess(self):
        pass

    def run(self):
        logger.info("start run asr engine")
        return "hello world"


if __name__ == "__main__":
    # test Singleton 
    class1 = ASREngine("ASREngine")
    class2 = ASREngine()
    print(class1 is class2)
    print(id(class1))
    print(id(class2))
