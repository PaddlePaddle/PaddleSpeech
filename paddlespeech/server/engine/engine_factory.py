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
from typing import Text

from paddlespeech.server.engine.asr.python.asr_engine import ASREngine
#from paddlespeech.server.engine.tts.python.tts_engine import TTSEngine
from paddlespeech.server.engine.tts.paddleinference.tts_engine import TTSEngine


__all__ = ['EngineFactory']


class EngineFactory(object):
    @staticmethod
    def get_engine(engine_name: Text):
        if engine_name == 'asr':
            return ASREngine()
        elif engine_name == 'tts':
            return TTSEngine()
        else:
            return None
