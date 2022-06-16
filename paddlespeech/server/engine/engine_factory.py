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

from ..utils.log import logger

__all__ = ['EngineFactory']


class EngineFactory(object):
    @staticmethod
    def get_engine(engine_name: Text, engine_type: Text):
        logger.info(f"{engine_name} : {engine_type} engine.")

        if engine_name == 'asr' and engine_type == 'inference':
            from paddlespeech.server.engine.asr.paddleinference.asr_engine import ASREngine
            return ASREngine()
        elif engine_name == 'asr' and engine_type == 'python':
            from paddlespeech.server.engine.asr.python.asr_engine import ASREngine
            return ASREngine()
        elif engine_name == 'asr' and engine_type == 'online':
            from paddlespeech.server.engine.asr.online.python.asr_engine import ASREngine
            return ASREngine()
        elif engine_name == 'asr' and engine_type == 'online-inference':
            from paddlespeech.server.engine.asr.online.paddleinference.asr_engine import ASREngine
            return ASREngine()
        elif engine_name == 'asr' and engine_type == 'online-onnx':
            from paddlespeech.server.engine.asr.online.onnx.asr_engine import ASREngine
            return ASREngine()
        elif engine_name == 'tts' and engine_type == 'inference':
            from paddlespeech.server.engine.tts.paddleinference.tts_engine import TTSEngine
            return TTSEngine()
        elif engine_name == 'tts' and engine_type == 'python':
            from paddlespeech.server.engine.tts.python.tts_engine import TTSEngine
            return TTSEngine()
        elif engine_name == 'tts' and engine_type == 'online':
            from paddlespeech.server.engine.tts.online.python.tts_engine import TTSEngine
            return TTSEngine()
        elif engine_name == 'tts' and engine_type == 'online-onnx':
            from paddlespeech.server.engine.tts.online.onnx.tts_engine import TTSEngine
            return TTSEngine()
        elif engine_name == 'cls' and engine_type == 'inference':
            from paddlespeech.server.engine.cls.paddleinference.cls_engine import CLSEngine
            return CLSEngine()
        elif engine_name == 'cls' and engine_type == 'python':
            from paddlespeech.server.engine.cls.python.cls_engine import CLSEngine
            return CLSEngine()
        elif engine_name.lower() == 'text' and engine_type.lower() == 'python':
            from paddlespeech.server.engine.text.python.text_engine import TextEngine
            return TextEngine()
        elif engine_name.lower() == 'vector' and engine_type.lower() == 'python':
            from paddlespeech.server.engine.vector.python.vector_engine import VectorEngine
            return VectorEngine()
        elif engine_name.lower() == 'acs' and engine_type.lower() == 'python':
            from paddlespeech.server.engine.acs.python.acs_engine import ACSEngine
            return ACSEngine()
        else:
            return None
