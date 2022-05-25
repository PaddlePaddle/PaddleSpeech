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
import sys
import time

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.engine_factory import EngineFactory

# global value
ENGINE_POOL = {}


def get_engine_pool() -> dict:
    """ Get engine pool
    """
    global ENGINE_POOL
    return ENGINE_POOL


def warm_up(engine_and_type: str, engine, warm_up_time: int=3) -> bool:
    if "tts" in engine_and_type:
        if engine.lang == 'zh':
            sentence = "您好，欢迎使用语音合成服务。"
        elif engine.lang == 'en':
            sentence = "Hello and welcome to the speech synthesis service."
        else:
            logger.error("tts engine only support lang: zh or en.")
            sys.exit(-1)

        if engine_and_type == "tts_python":
            from paddlespeech.server.engine.tts.python.tts_engine import TTSHandler
        elif engine_and_type == "tts_inference":
            from paddlespeech.server.engine.tts.paddleinference.tts_engine import TTSHandler
        elif engine_and_type == "tts_online":
            pass
        elif engine_and_type == "tts_online-onnx":
            pass
        else:
            logger.error("Please check tte engine type.")

        try:
            logger.info("Start to warm up tts engine.")
            for i in range(warm_up_time):
                tts_handler = TTSHandler(engine)
                st = time.time()
                tts_handler.infer(
                    text=sentence,
                    lang=engine.config.lang,
                    am=engine.config.am,
                    spk_id=0, )
                logger.info(
                    f"The response time of the {i} warm up: {time.time() - st} s"
                )
        except Exception as e:
            logger.error("Failed to warm up on tts engine.")
            logger.error(e)
            return False

    else:
        pass
    return True


def init_engine_pool(config) -> bool:
    """ Init engine pool
    """
    global ENGINE_POOL

    for engine_and_type in config.engine_list:
        engine = engine_and_type.split("_")[0]
        engine_type = engine_and_type.split("_")[1]
        ENGINE_POOL[engine] = EngineFactory.get_engine(
            engine_name=engine, engine_type=engine_type)

        if not ENGINE_POOL[engine].init(config=config[engine_and_type]):
            return False

        if not warm_up(engine_and_type, ENGINE_POOL[engine]):
            return False

    return True
