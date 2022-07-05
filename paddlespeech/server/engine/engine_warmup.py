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
import time

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.engine_pool import get_engine_pool


def warm_up(engine_and_type: str, warm_up_time: int=3) -> bool:
    engine_pool = get_engine_pool()

    if "tts" in engine_and_type:
        tts_engine = engine_pool['tts']
        flag_online = False
        if tts_engine.lang == 'zh':
            sentence = "您好，欢迎使用语音合成服务。"
        elif tts_engine.lang == 'en':
            sentence = "Hello and welcome to the speech synthesis service."
        else:
            logger.error("tts engine only support lang: zh or en.")
            sys.exit(-1)

        if engine_and_type == "tts_python":
            from paddlespeech.server.engine.tts.python.tts_engine import PaddleTTSConnectionHandler
        elif engine_and_type == "tts_inference":
            from paddlespeech.server.engine.tts.paddleinference.tts_engine import PaddleTTSConnectionHandler
        elif engine_and_type == "tts_online":
            from paddlespeech.server.engine.tts.online.python.tts_engine import PaddleTTSConnectionHandler
            flag_online = True
        elif engine_and_type == "tts_online-onnx":
            from paddlespeech.server.engine.tts.online.onnx.tts_engine import PaddleTTSConnectionHandler
            flag_online = True
        else:
            logger.error("Please check tte engine type.")

        try:
            logger.debug("Start to warm up tts engine.")
            for i in range(warm_up_time):
                connection_handler = PaddleTTSConnectionHandler(tts_engine)
                if flag_online:
                    for wav in connection_handler.infer(
                            text=sentence,
                            lang=tts_engine.lang,
                            am=tts_engine.config.am):
                        logger.debug(
                            f"The first response time of the {i} warm up: {connection_handler.first_response_time} s"
                        )
                        break

                else:
                    st = time.time()
                    connection_handler.infer(text=sentence)
                    et = time.time()
                    logger.debug(
                        f"The response time of the {i} warm up: {et - st} s")
        except Exception as e:
            logger.error("Failed to warm up on tts engine.")
            logger.error(e)
            return False

    else:
        pass

    return True
