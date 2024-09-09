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
import traceback
from typing import Union

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.engine_pool import get_engine_pool
from paddlespeech.server.restful.request import TTSRequest
from paddlespeech.server.restful.response import ErrorResponse
from paddlespeech.server.restful.response import TTSResponse
from paddlespeech.server.utils.errors import ErrorCode
from paddlespeech.server.utils.errors import failed_response
from paddlespeech.server.utils.exception import ServerBaseException

router = APIRouter()


@router.get('/paddlespeech/tts/help')
def help():
    """help

    Returns:
        json: [description]
    """
    response = {
        "success": "True",
        "code": 200,
        "message": {
            "global": "success"
        },
        "result": {
            "description": "tts server",
            "text": "sentence to be synthesized",
            "audio": "the base64 of audio"
        }
    }
    return response


@router.post(
    "/paddlespeech/tts", response_model=Union[TTSResponse, ErrorResponse])
def tts(request_body: TTSRequest):
    """tts api

    Args:
        request_body (TTSRequest): [description]

    Returns:
        json: [description]
    """

    logger.info("request: {}".format(request_body))

    # get params
    text = request_body.text
    spk_id = request_body.spk_id
    speed = request_body.speed
    volume = request_body.volume
    sample_rate = request_body.sample_rate
    save_path = request_body.save_path

    # Check parameters
    if speed <= 0 or speed > 3:
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid speed value, the value should be between 0 and 3.")
    if volume <= 0 or volume > 3:
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid volume value, the value should be between 0 and 3.")
    if sample_rate not in [0, 16000, 8000]:
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid sample_rate value, the choice of value is 0, 8000, 16000.")
    if save_path is not None and not save_path.endswith(
            "pcm") and not save_path.endswith("wav"):
        return failed_response(
            ErrorCode.SERVER_PARAM_ERR,
            "invalid save_path, saved audio formats support pcm and wav")

    # run
    try:
        # get single engine from engine pool
        engine_pool = get_engine_pool()
        tts_engine = engine_pool['tts']
        logger.info("Get tts engine successfully.")

        if tts_engine.engine_type == "python":
            from paddlespeech.server.engine.tts.python.tts_engine import PaddleTTSConnectionHandler
        elif tts_engine.engine_type == "inference":
            from paddlespeech.server.engine.tts.paddleinference.tts_engine import PaddleTTSConnectionHandler
        else:
            logger.error("Offline tts engine only support python or inference.")
            sys.exit(-1)

        connection_handler = PaddleTTSConnectionHandler(tts_engine)
        lang, target_sample_rate, duration, wav_base64 = connection_handler.run(
            text, spk_id, speed, volume, sample_rate, save_path)

        response = {
            "success": True,
            "code": 200,
            "message": {
                "description": "success."
            },
            "result": {
                "lang": lang,
                "spk_id": spk_id,
                "speed": speed,
                "volume": volume,
                "sample_rate": target_sample_rate,
                "duration": duration,
                "save_path": save_path,
                "audio": wav_base64
            }
        }
    except ServerBaseException as e:
        response = failed_response(e.error_code, e.msg)
    except BaseException:
        response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
        traceback.print_exc()

    return response


@router.post("/paddlespeech/tts/streaming")
async def stream_tts(request_body: TTSRequest):
    # get params
    text = request_body.text
    spk_id = request_body.spk_id

    engine_pool = get_engine_pool()
    tts_engine = engine_pool['tts']
    logger.info("Get tts engine successfully.")

    if tts_engine.engine_type == "online":
        from paddlespeech.server.engine.tts.online.python.tts_engine import PaddleTTSConnectionHandler
    elif tts_engine.engine_type == "online-onnx":
        from paddlespeech.server.engine.tts.online.onnx.tts_engine import PaddleTTSConnectionHandler
    else:
        logger.error("Online tts engine only support online or online-onnx.")
        sys.exit(-1)

    connection_handler = PaddleTTSConnectionHandler(tts_engine)

    return StreamingResponse(
        connection_handler.run(sentence=text, spk_id=spk_id))


@router.get("/paddlespeech/tts/streaming/samplerate")
def get_samplerate():
    try:
        engine_pool = get_engine_pool()
        tts_engine = engine_pool['tts']
        logger.info("Get tts engine successfully.")
        sample_rate = tts_engine.sample_rate

        response = {"sample_rate": sample_rate}

    except ServerBaseException as e:
        response = failed_response(e.error_code, e.msg)
    except BaseException:
        response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
        traceback.print_exc()

    return response
