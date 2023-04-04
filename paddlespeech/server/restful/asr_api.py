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
import base64
import sys
import traceback
from typing import Union

from fastapi import APIRouter

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.engine_pool import get_engine_pool
from paddlespeech.server.restful.request import ASRRequest
from paddlespeech.server.restful.response import ASRResponse
from paddlespeech.server.restful.response import ErrorResponse
from paddlespeech.server.utils.errors import ErrorCode
from paddlespeech.server.utils.errors import failed_response
from paddlespeech.server.utils.exception import ServerBaseException

router = APIRouter()


@router.get('/paddlespeech/asr/help')
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
            "description": "asr server",
            "input": "base64 string of wavfile",
            "output": "transcription"
        }
    }
    return response


@router.post("/paddlespeech/asr",
             response_model=Union[ASRResponse, ErrorResponse])
def asr(request_body: ASRRequest):
    """asr api 

    Args:
        request_body (ASRRequest): [description]

    Returns:
        json: [description]
    """
    try:
        audio_data = base64.b64decode(request_body.audio)

        # get single engine from engine pool
        engine_pool = get_engine_pool()
        asr_engine = engine_pool['asr']

        if asr_engine.engine_type == "python":
            from paddlespeech.server.engine.asr.python.asr_engine import PaddleASRConnectionHandler
        elif asr_engine.engine_type == "inference":
            from paddlespeech.server.engine.asr.paddleinference.asr_engine import PaddleASRConnectionHandler
        else:
            logger.error("Offline asr engine only support python or inference.")
            sys.exit(-1)

        connection_handler = PaddleASRConnectionHandler(asr_engine)

        connection_handler.run(audio_data)
        asr_results = connection_handler.postprocess()

        response = {
            "success": True,
            "code": 200,
            "message": {
                "description": "success"
            },
            "result": {
                "transcription": asr_results
            }
        }

    except ServerBaseException as e:
        response = failed_response(e.error_code, e.msg)
    except BaseException:
        response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
        traceback.print_exc()

    return response
