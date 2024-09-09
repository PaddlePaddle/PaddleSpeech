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
from typing import Union

from fastapi import APIRouter

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.engine_pool import get_engine_pool
from paddlespeech.server.restful.request import ASRRequest
from paddlespeech.server.restful.response import ACSResponse
from paddlespeech.server.restful.response import ErrorResponse
from paddlespeech.server.utils.errors import ErrorCode
from paddlespeech.server.utils.errors import failed_response
from paddlespeech.server.utils.exception import ServerBaseException

router = APIRouter()


@router.get('/paddlespeech/asr/search/help')
def help():
    """help

    Returns:
        json: the audio content search result
    """
    response = {
        "success": "True",
        "code": 200,
        "message": {
            "global": "success"
        },
        "result": {
            "description": "acs server",
            "input": "base64 string of wavfile",
            "output": {
                "asr_result": "你好",
                "acs_result": [{
                    'w': '你',
                    'bg': 0.0,
                    'ed': 1.2
                }]
            }
        }
    }
    return response


@router.post(
    "/paddlespeech/asr/search",
    response_model=Union[ACSResponse, ErrorResponse])
def acs(request_body: ASRRequest):
    """acs api 

    Args:
        request_body (ASRRequest): the acs request, we reuse the http ASRRequest

    Returns:
        json: the acs result
    """
    try:
        # 1. get the audio data via base64 decoding
        audio_data = base64.b64decode(request_body.audio)

        # 2. get single engine from engine pool
        engine_pool = get_engine_pool()
        acs_engine = engine_pool['acs']

        # 3. no data stored in acs_engine, so we need to create the another instance process the data
        acs_result, asr_result = acs_engine.run(audio_data)

        response = {
            "success": True,
            "code": 200,
            "message": {
                "description": "success"
            },
            "result": {
                "transcription": asr_result,
                "acs": acs_result
            }
        }

    except ServerBaseException as e:
        response = failed_response(e.error_code, e.msg)
    except BaseException as e:
        response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
        logger.error(e)

    return response
