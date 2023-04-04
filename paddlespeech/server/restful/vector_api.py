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
import traceback
from typing import Union

import numpy as np
from fastapi import APIRouter

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.engine_pool import get_engine_pool
from paddlespeech.server.engine.vector.python.vector_engine import PaddleVectorConnectionHandler
from paddlespeech.server.restful.request import VectorRequest
from paddlespeech.server.restful.request import VectorScoreRequest
from paddlespeech.server.restful.response import ErrorResponse
from paddlespeech.server.restful.response import VectorResponse
from paddlespeech.server.restful.response import VectorScoreResponse
from paddlespeech.server.utils.errors import ErrorCode
from paddlespeech.server.utils.errors import failed_response
from paddlespeech.server.utils.exception import ServerBaseException

router = APIRouter()


@router.get('/paddlespeech/vector/help')
def help():
    """help

    Returns:
        json: The /paddlespeech/vector api response content
    """
    response = {
        "success": "True",
        "code": 200,
        "message": {
            "global": "success"
        },
        "vector": [2.3, 3.5, 5.5, 6.2, 2.8, 1.2, 0.3, 3.6]
    }
    return response


@router.post("/paddlespeech/vector",
             response_model=Union[VectorResponse, ErrorResponse])
def vector(request_body: VectorRequest):
    """vector api 

    Args:
        request_body (VectorRequest): the vector request body

    Returns:
        json: the vector response body
    """
    try:
        # 1. get the audio data
        #    the audio must be base64 format
        audio_data = base64.b64decode(request_body.audio)

        # 2. get single engine from engine pool
        #    and we use the vector_engine to create an connection handler to process the request
        engine_pool = get_engine_pool()
        vector_engine = engine_pool['vector']
        connection_handler = PaddleVectorConnectionHandler(vector_engine)

        # 3. we use the connection handler to process the audio
        audio_vec = connection_handler.run(audio_data, request_body.task)

        # 4. we need the result of the vector instance be numpy.ndarray
        if not isinstance(audio_vec, np.ndarray):
            logger.error(
                f"the vector type is not numpy.array, that is: {type(audio_vec)}"
            )
            error_reponse = ErrorResponse()
            error_reponse.message.description = f"the vector type is not numpy.array, that is: {type(audio_vec)}"
            return error_reponse

        response = {
            "success": True,
            "code": 200,
            "message": {
                "description": "success"
            },
            "result": {
                "vec": audio_vec.tolist()
            }
        }

    except ServerBaseException as e:
        response = failed_response(e.error_code, e.msg)
    except BaseException:
        response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
        traceback.print_exc()

    return response


@router.post("/paddlespeech/vector/score",
             response_model=Union[VectorScoreResponse, ErrorResponse])
def score(request_body: VectorScoreRequest):
    """vector api 

    Args:
        request_body (VectorScoreRequest): the punctuation request body

    Returns:
        json: the punctuation response body
    """
    try:
        # 1. get the audio data
        #    the audio must be base64 format
        enroll_data = base64.b64decode(request_body.enroll_audio)
        test_data = base64.b64decode(request_body.test_audio)

        # 2. get single engine from engine pool
        #    and we use the vector_engine to create an connection handler to process the request
        engine_pool = get_engine_pool()
        vector_engine = engine_pool['vector']
        connection_handler = PaddleVectorConnectionHandler(vector_engine)

        # 3. we use the connection handler to process the audio
        score = connection_handler.get_enroll_test_score(enroll_data, test_data)

        response = {
            "success": True,
            "code": 200,
            "message": {
                "description": "success"
            },
            "result": {
                "score": score
            }
        }

    except ServerBaseException as e:
        response = failed_response(e.error_code, e.msg)
    except BaseException:
        response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
        traceback.print_exc()

    return response
