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
import traceback
from typing import Union

from fastapi import APIRouter

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.engine_pool import get_engine_pool
from paddlespeech.server.engine.text.python.text_engine import PaddleTextConnectionHandler
from paddlespeech.server.restful.request import TextRequest
from paddlespeech.server.restful.response import ErrorResponse
from paddlespeech.server.restful.response import TextResponse
from paddlespeech.server.utils.errors import ErrorCode
from paddlespeech.server.utils.errors import failed_response
from paddlespeech.server.utils.exception import ServerBaseException

router = APIRouter()


@router.get('/paddlespeech/text/help')
def help():
    """help

    Returns:
        json: The /paddlespeech/text api response content
    """
    response = {
        "success": "True",
        "code": 200,
        "message": {
            "global": "success"
        },
        "result": {
            "punc_text": "The punctuation text content"
        }
    }
    return response


@router.post("/paddlespeech/text",
             response_model=Union[TextResponse, ErrorResponse])
def asr(request_body: TextRequest):
    """asr api 

    Args:
        request_body (TextRequest): the punctuation request body

    Returns:
        json: the punctuation response body
    """
    try:
        # 1. we get the sentence content from the request
        text = request_body.text
        logger.info(f"Text service receive the {text}")

        # 2. get single engine from engine pool
        #    and each request has its own connection to process the text
        engine_pool = get_engine_pool()
        text_engine = engine_pool['text']
        connection_handler = PaddleTextConnectionHandler(text_engine)
        punc_text = connection_handler.run(text)
        logger.info(f"Get the Text Connection result {punc_text}")

        # 3. create the response
        if punc_text is None:
            punc_text = text
        response = {
            "success": True,
            "code": 200,
            "message": {
                "description": "success"
            },
            "result": {
                "punc_text": punc_text
            }
        }

        logger.info(f"The Text Service final response: {response}")
    except ServerBaseException as e:
        response = failed_response(e.error_code, e.msg)
    except BaseException:
        response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
        traceback.print_exc()

    return response
