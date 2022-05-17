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
from typing import List

from fastapi import APIRouter

from paddlespeech.cli.log import logger
from paddlespeech.server.restful.asr_api import router as asr_router
from paddlespeech.server.restful.cls_api import router as cls_router
from paddlespeech.server.restful.text_api import router as text_router
from paddlespeech.server.restful.tts_api import router as tts_router
from paddlespeech.server.restful.vector_api import router as vec_router
_router = APIRouter()


def setup_router(api_list: List):
    """setup router for fastapi

    Args:
        api_list (List): [asr, tts, cls, text, vecotr]

    Returns:
        APIRouter
    """
    for api_name in api_list:
        if api_name.lower() == 'asr':
            _router.include_router(asr_router)
        elif api_name.lower() == 'tts':
            _router.include_router(tts_router)
        elif api_name.lower() == 'cls':
            _router.include_router(cls_router)
        elif api_name.lower() == 'text':
            _router.include_router(text_router)
        elif api_name.lower() == 'vector':
            _router.include_router(vec_router)
        else:
            logger.error(
                f"PaddleSpeech has not support such service: {api_name}")
            sys.exit(-1)

    return _router
