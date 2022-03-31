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
from typing import List

from fastapi import APIRouter

from paddlespeech.server.ws.asr_socket import router as asr_router

_router = APIRouter()


def setup_router(api_list: List):
    """setup router for fastapi
    Args:
        api_list (List): [asr, tts]
    Returns:
        APIRouter
    """
    for api_name in api_list:
        if api_name == 'asr':
            _router.include_router(asr_router)
        elif api_name == 'tts':
            pass
        else:
            pass

    return _router
