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
from fastapi import APIRouter
import base64


from engine.asr.python.asr_engine import ASREngine
from .response import ASRResponse
from .request import ASRRequest

router = APIRouter()


@router.get('/paddlespeech/asr/help')
def help():
    """help

    Returns:
        json: [description]
    """
    return {'hello': 'world'}


@router.post("/paddlespeech/asr", response_model=ASRResponse)
def asr(request_body: ASRRequest):
    """asr api 

    Args:
        request_body (ASRRequest): [description]

    Returns:
        json: [description]
    """
    # single 
    asr_engine = ASREngine()

    asr_engine.init()
    asr_results = asr_engine.run()
    asr_engine.postprocess()

    json_body = {
                    "success": True,
                    "code": 0,
                    "message": {
                        "description": "success" 
                    },
                    "result": {
                        "transcription": asr_results
                    }
                }

    return json_body
