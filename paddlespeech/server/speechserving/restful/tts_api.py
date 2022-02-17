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

from engine.tts.python.tts_engine import TTSEngine
from fastapi import APIRouter

from .request import TTSRequest
from .response import ErrorResponse
from .response import TTSResponse
from utils.errors import ErrorCode
from utils.errors import failed_response
from utils.exception import ServerBaseException

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
    # json to dict 
    item_dict = request_body.dict()
    sentence = item_dict['text']
    spk_id = item_dict['spk_id']
    speed = item_dict['speed']
    volume = item_dict['volume']
    sample_rate = item_dict['sample_rate']
    save_path = item_dict['save_path']

    # Check parameters
    if speed <=0 or speed > 3 or volume <=0 or volume > 3 or \
        sample_rate not in [0, 16000, 8000] or \
        (save_path is not None and save_path.endswith("pcm") == False and save_path.endswith("wav") == False):
        return failed_response(ErrorCode.SERVER_PARAM_ERR)

    # single
    tts_engine = TTSEngine()

    # run
    try:
        lang, target_sample_rate, wav_base64 = tts_engine.run(
            sentence, spk_id, speed, volume, sample_rate, save_path)

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
                "save_path": save_path,
                "audio": wav_base64
            }
        }
    except ServerBaseException as e:
        response = failed_response(e.error_code, e.msg)
    except:
        response = failed_response(ErrorCode.SERVER_UNKOWN_ERR)
        traceback.print_exc()

    return response
