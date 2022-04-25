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

from pydantic import BaseModel

__all__ = ['ASRResponse', 'TTSResponse', 'CLSResponse']


class Message(BaseModel):
    description: str


#****************************************************************************************/
#************************************ ASR response **************************************/
#****************************************************************************************/
class AsrResult(BaseModel):
    transcription: str


class ASRResponse(BaseModel):
    """
    response example
    {
        "success": true,
        "code": 0,
        "message": {
            "description": "success" 
        },
        "result": {
            "transcription": "你好，飞桨"
        }
    }
    """
    success: bool
    code: int
    message: Message
    result: AsrResult


#****************************************************************************************/
#************************************ TTS response **************************************/
#****************************************************************************************/
class TTSResult(BaseModel):
    lang: str = "zh"
    spk_id: int = 0
    speed: float = 1.0
    volume: float = 1.0
    sample_rate: int
    duration: float
    save_path: str = None
    audio: str


class TTSResponse(BaseModel):
    """
    response example
    {
        "success": true,
        "code": 200,
        "message": {
            "description": "success" 
        },
        "result": {
            "lang": "zh",
            "spk_id": 0,
            "speed": 1.0,
            "volume": 1.0,
            "sample_rate": 24000,
            "duration": 3.6125,
            "audio": "LTI1OTIuNjI1OTUwMzQsOTk2OS41NDk4...",
            "save_path": "./tts.wav"
        }
    }
    """
    success: bool
    code: int
    message: Message
    result: TTSResult


#****************************************************************************************/
#************************************ CLS response **************************************/
#****************************************************************************************/
class CLSResults(BaseModel):
    class_name: str
    prob: float


class CLSResult(BaseModel):
    topk: int
    results: List[CLSResults]


class CLSResponse(BaseModel):
    """
    response example
    {
        "success": true,
        "code": 0,
        "message": {
            "description": "success" 
        },
        "result": {
            topk: 1
            results: [
            {
                "class":"Speech",
                "prob": 0.9027184844017029
            }
            ]
        }
    }
    """
    success: bool
    code: int
    message: Message
    result: CLSResult


class TextResult(BaseModel):
    punc_text: str


class TextResponse(BaseModel):
    """
    response example
    {
        "success": true,
        "code": 0,
        "message": {
            "description": "success" 
        },
        "result": {
            "punc_text": "你好，飞桨"
        }
    }
    """
    success: bool
    code: int
    message: Message
    result: TextResult


#****************************************************************************************/
#********************************** Error response **************************************/
#****************************************************************************************/
class ErrorResponse(BaseModel):
    """
    response example
    {
        "success": false,
        "code": 0,
        "message": {
            "description": "Unknown error occurred."
        }
    }
    """
    success: bool
    code: int
    message: Message
