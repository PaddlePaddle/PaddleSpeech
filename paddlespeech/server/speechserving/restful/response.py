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
from typing import Optional

from pydantic import BaseModel

__all__ = ['ASRResponse', 'TTSResponse']


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
    sample_rate: int
    spk_id: int = 0
    speed: float = 1.0
    volume: float = 1.0
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
            "sample_rate": 24000,
            "speed": 1.0,
            "volume": 1.0,
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
