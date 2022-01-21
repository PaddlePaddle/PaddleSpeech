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
from typing import Optional
from typing import List

from pydantic import BaseModel

__all__ = ['ASRResponse']


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
        }
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
