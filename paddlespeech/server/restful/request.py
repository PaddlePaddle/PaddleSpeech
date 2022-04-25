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

from pydantic import BaseModel

__all__ = ['ASRRequest', 'TTSRequest', 'CLSRequest']


#****************************************************************************************/
#************************************ ASR request ***************************************/
#****************************************************************************************/
class ASRRequest(BaseModel):
    """
    request body example
    {
        "audio": "exSI6ICJlbiIsCgkgICAgInBvc2l0aW9uIjogImZhbHNlIgoJf...",
        "audio_format": "wav",
        "sample_rate": 16000,
        "lang": "zh_cn",
        "punc":false
    }
    """
    audio: str
    audio_format: str
    sample_rate: int
    lang: str
    punc: Optional[bool] = None


#****************************************************************************************/
#************************************ TTS request ***************************************/
#****************************************************************************************/
class TTSRequest(BaseModel):
    """TTS request

    request body example
    {
        "text": "你好，欢迎使用百度飞桨语音合成服务。",
        "spk_id": 0,
        "speed": 1.0,
        "volume": 1.0,
        "sample_rate": 0,
        "tts_audio_path": "./tts.wav"
    }
    
    """

    text: str
    spk_id: int = 0
    speed: float = 1.0
    volume: float = 1.0
    sample_rate: int = 0
    save_path: str = None


#****************************************************************************************/
#************************************ CLS request ***************************************/
#****************************************************************************************/
class CLSRequest(BaseModel):
    """
    request body example
    {
        "audio": "exSI6ICJlbiIsCgkgICAgInBvc2l0aW9uIjogImZhbHNlIgoJf...",
        "topk": 1
    }
    """
    audio: str
    topk: int = 1


#****************************************************************************************/
#************************************ Text request **************************************/
#****************************************************************************************/
class TextRequest(BaseModel):
    text: str
