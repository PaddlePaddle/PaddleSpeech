# !/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2021    liangyunming(liangyunming@baidu.com)
#
########################################################################
"""tts fastapi Request basemodel
"""
from pydantic import BaseModel

class TTSRequest(BaseModel):
    
    """
    request body example
    {
    "apikey": "ASDF123adfyesdf", 
    "secrit": "sKewk23m", 
    "appid": "10123456", 
    "data": "你好，欢迎使用语音合成服务"
    }
    """

    text:str
    spk_id:int=0
    speed:float=1.0
    volume:float=1.0
    sample_rate:int=0
    tts_audio_path:str=None
    audio_format:str="wav"