# !/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2021    liangyunming(liangyunming@baidu.com)
#
########################################################################
"""fastapi tts server
"""

import paddle
import argparse
import numpy as np
import uvicorn
import io

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy.io import wavfile

from engine import TTSEngine
from request import TTSRequest

import yaml
from yacs.config import CfgNode

def tts(sentence: str, 
        spk_id: int=0, 
        speed: float=1.0, 
        volume: float=1.0, 
        sample_rate: int=0, 
        tts_audio_path: str=None, 
        audio_format: str="wav"):
    # singleton
    tts_engine = TTSEngine()
    wav_np, sample_rate = tts_engine.run(sentence, spk_id, speed, volume, sample_rate, tts_audio_path, audio_format)

    return wav_np, sample_rate

# fastapi
app = FastAPI()

@app.get("/paddlespeech/tts/help")
def read_root():
    return "hello tts"

@app.post("/paddlespeech/tts")
async def synthesis(item: TTSRequest):
    item_dict = item.dict()
    sentence = item_dict['text']
    spk_id = item_dict['spk_id']
    speed = item_dict['speed']
    volume = item_dict['volume']
    sample_rate = item_dict['sample_rate']
    tts_audio_path = item_dict['tts_audio_path']
    audio_format = item_dict['audio_format']

    wav_data, sample_rate = tts(sentence, spk_id, speed, volume, sample_rate, tts_audio_path, audio_format)

    # wav to base64
    #TODO
    
    buf = io.BytesIO()
    wav_norm = wav_data * (32767 / max(0.01, np.max(np.abs(wav_data))))
    wavfile.write(buf, sample_rate, wav_norm.astype(np.int16))

    # response TODO
    # return json

    return StreamingResponse(buf)


if  __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--conf', type=str, default='./server.yaml', help='Configuration parameters.')
    args = parser.parse_args()

    with open(args.conf, 'rt') as f:
        config = CfgNode(yaml.safe_load(f))

    # singleton
    TTSNEW = TTSEngine()

    uvicorn.run(app='server:app', host=config.host, port=config.port, workers=1, debug=False)