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
# See the License for the 

#!/usr/bin/env python3
#! coding:utf-8
import logging
import time
import os
import json
import wave
import numpy as np
import asyncio
import websockets

class ASRAudioHandler:
    def __init__(self,
                 url="127.0.0.1",
                 port=8310):
        self.url = url
        self.port = port
        self.url = "ws://" + self.url + ":" + str(self.port) + "/ws/asr"
        self.chunk = 0.36
        # self.ws = websockets.connect('ws://szth-yc2-bce-ai2b-audio-727611.szth.baidu.com:8310')

    # def read_wave(self):
    #     audio = wave.open("/Users/zhangyinhui/Downloads/ffmpeg_audio-16k.wav", "rb")
    #     params = audio.getparams()
    #     nchannels, sampwidth, framerate, samplenum = params[:4]
    #     self.audio_data = np.frombuffer(audio.readframes(samplenum), np.int16)
    #     self.chunk = self.chunk * framerate
    #     audio.close()

    async def run(self):
        logging.info("send a message to the server")
        # 读取音频
        # self.read_wave()
        # 发送 websocket 的 handshake 协议头
        async with websockets.connect(self.url) as ws:
            # server 端已经接收到 handshake 协议头
            # 发送开始指令
            audio_info = json.dumps({
                            "name": "test.wav",
                            "signal": "start",
                            "nbest": 5
                            }, sort_keys=True, indent=4, separators=(',', ': '))
            await ws.send(audio_info)
            msg = await ws.recv()
            logging.info("receive msg={}".format(msg))

            # start = 0
            # isFinished = False
            # while start < len(self.audio_data):
            #     end = int(min(start + self.chunk, len(self.audio_data)))
            #     if end >= len(self.audio_data):
            #         isFinished = True
            #     chunk_data = self.audio_data[start:end].tobytes()
            #     await ws.send(chunk_data)
            #     logging.info("start={}, end={}, chunk size={}, isFinished={}".format(start, end, end-start, isFinished))
            #     start = end
            #     msg = await ws.recv()
            #     logging.info("receive msg={}".format(msg))

            # audio_info = json.dumps({
            #                 "name": "test.wav",
            #                 "signal": "end",
            #                 "nbest": 5
            #                 }, sort_keys=True, indent=4, separators=(',', ': '))
            # await ws.send(audio_info)
            # msg = await ws.recv()
            # logging.info("receive msg={}".format(msg))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("asr websocket client start")
    handler = ASRAudioHandler("127.0.0.1", 8090)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(handler.run())
    logging.info("asr websocket client finished")
