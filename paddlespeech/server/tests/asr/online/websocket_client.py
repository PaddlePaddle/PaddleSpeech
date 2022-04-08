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
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import asyncio
import json
import logging

import numpy as np
import soundfile
import websockets


class ASRAudioHandler:
    def __init__(self, url="127.0.0.1", port=8090):
        self.url = url
        self.port = port
        self.url = "ws://" + self.url + ":" + str(self.port) + "/ws/asr"

    def read_wave(self, wavfile_path: str):
        samples, sample_rate = soundfile.read(wavfile_path, dtype='int16')
        x_len = len(samples)
        chunk_stride = 40 * 16  #40ms, sample_rate = 16kHz
        chunk_size = 80 * 16  #80ms, sample_rate = 16kHz

        if (x_len - chunk_size) % chunk_stride != 0:
            padding_len_x = chunk_stride - (x_len - chunk_size) % chunk_stride
        else:
            padding_len_x = 0

        padding = np.zeros((padding_len_x), dtype=samples.dtype)
        padded_x = np.concatenate([samples, padding], axis=0)

        num_chunk = (x_len + padding_len_x - chunk_size) / chunk_stride + 1
        num_chunk = int(num_chunk)

        for i in range(0, num_chunk):
            start = i * chunk_stride
            end = start + chunk_size
            x_chunk = padded_x[start:end]
            yield x_chunk

    async def run(self, wavfile_path: str):
        logging.info("send a message to the server")
        # 读取音频
        # self.read_wave()
        # 发送 websocket 的 handshake 协议头
        async with websockets.connect(self.url) as ws:
            # server 端已经接收到 handshake 协议头
            # 发送开始指令
            audio_info = json.dumps(
                {
                    "name": "test.wav",
                    "signal": "start",
                    "nbest": 5
                },
                sort_keys=True,
                indent=4,
                separators=(',', ': '))
            await ws.send(audio_info)
            msg = await ws.recv()
            logging.info("receive msg={}".format(msg))

            # send chunk audio data to engine
            for chunk_data in self.read_wave(wavfile_path):
                await ws.send(chunk_data.tobytes())
                msg = await ws.recv()
                logging.info("receive msg={}".format(msg))

            # finished 
            audio_info = json.dumps(
                {
                    "name": "test.wav",
                    "signal": "end",
                    "nbest": 5
                },
                sort_keys=True,
                indent=4,
                separators=(',', ': '))
            await ws.send(audio_info)
            msg = await ws.recv()
            logging.info("receive msg={}".format(msg))


def main(args):
    logging.basicConfig(level=logging.INFO)
    logging.info("asr websocket client start")
    handler = ASRAudioHandler("127.0.0.1", 8091)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(handler.run(args.wavfile))
    logging.info("asr websocket client finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wavfile",
        action="store",
        help="wav file path ",
        default="./16_audio.wav")
    args = parser.parse_args()

    main(args)
