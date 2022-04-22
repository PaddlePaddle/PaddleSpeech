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
import json
import logging

import numpy as np
import soundfile
import websockets

from paddlespeech.cli.log import logger


class ASRAudioHandler:
    def __init__(self, url="127.0.0.1", port=8090):
        """PaddleSpeech Online ASR Server Client  audio handler
           Online asr server use the websocket protocal
        Args:
            url (str, optional): the server ip. Defaults to "127.0.0.1".
            port (int, optional): the server port. Defaults to 8090.
        """
        self.url = url
        self.port = port
        self.url = "ws://" + self.url + ":" + str(self.port) + "/ws/asr"

    def read_wave(self, wavfile_path: str):
        """read the audio file from specific wavfile path

        Args:
            wavfile_path (str): the audio wavfile, 
                                 we assume that audio sample rate matches the model

        Yields:
            numpy.array: the samall package audio pcm data
        """
        samples, sample_rate = soundfile.read(wavfile_path, dtype='int16')
        x_len = len(samples)

        chunk_size = 85 * 16  #80ms, sample_rate = 16kHz
        if x_len % chunk_size != 0:
            padding_len_x = chunk_size - x_len % chunk_size
        else:
            padding_len_x = 0

        padding = np.zeros((padding_len_x), dtype=samples.dtype)
        padded_x = np.concatenate([samples, padding], axis=0)

        assert (x_len + padding_len_x) % chunk_size == 0
        num_chunk = (x_len + padding_len_x) / chunk_size
        num_chunk = int(num_chunk)
        for i in range(0, num_chunk):
            start = i * chunk_size
            end = start + chunk_size
            x_chunk = padded_x[start:end]
            yield x_chunk

    async def run(self, wavfile_path: str):
        """Send a audio file to online server

        Args:
            wavfile_path (str): audio path

        Returns:
            str: the final asr result
        """
        logging.info("send a message to the server")

        # 1. send websocket handshake protocal
        async with websockets.connect(self.url) as ws:
            # 2. server has already received handshake protocal
            # client start to send the command
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
            logger.info("receive msg={}".format(msg))

            # 3. send chunk audio data to engine
            for chunk_data in self.read_wave(wavfile_path):
                await ws.send(chunk_data.tobytes())
                msg = await ws.recv()
                msg = json.loads(msg)
                logger.info("receive msg={}".format(msg))

            # 4. we must send finished signal to the server
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

            # 5. decode the bytes to str
            msg = json.loads(msg)
            logger.info("final receive msg={}".format(msg))
            result = msg
            return result
