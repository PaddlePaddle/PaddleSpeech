import numpy as np
import logging
import argparse
import asyncio
import codecs
import json
import logging
import os

import numpy as np
import soundfile
import websockets
from paddlespeech.cli.log import logger
class ASRAudioHandler:
    def __init__(self, url="127.0.0.1", port=8090):
        self.url = url
        self.port = port
        self.url = "ws://" + self.url + ":" + str(self.port) + "/ws/asr"

    def read_wave(self, wavfile_path: str):
        samples, sample_rate = soundfile.read(wavfile_path, dtype='int16')
        x_len = len(samples)

        chunk_size = 85 * 16  #80ms, sample_rate = 16kHz
        if x_len % chunk_size!= 0:
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
        logging.info("send a message to the server")
        # self.read_wave()
        # send websocket handshake protocal
        async with websockets.connect(self.url) as ws:
            # server has already received handshake protocal
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

            # send chunk audio data to engine
            for chunk_data in self.read_wave(wavfile_path):
                await ws.send(chunk_data.tobytes())
                msg = await ws.recv()
                msg = json.loads(msg)
                logger.info("receive msg={}".format(msg))

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
            
            # decode the bytes to str
            msg = json.loads(msg)
            logger.info("final receive msg={}".format(msg))
            result = msg
            return result