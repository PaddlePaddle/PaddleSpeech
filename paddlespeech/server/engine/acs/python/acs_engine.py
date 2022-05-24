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
import io
import json
import os
import re

import numpy as np
import paddle
import soundfile
import websocket

from paddlespeech.cli.log import logger
from paddlespeech.server.engine.base_engine import BaseEngine


class ACSEngine(BaseEngine):
    def __init__(self):
        """The ACSEngine Engine
        """
        super(ACSEngine, self).__init__()
        logger.info("Create the ACSEngine Instance")
        self.word_list = []

    def init(self, config: dict):
        """Init the ACSEngine Engine

        Args:
            config (dict): The server configuation

        Returns:
            bool: The engine instance flag
        """
        logger.info("Init the acs engine")
        try:
            self.config = config
            self.device = self.config.get("device", paddle.get_device())
            paddle.set_device(self.device)
            logger.info(f"ACS Engine set the device: {self.device}")

        except BaseException as e:
            logger.error(
                "Set device failed, please check if device is already used and the parameter 'device' in the yaml file"
            )
            logger.error("Initialize Text server engine Failed on device: %s." %
                         (self.device))
            return False

        self.read_search_words()

        # init the asr url
        self.url = "ws://" + self.config.asr_server_ip + ":" + str(
            self.config.asr_server_port) + "/paddlespeech/asr/streaming"

        logger.info("Init the acs engine successfully")
        return True

    def read_search_words(self):
        word_list = self.config.word_list
        if word_list is None:
            logger.error(
                "No word list file in config, please set the word list parameter"
            )
            return

        if not os.path.exists(word_list):
            logger.error("Please input correct word list file")
            return

        with open(word_list, 'r') as fp:
            self.word_list = [line.strip() for line in fp.readlines()]

        logger.info(f"word list: {self.word_list}")

    def get_asr_content(self, audio_data):
        """Get the streaming asr result

        Args:
            audio_data (_type_): _description_

        Returns:
            _type_: _description_
        """
        logger.info("send a message to the server")
        if self.url is None:
            logger.error("No asr server, please input valid ip and port")
            return ""
        ws = websocket.WebSocket()
        ws.connect(self.url)
        # with websocket.WebSocket.connect(self.url) as ws:
        audio_info = json.dumps(
            {
                "name": "test.wav",
                "signal": "start",
                "nbest": 1
            },
            sort_keys=True,
            indent=4,
            separators=(',', ': '))
        ws.send(audio_info)
        msg = ws.recv()
        logger.info("client receive msg={}".format(msg))

        # send the total audio data
        for chunk_data in self.read_wave(audio_data):
            ws.send_binary(chunk_data.tobytes())
            msg = ws.recv()
            msg = json.loads(msg)
            logger.info(f"audio result: {msg}")
        # samples, sample_rate = soundfile.read(audio_data, dtype='int16')

        # ws.send_binary(samples.tobytes())
        # msg = ws.recv()
        # msg = json.loads(msg)
        # logger.info(f"audio result: {msg}")

        # 3. send chunk audio data to engine
        logger.info("send the end signal")
        audio_info = json.dumps(
            {
                "name": "test.wav",
                "signal": "end",
                "nbest": 1
            },
            sort_keys=True,
            indent=4,
            separators=(',', ': '))
        ws.send(audio_info)
        msg = ws.recv()
        msg = json.loads(msg)

        logger.info(f"the final result: {msg}")
        ws.close()

        return msg

    def read_wave(self, audio_data: str):
        """read the audio file from specific wavfile path

        Args:
            audio_data (str): the audio data, 
                                 we assume that audio sample rate matches the model

        Yields:
            numpy.array: the samall package audio pcm data
        """
        samples, sample_rate = soundfile.read(audio_data, dtype='int16')
        x_len = len(samples)
        assert sample_rate == 16000

        chunk_size = int(85 * sample_rate / 1000)  # 85ms, sample_rate = 16kHz

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

    def get_macthed_word(self, msg):
        """Get the matched info in msg

        Args:
            msg (dict): the asr info, including the asr result and time stamp

        Returns:
            acs_result, asr_result: the acs result and the asr result
        """
        asr_result = msg['result']
        time_stamp = msg['times']
        acs_result = []

        # search for each word in self.word_list
        offset = self.config.offset
        max_ed = time_stamp[-1]['ed']
        for w in self.word_list:
            # search the w in asr_result and the index in asr_result
            for m in re.finditer(w, asr_result):
                start = max(time_stamp[m.start(0)]['bg'] - offset, 0)

                end = min(time_stamp[m.end(0) - 1]['ed'] + offset, max_ed)
                logger.info(f'start: {start}, end: {end}')
                acs_result.append({'w': w, 'bg': start, 'ed': end})

        return acs_result, asr_result

    def run(self, audio_data):
        """process the audio data in acs engine
           the engine does not store any data, so all the request use the self.run api

        Args:
            audio_data (str): the audio data

        Returns:
            acs_result, asr_result: the acs result and the asr result
        """
        logger.info("start to process the audio content search")
        msg = self.get_asr_content(io.BytesIO(audio_data))

        acs_result, asr_result = self.get_macthed_word(msg)
        logger.info(f'the asr result {asr_result}')
        logger.info(f'the acs result: {acs_result}')
        return acs_result, asr_result
