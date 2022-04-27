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
import base64
import json
import logging
import threading
import time

import numpy as np
import requests
import soundfile
import websockets

from paddlespeech.cli.log import logger
from paddlespeech.server.utils.audio_process import save_audio
from paddlespeech.server.utils.util import wav2base64


class TextHttpHandler:
    def __init__(self, server_ip="127.0.0.1", port=8090):
        """Text http client request 

        Args:
            server_ip (str, optional): the text server ip. Defaults to "127.0.0.1".
            port (int, optional): the text server port. Defaults to 8090.
        """
        super().__init__()
        self.server_ip = server_ip
        self.port = port
        if server_ip is None or port is None:
            self.url = None
        else:
            self.url = 'http://' + self.server_ip + ":" + str(
                self.port) + '/paddlespeech/text'

    def run(self, text):
        """Call the text server to process the specific text

        Args:
            text (str): the text to be processed

        Returns:
            str: punctuation text
        """
        if self.server_ip is None or self.port is None:
            return text
        request = {
            "text": text,
        }
        try:
            res = requests.post(url=self.url, data=json.dumps(request))
            response_dict = res.json()
            punc_text = response_dict["result"]["punc_text"]
        except Exception as e:
            logger.error(f"Call punctuation {self.url} occurs error")
            logger.error(e)
            punc_text = text

        return punc_text


class ASRWsAudioHandler:
    def __init__(self,
                 url=None,
                 port=None,
                 endpoint="/paddlespeech/asr/streaming",
                 punc_server_ip=None,
                 punc_server_port=None):
        """PaddleSpeech Online ASR Server Client  audio handler
           Online asr server use the websocket protocal
        Args:
            url (str, optional): the server ip. Defaults to None.
            port (int, optional): the server port. Defaults to None.
            endpoint(str, optional): to compatiable with python server and c++ server.
            punc_server_ip(str, optional): the punctuation server ip. Defaults to None. 
            punc_server_port(int, optional): the punctuation port. Defaults to None
        """
        self.url = url
        self.port = port
        if url is None or port is None or endpoint is None:
            self.url = None
        else:
            self.url = "ws://" + self.url + ":" + str(self.port) + endpoint
        self.punc_server = TextHttpHandler(punc_server_ip, punc_server_port)
        logger.info(f"endpoint: {self.url}")

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

        if self.url is None:
            logger.error("No asr server, please input valid ip and port")
            return ""

        # 1. send websocket handshake protocal
        async with websockets.connect(self.url) as ws:
            # 2. server has already received handshake protocal
            # client start to send the command
            audio_info = json.dumps(
                {
                    "name": "test.wav",
                    "signal": "start",
                    "nbest": 1
                },
                sort_keys=True,
                indent=4,
                separators=(',', ': '))
            await ws.send(audio_info)
            msg = await ws.recv()
            logger.info("client receive msg={}".format(msg))

            # 3. send chunk audio data to engine
            for chunk_data in self.read_wave(wavfile_path):
                await ws.send(chunk_data.tobytes())
                msg = await ws.recv()
                msg = json.loads(msg)

                if self.punc_server and len(msg["result"]) > 0:
                    msg["result"] = self.punc_server.run(msg["result"])
                logger.info("client receive msg={}".format(msg))

            # 4. we must send finished signal to the server
            audio_info = json.dumps(
                {
                    "name": "test.wav",
                    "signal": "end",
                    "nbest": 1
                },
                sort_keys=True,
                indent=4,
                separators=(',', ': '))
            await ws.send(audio_info)
            msg = await ws.recv()

            # 5. decode the bytes to str
            msg = json.loads(msg)

            if self.punc_server:
                msg["result"] = self.punc_server.run(msg["result"])

            logger.info("client final receive msg={}".format(msg))
            result = msg

            return result


class ASRHttpHandler:
    def __init__(self, server_ip=None, port=None):
        """The ASR client http request

        Args:
            server_ip (str, optional): the http asr server ip. Defaults to "127.0.0.1".
            port (int, optional): the http asr server port. Defaults to 8090.
        """
        super().__init__()
        self.server_ip = server_ip
        self.port = port
        if server_ip is None or port is None:
            self.url = None
        else:
            self.url = 'http://' + self.server_ip + ":" + str(
                self.port) + '/paddlespeech/asr'

    def run(self, input, audio_format, sample_rate, lang):
        """Call the http asr to process the audio

        Args:
            input (str): the audio file path
            audio_format (str): the audio format
            sample_rate (str): the audio sample rate
            lang (str): the audio language type

        Returns:
            str: the final asr result
        """
        if self.url is None:
            logger.error(
                "No punctuation server, please input valid ip and port")
            return ""

        audio = wav2base64(input)
        data = {
            "audio": audio,
            "audio_format": audio_format,
            "sample_rate": sample_rate,
            "lang": lang,
        }

        res = requests.post(url=self.url, data=json.dumps(data))

        return res.json()


class TTSWsHandler:
    def __init__(self, server="127.0.0.1", port=8092, play: bool=False):
        """PaddleSpeech Online TTS Server Client  audio handler
           Online tts server use the websocket protocal
        Args:
            server (str, optional): the server ip. Defaults to "127.0.0.1".
            port (int, optional): the server port. Defaults to 8092.
            play (bool, optional): whether to play audio. Defaults False
        """
        self.server = server
        self.port = port
        self.url = "ws://" + self.server + ":" + str(self.port) + "/ws/tts"
        self.play = play
        if self.play:
            import pyaudio
            self.buffer = b''
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=self.p.get_format_from_width(2),
                channels=1,
                rate=24000,
                output=True)
            self.mutex = threading.Lock()
            self.start_play = True
            self.t = threading.Thread(target=self.play_audio)
            self.max_fail = 50

    def play_audio(self):
        while True:
            if not self.buffer:
                self.max_fail -= 1
                time.sleep(0.05)
                if self.max_fail < 0:
                    break
            self.mutex.acquire()
            self.stream.write(self.buffer)
            self.buffer = b''
            self.mutex.release()

    async def run(self, text: str, output: str=None):
        """Send a text to online server

        Args:
            text (str): sentence to be synthesized
            output (str): save audio path
        """
        all_bytes = b''

        # 1. Send websocket handshake protocal
        async with websockets.connect(self.url) as ws:
            # 2. Server has already received handshake protocal
            # send text to engine
            text_base64 = str(base64.b64encode((text).encode('utf-8')), "UTF8")
            d = {"text": text_base64}
            d = json.dumps(d)
            st = time.time()
            await ws.send(d)
            logging.info("send a message to the server")

            # 3. Process the received response 
            message = await ws.recv()
            logger.info(f"句子：{text}")
            logger.info(f"首包响应：{time.time() - st} s")
            message = json.loads(message)
            status = message["status"]

            while (status == 1):
                audio = message["audio"]
                audio = base64.b64decode(audio)  # bytes
                all_bytes += audio
                if self.play:
                    self.mutex.acquire()
                    self.buffer += audio
                    self.mutex.release()
                    if self.start_play:
                        self.t.start()
                        self.start_play = False

                message = await ws.recv()
                message = json.loads(message)
                status = message["status"]

            # 4. Last packet, no audio information
            if status == 2:
                final_response = time.time() - st
                duration = len(all_bytes) / 2.0 / 24000
                logger.info(f"尾包响应：{final_response} s")
                logger.info(f"音频时长：{duration} s")
                logger.info(f"RTF: {final_response / duration}")

                if output is not None:
                    if save_audio(all_bytes, output):
                        logger.info(f"音频保存至：{output}")
                    else:
                        logger.error("save audio error")
            else:
                logger.error("infer error")

            if self.play:
                self.t.join()
                self.stream.stop_stream()
                self.stream.close()
                self.p.terminate()


class TTSHttpHandler:
    def __init__(self, server="127.0.0.1", port=8092, play: bool=False):
        """PaddleSpeech Online TTS Server Client  audio handler
           Online tts server use the websocket protocal
        Args:
            server (str, optional): the server ip. Defaults to "127.0.0.1".
            port (int, optional): the server port. Defaults to 8092.
            play (bool, optional): whether to play audio. Defaults False
        """
        self.server = server
        self.port = port
        self.url = "http://" + str(self.server) + ":" + str(
            self.port) + "/paddlespeech/streaming/tts"
        self.play = play

        if self.play:
            import pyaudio
            self.buffer = b''
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=self.p.get_format_from_width(2),
                channels=1,
                rate=24000,
                output=True)
            self.mutex = threading.Lock()
            self.start_play = True
            self.t = threading.Thread(target=self.play_audio)
            self.max_fail = 50

    def play_audio(self):
        while True:
            if not self.buffer:
                self.max_fail -= 1
                time.sleep(0.05)
                if self.max_fail < 0:
                    break
            self.mutex.acquire()
            self.stream.write(self.buffer)
            self.buffer = b''
            self.mutex.release()

    def run(self,
            text: str,
            spk_id=0,
            speed=1.0,
            volume=1.0,
            sample_rate=0,
            output: str=None):
        """Send a text to tts online server

        Args:
            text (str): sentence to be synthesized.
            spk_id (int, optional): speaker id. Defaults to 0.
            speed (float, optional): audio speed. Defaults to 1.0.
            volume (float, optional): audio volume. Defaults to 1.0.
            sample_rate (int, optional): audio sample rate, 0 means the same as model. Defaults to 0.
            output (str, optional): save audio path. Defaults to None.
        """
        # 1. Create request
        params = {
            "text": text,
            "spk_id": spk_id,
            "speed": speed,
            "volume": volume,
            "sample_rate": sample_rate,
            "save_path": output
        }

        all_bytes = b''
        first_flag = 1

        # 2. Send request
        st = time.time()
        html = requests.post(self.url, json.dumps(params), stream=True)

        # 3. Process the received response 
        for chunk in html.iter_content(chunk_size=1024):
            audio = base64.b64decode(chunk)  # bytes
            if first_flag:
                first_response = time.time() - st
                first_flag = 0

            if self.play:
                self.mutex.acquire()
                self.buffer += audio
                self.mutex.release()
                if self.start_play:
                    self.t.start()
                    self.start_play = False
            all_bytes += audio

        final_response = time.time() - st
        duration = len(all_bytes) / 2.0 / 24000

        logger.info(f"句子：{text}")
        logger.info(f"首包响应：{first_response} s")
        logger.info(f"尾包响应：{final_response} s")
        logger.info(f"音频时长：{duration} s")
        logger.info(f"RTF: {final_response / duration}")

        if output is not None:
            if save_audio(all_bytes, output):
                logger.info(f"音频保存至：{output}")
            else:
                logger.error("save audio error")

        if self.play:
            self.t.join()
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
