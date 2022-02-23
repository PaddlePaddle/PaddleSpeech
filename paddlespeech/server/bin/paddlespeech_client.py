# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import base64
import io
import json
import os
import random
import time
from typing import List

import numpy as np
import requests
import soundfile

from ..executor import BaseExecutor
from ..util import cli_client_register
from ..util import stats_wrapper
from paddlespeech.cli.log import logger
from paddlespeech.server.utils.audio_process import wav2pcm
from paddlespeech.server.utils.util import wav2base64

__all__ = ['TTSClientExecutor', 'ASRClientExecutor']


@cli_client_register(
    name='paddlespeech_client.tts', description='visit tts service')
class TTSClientExecutor(BaseExecutor):
    def __init__(self):
        super(TTSClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech_client.tts', add_help=True)
        self.parser.add_argument(
            '--server_ip', type=str, default='127.0.0.1', help='server ip')
        self.parser.add_argument(
            '--port', type=int, default=8090, help='server port')
        self.parser.add_argument(
            '--input',
            type=str,
            default="你好，欢迎使用语音合成服务",
            help='A sentence to be synthesized.')
        self.parser.add_argument(
            '--spk_id', type=int, default=0, help='Speaker id')
        self.parser.add_argument(
            '--speed',
            type=float,
            default=1.0,
            help='Audio speed, the value should be set between 0 and 3')
        self.parser.add_argument(
            '--volume',
            type=float,
            default=1.0,
            help='Audio volume, the value should be set between 0 and 3')
        self.parser.add_argument(
            '--sample_rate',
            type=int,
            default=0,
            choices=[0, 8000, 16000],
            help='Sampling rate, the default is the same as the model')
        self.parser.add_argument(
            '--output',
            type=str,
            default="./output.wav",
            help='Synthesized audio file')

    def postprocess(self, response_dict: dict, outfile: str) -> float:
        wav_base64 = response_dict["result"]["audio"]
        audio_data_byte = base64.b64decode(wav_base64)
        # from byte
        samples, sample_rate = soundfile.read(
            io.BytesIO(audio_data_byte), dtype='float32')

        # transform audio
        if outfile.endswith(".wav"):
            soundfile.write(outfile, samples, sample_rate)
        elif outfile.endswith(".pcm"):
            temp_wav = str(random.getrandbits(128)) + ".wav"
            soundfile.write(temp_wav, samples, sample_rate)
            wav2pcm(temp_wav, outfile, data_type=np.int16)
            os.system("rm %s" % (temp_wav))
        else:
            logger.error("The format for saving audio only supports wav or pcm")

        duration = len(samples) / sample_rate
        return duration

    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        try:
            url = 'http://' + args.server_ip + ":" + str(
                args.port) + '/paddlespeech/tts'
            request = {
                "text": args.input,
                "spk_id": args.spk_id,
                "speed": args.speed,
                "volume": args.volume,
                "sample_rate": args.sample_rate,
                "save_path": args.output
            }
            st = time.time()
            response = requests.post(url, json.dumps(request))
            time_consume = time.time() - st

            response_dict = response.json()
            duration = self.postprocess(response_dict, args.output)

            logger.info(response_dict["message"])
            logger.info("Save synthesized audio successfully on %s." %
                        (args.output))
            logger.info("Audio duration: %f s." % (duration))
            logger.info("Response time: %f s." % (time_consume))
            logger.info("RTF: %f " % (time_consume / duration))

            return True
        except:
            logger.error("Failed to synthesized audio.")
            return False

    @stats_wrapper
    def __call__(self,
                 input: str,
                 server_ip: str="127.0.0.1",
                 port: int=8090,
                 spk_id: int=0,
                 speed: float=1.0,
                 volume: float=1.0,
                 sample_rate: int=0,
                 output: str="./output.wav"):
        """
        Python API to call an executor.
        """

        url = 'http://' + server_ip + ":" + str(port) + '/paddlespeech/tts'
        request = {
            "text": input,
            "spk_id": spk_id,
            "speed": speed,
            "volume": volume,
            "sample_rate": sample_rate,
            "save_path": output
        }

        try:
            st = time.time()
            response = requests.post(url, json.dumps(request))
            time_consume = time.time() - st
            response_dict = response.json()
            duration = self.postprocess(response_dict, output)

            print(response_dict["message"])
            print("Save synthesized audio successfully on %s." % (output))
            print("Audio duration: %f s." % (duration))
            print("Response time: %f s." % (time_consume))
            print("RTF: %f " % (time_consume / duration))
        except:
            print("Failed to synthesized audio.")


@cli_client_register(
    name='paddlespeech_client.asr', description='visit asr service')
class ASRClientExecutor(BaseExecutor):
    def __init__(self):
        super(ASRClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech_client.asr', add_help=True)
        self.parser.add_argument(
            '--server_ip', type=str, default='127.0.0.1', help='server ip')
        self.parser.add_argument(
            '--port', type=int, default=8090, help='server port')
        self.parser.add_argument(
            '--input',
            type=str,
            default="./paddlespeech/server/tests/16_audio.wav",
            help='Audio file to be recognized')
        self.parser.add_argument(
            '--sample_rate', type=int, default=16000, help='audio sample rate')
        self.parser.add_argument(
            '--lang', type=str, default="zh_cn", help='language')
        self.parser.add_argument(
            '--audio_format', type=str, default="wav", help='audio format')

    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        url = 'http://' + args.server_ip + ":" + str(
            args.port) + '/paddlespeech/asr'
        audio = wav2base64(args.input)
        data = {
            "audio": audio,
            "audio_format": args.audio_format,
            "sample_rate": args.sample_rate,
            "lang": args.lang,
        }
        time_start = time.time()
        try:
            r = requests.post(url=url, data=json.dumps(data))
            # ending Timestamp
            time_end = time.time()
            logger.info(r.json())
            logger.info("time cost %f s." % (time_end - time_start))
            return True
        except:
            logger.error("Failed to speech recognition.")
            return False

    @stats_wrapper
    def __call__(self,
                 input: str,
                 server_ip: str="127.0.0.1",
                 port: int=8090,
                 sample_rate: int=16000,
                 lang: str="zh_cn",
                 audio_format: str="wav"):
        """
        Python API to call an executor.
        """

        url = 'http://' + server_ip + ":" + str(port) + '/paddlespeech/asr'
        audio = wav2base64(input)
        data = {
            "audio": audio,
            "audio_format": audio_format,
            "sample_rate": sample_rate,
            "lang": lang,
        }
        time_start = time.time()
        try:
            r = requests.post(url=url, data=json.dumps(data))
            # ending Timestamp
            time_end = time.time()
            print(r.json())
            print("time cost %f s." % (time_end - time_start))
        except:
            print("Failed to speech recognition.")