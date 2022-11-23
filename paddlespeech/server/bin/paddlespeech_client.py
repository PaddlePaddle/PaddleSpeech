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
import asyncio
import base64
import io
import json
import os
import random
import sys
import time
import warnings
from typing import List

import numpy as np
import requests
import soundfile

from ..executor import BaseExecutor
from ..util import cli_client_register
from ..util import stats_wrapper
from paddlespeech.cli.log import logger
from paddlespeech.server.utils.audio_handler import ASRWsAudioHandler
from paddlespeech.server.utils.audio_process import wav2pcm
from paddlespeech.server.utils.util import compute_delay
from paddlespeech.server.utils.util import wav2base64
warnings.filterwarnings("ignore")

__all__ = [
    'TTSClientExecutor', 'TTSOnlineClientExecutor', 'ASRClientExecutor',
    'ASROnlineClientExecutor', 'CLSClientExecutor', 'VectorClientExecutor'
]


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
            default=None,
            help='Text to be synthesized.',
            required=True)
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
            '--output', type=str, default=None, help='Synthesized audio file')

    def postprocess(self, wav_base64: str, outfile: str) -> float:
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
            os.remove(temp_wav)
        else:
            logger.error("The format for saving audio only supports wav or pcm")

    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        input_ = args.input
        server_ip = args.server_ip
        port = args.port
        spk_id = args.spk_id
        speed = args.speed
        volume = args.volume
        sample_rate = args.sample_rate
        output = args.output

        try:
            time_start = time.time()
            res = self(
                input=input_,
                server_ip=server_ip,
                port=port,
                spk_id=spk_id,
                speed=speed,
                volume=volume,
                sample_rate=sample_rate,
                output=output)
            time_end = time.time()
            time_consume = time_end - time_start
            response_dict = res.json()
            logger.info("Save synthesized audio successfully on %s." % (output))
            logger.info("Audio duration: %f s." %
                        (response_dict['result']['duration']))
            logger.info("Response time: %f s." % (time_consume))
            return True
        except Exception as e:
            logger.error("Failed to synthesized audio.")
            logger.error(e)
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
                 output: str=None):
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

        res = requests.post(url, json.dumps(request))
        response_dict = res.json()
        if output is not None:
            self.postprocess(response_dict["result"]["audio"], output)
        return res


@cli_client_register(
    name='paddlespeech_client.tts_online',
    description='visit tts online service')
class TTSOnlineClientExecutor(BaseExecutor):
    def __init__(self):
        super(TTSOnlineClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech_client.tts_online', add_help=True)
        self.parser.add_argument(
            '--server_ip', type=str, default='127.0.0.1', help='server ip')
        self.parser.add_argument(
            '--port', type=int, default=8092, help='server port')
        self.parser.add_argument(
            '--protocol',
            type=str,
            default="http",
            choices=["http", "websocket"],
            help='server protocol')
        self.parser.add_argument(
            '--input',
            type=str,
            default=None,
            help='Text to be synthesized.',
            required=True)
        self.parser.add_argument(
            '--spk_id', type=int, default=0, help='Speaker id')
        self.parser.add_argument(
            '--output',
            type=str,
            default=None,
            help='Client saves synthesized audio')
        self.parser.add_argument(
            "--play", type=bool, help="whether to play audio", default=False)

    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        input_ = args.input
        server_ip = args.server_ip
        port = args.port
        protocol = args.protocol
        spk_id = args.spk_id
        output = args.output
        play = args.play

        try:
            self(
                input=input_,
                server_ip=server_ip,
                port=port,
                protocol=protocol,
                spk_id=spk_id,
                output=output,
                play=play)
            return True
        except Exception as e:
            logger.error("Failed to synthesized audio.")
            logger.error(e)
            return False

    @stats_wrapper
    def __call__(self,
                 input: str,
                 server_ip: str="127.0.0.1",
                 port: int=8092,
                 protocol: str="http",
                 spk_id: int=0,
                 output: str=None,
                 play: bool=False):
        """
        Python API to call an executor.
        """

        if protocol == "http":
            logger.info("tts http client start")
            from paddlespeech.server.utils.audio_handler import TTSHttpHandler
            handler = TTSHttpHandler(server_ip, port, play)
            first_response, final_response, duration, save_audio_success, receive_time_list, chunk_duration_list = handler.run(
                input, spk_id, output)
            delay_time_list = compute_delay(receive_time_list,
                                            chunk_duration_list)

        elif protocol == "websocket":
            from paddlespeech.server.utils.audio_handler import TTSWsHandler
            logger.info("tts websocket client start")
            handler = TTSWsHandler(server_ip, port, play)
            loop = asyncio.get_event_loop()
            first_response, final_response, duration, save_audio_success, receive_time_list, chunk_duration_list = loop.run_until_complete(
                handler.run(input, spk_id, output))
            delay_time_list = compute_delay(receive_time_list,
                                            chunk_duration_list)

        else:
            logger.error("Please set correct protocol, http or websocket")
            sys.exit(-1)

        logger.info(f"sentence: {input}")
        logger.info(f"duration: {duration} s")
        logger.info(f"first response: {first_response} s")
        logger.info(f"final response: {final_response} s")
        logger.info(f"RTF: {final_response/duration}")
        if output is not None:
            if save_audio_success:
                logger.info(f"Audio successfully saved in {output}")
            else:
                logger.error("Audio save failed.")

        if delay_time_list != []:
            logger.info(
                f"Delay situation: total number of packages: {len(receive_time_list)}, the number of delayed packets: {len(delay_time_list)}, minimum delay time: {min(delay_time_list)} s, maximum delay time: {max(delay_time_list)} s, average delay time: {sum(delay_time_list)/len(delay_time_list)} s, delay rate:{len(delay_time_list)/len(receive_time_list)}"
            )
        else:
            logger.info("The sentence has no delay in streaming synthesis.")


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
            default=None,
            help='Audio file to be recognized',
            required=True)
        self.parser.add_argument(
            '--protocol',
            type=str,
            default="http",
            choices=["http", "websocket"],
            help='server protocol')
        self.parser.add_argument(
            '--sample_rate', type=int, default=16000, help='audio sample rate')
        self.parser.add_argument(
            '--lang', type=str, default="zh_cn", help='language')
        self.parser.add_argument(
            '--audio_format', type=str, default="wav", help='audio format')

        self.parser.add_argument(
            '--punc.server_ip',
            type=str,
            default=None,
            dest="punc_server_ip",
            help='Punctuation server ip')
        self.parser.add_argument(
            '--punc.port',
            type=int,
            default=8091,
            dest="punc_server_port",
            help='Punctuation server port')

    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        input_ = args.input
        server_ip = args.server_ip
        port = args.port
        sample_rate = args.sample_rate
        lang = args.lang
        audio_format = args.audio_format
        protocol = args.protocol

        try:
            time_start = time.time()
            res = self(
                input=input_,
                server_ip=server_ip,
                port=port,
                sample_rate=sample_rate,
                lang=lang,
                audio_format=audio_format,
                protocol=protocol,
                punc_server_ip=args.punc_server_ip,
                punc_server_port=args.punc_server_port)
            time_end = time.time()
            logger.info(f"ASR result: {res}")
            logger.info("Response time %f s." % (time_end - time_start))
            return True
        except Exception as e:
            logger.error("Failed to speech recognition.")
            logger.error(e)
            return False

    @stats_wrapper
    def __call__(self,
                 input: str,
                 server_ip: str="127.0.0.1",
                 port: int=8090,
                 sample_rate: int=16000,
                 lang: str="zh_cn",
                 audio_format: str="wav",
                 protocol: str="http",
                 punc_server_ip: str=None,
                 punc_server_port: int=None):
        """Python API to call an executor.

        Args:
            input (str): The input audio file path
            server_ip (str, optional): The ASR server ip. Defaults to "127.0.0.1".
            port (int, optional): The ASR server port. Defaults to 8090.
            sample_rate (int, optional): The audio sample rate. Defaults to 16000.
            lang (str, optional): The audio language type. Defaults to "zh_cn".
            audio_format (str, optional): The audio format information. Defaults to "wav".
            protocol (str, optional): The ASR server. Defaults to "http".

        Returns:
            str: The ASR results
        """
        # we use the asr server to recognize the audio text content
        # and paddlespeech_client asr only support http protocol
        protocol = "http"
        if protocol.lower() == "http":
            from paddlespeech.server.utils.audio_handler import ASRHttpHandler
            logger.info("asr http client start")
            handler = ASRHttpHandler(server_ip=server_ip, port=port)
            res = handler.run(input, audio_format, sample_rate, lang)
            res = res['result']['transcription']
            logger.info("asr http client finished")
        else:
            logger.error(f"Sorry, we have not support protocol: {protocol},"
                         "please use http or websocket protocol")
            sys.exit(-1)

        return res


@cli_client_register(
    name='paddlespeech_client.asr_online',
    description='visit asr online service')
class ASROnlineClientExecutor(BaseExecutor):
    def __init__(self):
        super(ASROnlineClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech_client.asr_online', add_help=True)
        self.parser.add_argument(
            '--server_ip', type=str, default='127.0.0.1', help='server ip')
        self.parser.add_argument(
            '--port', type=int, default=8091, help='server port')
        self.parser.add_argument(
            '--input',
            type=str,
            default=None,
            help='Audio file to be recognized',
            required=True)
        self.parser.add_argument(
            '--sample_rate', type=int, default=16000, help='audio sample rate')
        self.parser.add_argument(
            '--lang', type=str, default="zh_cn", help='language')
        self.parser.add_argument(
            '--audio_format', type=str, default="wav", help='audio format')
        self.parser.add_argument(
            '--punc.server_ip',
            type=str,
            default=None,
            dest="punc_server_ip",
            help='Punctuation server ip')
        self.parser.add_argument(
            '--punc.port',
            type=int,
            default=8190,
            dest="punc_server_port",
            help='Punctuation server port')

    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        input_ = args.input
        server_ip = args.server_ip
        port = args.port
        sample_rate = args.sample_rate
        lang = args.lang
        audio_format = args.audio_format
        try:
            time_start = time.time()
            res = self(
                input=input_,
                server_ip=server_ip,
                port=port,
                sample_rate=sample_rate,
                lang=lang,
                audio_format=audio_format,
                punc_server_ip=args.punc_server_ip,
                punc_server_port=args.punc_server_port)
            time_end = time.time()
            logger.info(res)
            logger.info("Response time %f s." % (time_end - time_start))
            return True
        except Exception as e:
            logger.error("Failed to speech recognition.")
            logger.error(e)
            return False

    @stats_wrapper
    def __call__(self,
                 input: str,
                 server_ip: str="127.0.0.1",
                 port: int=8091,
                 sample_rate: int=16000,
                 lang: str="zh_cn",
                 audio_format: str="wav",
                 punc_server_ip: str=None,
                 punc_server_port: str=None):
        """Python API to call asr online executor.

        Args:
            input (str): the audio file to be send to streaming asr service.
            server_ip (str, optional): streaming asr server ip. Defaults to "127.0.0.1".
            port (int, optional): streaming asr server port. Defaults to 8091.
            sample_rate (int, optional): audio sample rate. Defaults to 16000.
            lang (str, optional): audio language type. Defaults to "zh_cn".
            audio_format (str, optional): audio format. Defaults to "wav".
            punc_server_ip (str, optional): punctuation server ip. Defaults to None.
            punc_server_port (str, optional): punctuation server port. Defaults to None.

        Returns:
            str: the audio text
        """

        logger.info("asr websocket client start")
        handler = ASRWsAudioHandler(
            server_ip,
            port,
            punc_server_ip=punc_server_ip,
            punc_server_port=punc_server_port)
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(handler.run(input))
        logger.info("asr websocket client finished")

        return res['result']


@cli_client_register(
    name='paddlespeech_client.cls', description='visit cls service')
class CLSClientExecutor(BaseExecutor):
    def __init__(self):
        super(CLSClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech_client.cls', add_help=True)
        self.parser.add_argument(
            '--server_ip', type=str, default='127.0.0.1', help='server ip')
        self.parser.add_argument(
            '--port', type=int, default=8090, help='server port')
        self.parser.add_argument(
            '--input',
            type=str,
            default=None,
            help='Audio file to classify.',
            required=True)
        self.parser.add_argument(
            '--topk',
            type=int,
            default=1,
            help='Return topk scores of classification result.')

    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        input_ = args.input
        server_ip = args.server_ip
        port = args.port
        topk = args.topk

        try:
            time_start = time.time()
            res = self(input=input_, server_ip=server_ip, port=port, topk=topk)
            time_end = time.time()
            logger.info(res.json())
            logger.info("Response time %f s." % (time_end - time_start))
            return True
        except Exception as e:
            logger.error("Failed to speech classification.")
            logger.error(e)
            return False

    @stats_wrapper
    def __call__(self,
                 input: str,
                 server_ip: str="127.0.0.1",
                 port: int=8090,
                 topk: int=1):
        """
        Python API to call an executor.
        """
        url = 'http://' + server_ip + ":" + str(port) + '/paddlespeech/cls'
        audio = wav2base64(input)
        data = {"audio": audio, "topk": topk}
        res = requests.post(url=url, data=json.dumps(data))
        return res


@cli_client_register(
    name='paddlespeech_client.text', description='visit the text service')
class TextClientExecutor(BaseExecutor):
    def __init__(self):
        super(TextClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech_client.text', add_help=True)
        self.parser.add_argument(
            '--server_ip', type=str, default='127.0.0.1', help='server ip')
        self.parser.add_argument(
            '--port', type=int, default=8090, help='server port')
        self.parser.add_argument(
            '--input',
            type=str,
            default=None,
            help='sentence to be process by text server.',
            required=True)

    def execute(self, argv: List[str]) -> bool:
        """Execute the request from the argv.

        Args:
            argv (List): the request arguments

        Returns:
            str: the request flag
        """
        args = self.parser.parse_args(argv)
        input_ = args.input
        server_ip = args.server_ip
        port = args.port

        try:
            time_start = time.time()
            res = self(input=input_, server_ip=server_ip, port=port)
            time_end = time.time()
            logger.info(f"The punc text: {res}")
            logger.info("Response time %f s." % (time_end - time_start))
            return True
        except Exception as e:
            logger.error("Failed to Text punctuation.")
            return False

    @stats_wrapper
    def __call__(self, input: str, server_ip: str="127.0.0.1", port: int=8090):
        """
        Python API to call text executor.

        Args:
            input (str): the request sentence text
            server_ip (str, optional): the server ip. Defaults to "127.0.0.1".
            port (int, optional): the server port. Defaults to 8090.

        Returns:
            str: the punctuation text
        """

        url = 'http://' + server_ip + ":" + str(port) + '/paddlespeech/text'
        request = {
            "text": input,
        }

        res = requests.post(url=url, data=json.dumps(request))
        response_dict = res.json()
        punc_text = response_dict["result"]["punc_text"]
        return punc_text


@cli_client_register(
    name='paddlespeech_client.vector', description='visit the vector service')
class VectorClientExecutor(BaseExecutor):
    def __init__(self):
        super(VectorClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech_client.vector', add_help=True)
        self.parser.add_argument(
            '--server_ip', type=str, default='127.0.0.1', help='server ip')
        self.parser.add_argument(
            '--port', type=int, default=8090, help='server port')
        self.parser.add_argument(
            '--input',
            type=str,
            default=None,
            help='sentence to be process by text server.')
        self.parser.add_argument(
            '--task',
            type=str,
            default="spk",
            choices=["spk", "score"],
            help="The vector service task")
        self.parser.add_argument(
            "--enroll", type=str, default=None, help="The enroll audio")
        self.parser.add_argument(
            "--test", type=str, default=None, help="The test audio")

    def execute(self, argv: List[str]) -> bool:
        """Execute the request from the argv.

        Args:
            argv (List): the request arguments

        Returns:
            str: the request flag
        """
        args = self.parser.parse_args(argv)
        input_ = args.input
        server_ip = args.server_ip
        port = args.port
        task = args.task

        try:
            time_start = time.time()
            res = self(
                input=input_,
                server_ip=server_ip,
                port=port,
                enroll_audio=args.enroll,
                test_audio=args.test,
                task=task)
            time_end = time.time()
            logger.info(res.json())
            logger.info("Response time %f s." % (time_end - time_start))
            return True
        except Exception as e:
            logger.error("Failed to extract vector.")
            logger.error(e)
            return False

    @stats_wrapper
    def __call__(self,
                 input: str,
                 server_ip: str="127.0.0.1",
                 port: int=8090,
                 audio_format: str="wav",
                 sample_rate: int=16000,
                 enroll_audio: str=None,
                 test_audio: str=None,
                 task="spk"):
        """
        Python API to call text executor.

        Args:
            input (str): the request audio data
            server_ip (str, optional): the server ip. Defaults to "127.0.0.1".
            port (int, optional): the server port. Defaults to 8090.
            audio_format (str, optional): audio format. Defaults to "wav".
            sample_rate (str, optional): audio sample rate. Defaults to 16000.
            enroll_audio (str, optional): enroll audio data. Defaults to None.
            test_audio (str, optional): test audio data. Defaults to None.
            task (str, optional): the task type, "spk" or "socre". Defaults to "spk"
        Returns:
            str: the audio embedding or score between enroll and test audio
        """

        if task == "spk":
            from paddlespeech.server.utils.audio_handler import VectorHttpHandler
            logger.info("vector http client start")
            logger.info(f"the input audio: {input}")
            handler = VectorHttpHandler(server_ip=server_ip, port=port)
            res = handler.run(input, audio_format, sample_rate)
            return res
        elif task == "score":
            from paddlespeech.server.utils.audio_handler import VectorScoreHttpHandler
            logger.info("vector score http client start")
            logger.info(
                f"enroll audio: {enroll_audio}, test audio: {test_audio}")
            handler = VectorScoreHttpHandler(server_ip=server_ip, port=port)
            res = handler.run(enroll_audio, test_audio, audio_format,
                              sample_rate)
            return res
        else:
            logger.error(f"Sorry, we have not support such task {task}")


@cli_client_register(
    name='paddlespeech_client.acs', description='visit acs service')
class ACSClientExecutor(BaseExecutor):
    def __init__(self):
        super(ACSClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='paddlespeech_client.acs', add_help=True)
        self.parser.add_argument(
            '--server_ip', type=str, default='127.0.0.1', help='server ip')
        self.parser.add_argument(
            '--port', type=int, default=8090, help='server port')
        self.parser.add_argument(
            '--input',
            type=str,
            default=None,
            help='Audio file to be recognized',
            required=True)
        self.parser.add_argument(
            '--sample_rate', type=int, default=16000, help='audio sample rate')
        self.parser.add_argument(
            '--lang', type=str, default="zh_cn", help='language')
        self.parser.add_argument(
            '--audio_format', type=str, default="wav", help='audio format')

    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        input_ = args.input
        server_ip = args.server_ip
        port = args.port
        sample_rate = args.sample_rate
        lang = args.lang
        audio_format = args.audio_format

        try:
            time_start = time.time()
            res = self(
                input=input_,
                server_ip=server_ip,
                port=port,
                sample_rate=sample_rate,
                lang=lang,
                audio_format=audio_format, )
            time_end = time.time()
            logger.info(f"ACS result: {res}")
            logger.info("Response time %f s." % (time_end - time_start))
            return True
        except Exception as e:
            logger.error("Failed to speech recognition.")
            logger.error(e)
            return False

    @stats_wrapper
    def __call__(
            self,
            input: str,
            server_ip: str="127.0.0.1",
            port: int=8090,
            sample_rate: int=16000,
            lang: str="zh_cn",
            audio_format: str="wav", ):
        """Python API to call an executor.

        Args:
            input (str): The input audio file path
            server_ip (str, optional): The ASR server ip. Defaults to "127.0.0.1".
            port (int, optional): The ASR server port. Defaults to 8090.
            sample_rate (int, optional): The audio sample rate. Defaults to 16000.
            lang (str, optional): The audio language type. Defaults to "zh_cn".
            audio_format (str, optional): The audio format information. Defaults to "wav".

        Returns:
            str: The ACS results
        """
        # we use the acs server to get the key word time stamp in audio text content
        logger.info("acs http client start")
        from paddlespeech.server.utils.audio_handler import ASRHttpHandler
        handler = ASRHttpHandler(
            server_ip=server_ip, port=port, endpoint="/paddlespeech/asr/search")
        res = handler.run(input, audio_format, sample_rate, lang)
        res = res['result']
        logger.info("acs http client finished")

        return res
