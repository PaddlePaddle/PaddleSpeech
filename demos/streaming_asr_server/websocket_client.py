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
import codecs
import logging
import os

from paddlespeech.cli.log import logger
from paddlespeech.server.utils.audio_handler import ASRWsAudioHandler


def main(args):
    logger.info("asr websocket client start")
    handler = ASRWsAudioHandler(
        args.server_ip,
        args.port,
        endpoint=args.endpoint,
        punc_server_ip=args.punc_server_ip,
        punc_server_port=args.punc_server_port)
    loop = asyncio.get_event_loop()

    # support to process single audio file
    if args.wavfile and os.path.exists(args.wavfile):
        logger.info(f"start to process the wavscp: {args.wavfile}")
        result = loop.run_until_complete(handler.run(args.wavfile))
        result = result["result"]
        logger.info(f"asr websocket client finished : {result}")

    # support to process batch audios from wav.scp 
    if args.wavscp and os.path.exists(args.wavscp):
        logging.info(f"start to process the wavscp: {args.wavscp}")
        with codecs.open(args.wavscp, 'r', encoding='utf-8') as f,\
             codecs.open("result.txt", 'w', encoding='utf-8') as w:
            for line in f:
                utt_name, utt_path = line.strip().split()
                result = loop.run_until_complete(handler.run(utt_path))
                result = result["result"]
                w.write(f"{utt_name} {result}\n")


if __name__ == "__main__":
    logger.info("Start to do streaming asr client")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--server_ip', type=str, default='127.0.0.1', help='server ip')
    parser.add_argument('--port', type=int, default=8090, help='server port')
    parser.add_argument(
        '--punc.server_ip',
        type=str,
        default=None,
        dest="punc_server_ip",
        help='Punctuation server ip')
    parser.add_argument(
        '--punc.port',
        type=int,
        default=8091,
        dest="punc_server_port",
        help='Punctuation server port')
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/paddlespeech/asr/streaming",
        help="ASR websocket endpoint")
    parser.add_argument(
        "--wavfile",
        action="store",
        help="wav file path ",
        default="./16_audio.wav")
    parser.add_argument(
        "--wavscp", type=str, default=None, help="The batch audios dict text")
    args = parser.parse_args()

    main(args)
