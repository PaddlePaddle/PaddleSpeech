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

from paddlespeech.cli.log import logger
from paddlespeech.server.utils.audio_handler import ASRHttpHandler


def main(args):
    logger.info("asr http client start")
    audio_format = "wav"
    sample_rate = 16000
    lang = "zh"
    handler = ASRHttpHandler(server_ip=args.server_ip,
                             port=args.port,
                             endpoint=args.endpoint)
    res = handler.run(args.wavfile, audio_format, sample_rate, lang)
    # res = res['result']
    logger.info(f"the final result: {res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="audio content search client")
    parser.add_argument('--server_ip',
                        type=str,
                        default='127.0.0.1',
                        help='server ip')
    parser.add_argument('--port', type=int, default=8090, help='server port')
    parser.add_argument("--wavfile",
                        action="store",
                        help="wav file path ",
                        default="./16_audio.wav")
    parser.add_argument('--endpoint',
                        type=str,
                        default='/paddlespeech/asr/search',
                        help='server endpoint')
    args = parser.parse_args()

    main(args)
