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
import argparse

from paddlespeech.cli.log import logger
from paddlespeech.server.bin.paddlespeech_server import ServerExecutor
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='paddlespeech_server.start', add_help=True)
    parser.add_argument(
        "--config_file",
        action="store",
        help="yaml file of the app",
        default=None,
        required=True)

    parser.add_argument(
        "--log_file",
        action="store",
        help="log file",
        default="./log/paddlespeech.log")
    logger.info("start to parse the args")
    args = parser.parse_args()

    logger.info("start to launch the punctuation server")
    punc_server = ServerExecutor()
    punc_server(config_file=args.config_file, log_file=args.log_file)
