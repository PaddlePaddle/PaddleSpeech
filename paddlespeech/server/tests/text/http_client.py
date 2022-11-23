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
import json
import time

import requests

from paddlespeech.cli.log import logger


# Request and response
def text_client(args):
    """ Request and response
    Args:
        text: A sentence to be processed by PaddleSpeech Text Server
        outfile: The punctuation text
    """
    url = "http://" + str(args.server) + ":" + str(
        args.port) + "/paddlespeech/text"
    request = {
        "text": args.text,
    }

    response = requests.post(url, json.dumps(request))
    response_dict = response.json()
    punc_text = response_dict["result"]["punc_text"]

    # transform audio
    outfile = args.output
    if outfile:
        with open(outfile, 'w') as w:
            w.write(punc_text + "\n")

    logger.info(f"The punc text is: {punc_text}")
    return punc_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text',
        type=str,
        default="今天的天气真不错啊你下午有空吗我想约你一起去吃饭",
        help='A sentence to be synthesized')
    parser.add_argument(
        '--output', type=str, default="./punc_text", help='Punc text file')
    parser.add_argument(
        "--server", type=str, help="server ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="server port", default=8090)
    args = parser.parse_args()

    st = time.time()
    try:
        punc_text = text_client(args)
        time_consume = time.time() - st
        time_per_word = time_consume / len(args.text)
        print("Text Process successfully.")
        print("Inference time: %f" % (time_consume))
        print("The text length: %f" % (len(args.text)))
        print("The time per work is: %f" % (time_per_word))
    except BaseException as e:
        logger.info("Failed to Process text.")
        logger.info(e)
