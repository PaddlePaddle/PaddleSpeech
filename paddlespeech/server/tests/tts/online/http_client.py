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
import json
import os
import time

import requests

from paddlespeech.server.utils.audio_process import pcm2wav


def save_audio(buffer, audio_path) -> bool:
    if args.save_path.endswith("pcm"):
        with open(args.save_path, "wb") as f:
            f.write(buffer)
    elif args.save_path.endswith("wav"):
        with open("./tmp.pcm", "wb") as f:
            f.write(buffer)
        pcm2wav("./tmp.pcm", audio_path, channels=1, bits=16, sample_rate=24000)
        os.system("rm ./tmp.pcm")
    else:
        print("Only supports saved audio format is pcm or wav")
        return False

    return True


def test(args):
    params = {
        "text": args.text,
        "spk_id": args.spk_id,
        "speed": args.speed,
        "volume": args.volume,
        "sample_rate": args.sample_rate,
        "save_path": ''
    }

    buffer = b''
    flag = 1
    url = "http://" + str(args.server) + ":" + str(
        args.port) + "/paddlespeech/streaming/tts"
    st = time.time()
    html = requests.post(url, json.dumps(params), stream=True)
    for chunk in html.iter_content(chunk_size=1024):
        chunk = base64.b64decode(chunk)  # bytes
        if flag:
            first_response = time.time() - st
            print(f"首包响应：{first_response} s")
            flag = 0
        buffer += chunk

    final_response = time.time() - st
    duration = len(buffer) / 2.0 / 24000

    print(f"尾包响应：{final_response} s")
    print(f"音频时长：{duration} s")
    print(f"RTF: {final_response / duration}")

    if args.save_path is not None:
        if save_audio(buffer, args.save_path):
            print("音频保存至：", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text',
        type=str,
        default="您好，欢迎使用语音合成服务。",
        help='A sentence to be synthesized')
    parser.add_argument('--spk_id', type=int, default=0, help='Speaker id')
    parser.add_argument('--speed', type=float, default=1.0, help='Audio speed')
    parser.add_argument(
        '--volume', type=float, default=1.0, help='Audio volume')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=0,
        help='Sampling rate, the default is the same as the model')
    parser.add_argument(
        "--server", type=str, help="server ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="server port", default=8092)
    parser.add_argument(
        "--save_path", type=str, help="save audio path", default=None)

    args = parser.parse_args()
    test(args)
