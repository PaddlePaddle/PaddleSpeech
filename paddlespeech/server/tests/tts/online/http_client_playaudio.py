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
import threading
import time

import pyaudio
import requests

mutex = threading.Lock()
buffer = b''
p = pyaudio.PyAudio()
stream = p.open(
    format=p.get_format_from_width(2), channels=1, rate=24000, output=True)
max_fail = 50


def play_audio():
    global stream
    global buffer
    global max_fail
    while True:
        if not buffer:
            max_fail -= 1
            time.sleep(0.05)
            if max_fail < 0:
                break
        mutex.acquire()
        stream.write(buffer)
        buffer = b''
        mutex.release()


def test(args):
    global mutex
    global buffer
    params = {
        "text": args.text,
        "spk_id": args.spk_id,
        "speed": args.speed,
        "volume": args.volume,
        "sample_rate": args.sample_rate,
        "save_path": ''
    }

    all_bytes = 0.0
    t = threading.Thread(target=play_audio)
    flag = 1
    url = "http://" + str(args.server) + ":" + str(
        args.port) + "/paddlespeech/streaming/tts"
    st = time.time()
    html = requests.post(url, json.dumps(params), stream=True)
    for chunk in html.iter_content(chunk_size=1024):
        mutex.acquire()
        chunk = base64.b64decode(chunk)  # bytes
        buffer += chunk
        mutex.release()
        if flag:
            first_response = time.time() - st
            print(f"首包响应：{first_response} s")
            flag = 0
            t.start()
        all_bytes += len(chunk)

    final_response = time.time() - st
    duration = all_bytes / 2 / 24000

    print(f"尾包响应：{final_response} s")
    print(f"音频时长：{duration} s")
    print(f"RTF: {final_response / duration}")

    t.join()
    stream.stop_stream()
    stream.close()
    p.terminate()


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

    args = parser.parse_args()
    test(args)
