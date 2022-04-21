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
from paddlespeech.t2s.exps.syn_utils import get_sentences


def save_audio(buffer, audio_path) -> bool:
    if audio_path.endswith("pcm"):
        with open(audio_path, "wb") as f:
            f.write(buffer)
    elif audio_path.endswith("wav"):
        with open("./tmp.pcm", "wb") as f:
            f.write(buffer)
        pcm2wav("./tmp.pcm", audio_path, channels=1, bits=16, sample_rate=24000)
        os.system("rm ./tmp.pcm")
    else:
        print("Only supports saved audio format is pcm or wav")
        return False

    return True


def test(args, text, utt_id):
    params = {
        "text": text,
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

    print(f"sentence: {text}")
    print(f"尾包响应：{final_response} s")
    print(f"音频时长：{duration} s")
    print(f"RTF: {final_response / duration}")

    save_path = str(args.output_dir + "/" + utt_id + ".wav")
    save_audio(buffer, save_path)
    print("音频保存至：", save_path)

    return first_response, final_response, duration


def count_engine(logfile: str="./nohup.out"):
    """For inference on the statistical engine side

    Args:
        logfile (str, optional): server log. Defaults to "./nohup.out".
    """
    first_response_list = []
    final_response_list = []
    duration_list = []

    with open(logfile, "r") as f:
        for line in f.readlines():
            if "- first response time:" in line:
                first_response = float(line.splie(" ")[-2])
                first_response_list.append(first_response)
            elif "- final response time:" in line:
                final_response = float(line.splie(" ")[-2])
                final_response_list.append(final_response)
            elif "- The durations of audio is:" in line:
                duration = float(line.splie(" ")[-2])
                duration_list.append(duration)

    assert (len(first_response_list) == len(final_response_list) and
            len(final_response_list) == len(duration_list))

    avg_first_response = sum(first_response_list) / len(first_response_list)
    avg_final_response = sum(final_response_list) / len(final_response_list)
    avg_duration = sum(duration_list) / len(duration_list)
    RTF = sum(final_response_list) / sum(duration_list)

    print(
        "************************* engine result ***************************************"
    )
    print(
        f"test num: {len(duration_list)}, avg first response: {avg_first_response} s, avg final response: {avg_final_response} s, avg duration: {avg_duration}, RTF: {RTF}"
    )
    print(
        f"min duration: {min(duration_list)} s, max duration: {max(duration_list)} s"
    )
    print(
        f"max first response: {max(first_response_list)} s, min first response: {min(first_response_list)} s"
    )
    print(
        f"max final response: {max(final_response_list)} s, min final response: {min(final_response_list)} s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="../../../../../../paddlespeech/t2s/exps/csmsc_test.txt",
        help="text to synthesize, a 'utt_id sentence' pair per line")
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
        "--output_dir", type=str, default="./output", help="output dir")

    args = parser.parse_args()

    os.system("rm -rf %s" % (args.output_dir))
    os.mkdir(args.output_dir)

    first_response_list = []
    final_response_list = []
    duration_list = []

    sentences = get_sentences(text_file=args.text, lang="zh")
    for utt_id, sentence in sentences:
        first_response, final_response, duration = test(args, sentence, utt_id)
        first_response_list.append(first_response)
        final_response_list.append(final_response)
        duration_list.append(duration)

    assert (len(first_response_list) == len(final_response_list) and
            len(final_response_list) == len(duration_list))

    avg_first_response = sum(first_response_list) / len(first_response_list)
    avg_final_response = sum(final_response_list) / len(final_response_list)
    avg_duration = sum(duration_list) / len(duration_list)
    RTF = sum(final_response_list) / sum(duration_list)

    print(
        "************************* server/client result ***************************************"
    )
    print(
        f"test num: {len(duration_list)}, avg first response: {avg_first_response} s, avg final response: {avg_final_response} s, avg duration: {avg_duration}, RTF: {RTF}"
    )
    print(
        f"min duration: {min(duration_list)} s, max duration: {max(duration_list)} s"
    )
    print(
        f"max first response: {max(first_response_list)} s, min first response: {min(first_response_list)} s"
    )
    print(
        f"max final response: {max(final_response_list)} s, min final response: {min(final_response_list)} s"
    )
