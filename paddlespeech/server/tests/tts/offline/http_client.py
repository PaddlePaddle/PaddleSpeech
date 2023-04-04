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

import numpy as np
import requests
import soundfile

from paddlespeech.server.utils.audio_process import wav2pcm


# Request and response
def tts_client(args):
    """ Request and response
    Args:
        text: A sentence to be synthesized
        outfile: Synthetic audio file
    """
    url = "http://" + str(args.server) + ":" + str(
        args.port) + "/paddlespeech/tts"
    request = {
        "text": args.text,
        "spk_id": args.spk_id,
        "speed": args.speed,
        "volume": args.volume,
        "sample_rate": args.sample_rate,
        "save_path": args.output
    }

    response = requests.post(url, json.dumps(request))
    response_dict = response.json()
    wav_base64 = response_dict["result"]["audio"]

    audio_data_byte = base64.b64decode(wav_base64)
    # from byte
    samples, sample_rate = soundfile.read(io.BytesIO(audio_data_byte),
                                          dtype='float32')

    # transform audio
    outfile = args.output
    if outfile.endswith(".wav"):
        soundfile.write(outfile, samples, sample_rate)
    elif outfile.endswith(".pcm"):
        temp_wav = str(random.getrandbits(128)) + ".wav"
        soundfile.write(temp_wav, samples, sample_rate)
        wav2pcm(temp_wav, outfile, data_type=np.int16)
        os.remove(temp_wav)
    else:
        print("The format for saving audio only supports wav or pcm")

    return len(samples), sample_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text',
                        type=str,
                        default="您好，欢迎使用语音合成服务。",
                        help='A sentence to be synthesized')
    parser.add_argument('--spk_id', type=int, default=0, help='Speaker id')
    parser.add_argument('--speed', type=float, default=1.0, help='Audio speed')
    parser.add_argument('--volume',
                        type=float,
                        default=1.0,
                        help='Audio volume')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=0,
        help='Sampling rate, the default is the same as the model')
    parser.add_argument('--output',
                        type=str,
                        default="./out.wav",
                        help='Synthesized audio file')
    parser.add_argument("--server",
                        type=str,
                        help="server ip",
                        default="127.0.0.1")
    parser.add_argument("--port", type=int, help="server port", default=8090)
    args = parser.parse_args()

    st = time.time()
    try:
        samples_length, sample_rate = tts_client(args)
        time_consume = time.time() - st
        duration = samples_length / sample_rate
        rtf = time_consume / duration
        print("Synthesized audio successfully.")
        print("Inference time: %f" % (time_consume))
        print("The duration of synthesized audio: %f" % (duration))
        print("The RTF is: %f" % (rtf))
    except BaseException:
        print("Failed to synthesized audio.")
