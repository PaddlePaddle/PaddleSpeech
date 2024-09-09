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
# See the License for the 
import base64
import json
import time

import requests


def readwav2base64(wav_file):
    """
    read wave file and covert to base64 string
    """
    with open(wav_file, 'rb') as f:
        base64_bytes = base64.b64encode(f.read())
        base64_string = base64_bytes.decode('utf-8')
    return base64_string


def main():
    """
    main func
    """
    url = "http://127.0.0.1:8090/paddlespeech/asr"

    # start Timestamp
    time_start = time.time()

    test_audio_dir = "./16_audio.wav"
    audio = readwav2base64(test_audio_dir)

    data = {
        "audio": audio,
        "audio_format": "wav",
        "sample_rate": 16000,
        "lang": "zh_cn",
    }

    r = requests.post(url=url, data=json.dumps(data))

    # ending Timestamp
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    print(r.json())


if __name__ == "__main__":
    main()
