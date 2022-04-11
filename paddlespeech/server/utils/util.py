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
import math


def wav2base64(wav_file: str):
    """
    read wave file and covert to base64 string
    """
    with open(wav_file, 'rb') as f:
        base64_bytes = base64.b64encode(f.read())
        base64_string = base64_bytes.decode('utf-8')
    return base64_string


def base64towav(base64_string: str):
    pass


def self_check():
    """ self check resource
    """
    return True


def denorm(data, mean, std):
    """stream am model need to denorm
    """
    return data * std + mean


def get_chunks(data, block_size, pad_size, step):
    """Divide data into multiple chunks

    Args:
        data (tensor): data
        block_size (int): [description]
        pad_size (int): [description]
        step (str): set "am" or "voc", generate chunk for step am or vocoder(voc)

    Returns:
        list: chunks list
    """
    if step == "am":
        data_len = data.shape[1]
    elif step == "voc":
        data_len = data.shape[0]
    else:
        print("Please set correct type to get chunks, am or voc")

    chunks = []
    n = math.ceil(data_len / block_size)
    for i in range(n):
        start = max(0, i * block_size - pad_size)
        end = min((i + 1) * block_size + pad_size, data_len)
        if step == "am":
            chunks.append(data[:, start:end, :])
        elif step == "voc":
            chunks.append(data[start:end, :])
        else:
            print("Please set correct type to get chunks, am or voc")
    return chunks
