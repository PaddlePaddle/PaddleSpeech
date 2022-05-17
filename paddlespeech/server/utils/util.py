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

    if block_size == -1:
        return [data]

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


def compute_delay(receive_time_list, chunk_duration_list):
    """compute delay 
        Args:
            receive_time_list (list): Time to receive each packet
            chunk_duration_list (list): The audio duration corresponding to each packet
        Returns:
            [list]: Delay time list
        """
    assert (len(receive_time_list) == len(chunk_duration_list))
    delay_time_list = []
    play_time = receive_time_list[0] + chunk_duration_list[0]
    for i in range(1, len(receive_time_list)):
        receive_time = receive_time_list[i]
        delay_time = receive_time - play_time
        # 有延迟
        if delay_time > 0:
            play_time = play_time + delay_time + chunk_duration_list[i]
            delay_time_list.append(delay_time)
        # 没有延迟
        else:
            play_time = play_time + chunk_duration_list[i]

    return delay_time_list


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
