#!/usr/bin/python3
#! coding:utf-8

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

"""
Prepare VoxCeleb1 dataset

Download, unpack and create manifest files.
Manifest file is a json-format file with each line containing the meta data 
reference: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
"""

import os
import json
import glob
import random
import shutil
import sys  # noqa F401
import numpy as np
import paddle
import paddleaudio
import argparse

# prefix defined variable
# 预定义一些变量
DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')
SAMPLERATE = 16000      # default: sample rate = 16000
split_ratio = [90, 10]

# prefix defined argument
# 预定义参数
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
        "--data-dir",
        default=DATA_HOME + "/VoxCeleb1",
        type=str,
        help="Directory to store the data. (default: %(default)s)")
parser.add_argument(
        "--split-speaker",
        action='store_true',
        help="split the all dataset by speakers")
parser.add_argument(
        "--seg-dur",
        default=3.0,
        type=float,
        help="segment duration, seconds")
parser.add_argument("--amp-th",
        default=5e-04,
        type=float,
        help="minimum utterance duration")

args = parser.parse_args()

def create_manifest():
    data_dir = args.data_dir
    print("Voxceleb1 data directory is {} ... ".format(data_dir))

    # 获取整个 voxceleb1 数据集的内容
    spk2utt, utt2spk, dataset = get_dataset_utterances(data_dir)

    # 对数据集进行切分，得到训练集和开发集
    train_dataset, dev_dataset = split_utt_dataset(utt2spk, spk2utt)

    # 获取音频的片段数据
    train_json = os.path.join("data/train", "data.json")
    train_dataset = prepare_dataset_json(dataset, train_dataset)
    with open(train_json, 'w') as f:
        f.write(train_dataset)
    
    dev_json = os.path.join("data/dev/data.json")
    dev_dataset = prepare_dataset_json(dataset, dev_dataset)
    with open (dev_json, 'w') as f:
        f.write(dev_dataset)

def split_utt_dataset(utt2spk, spk2utt):
    if args.split_speaker:
        return split_spkers_dataset(spk2utt, split_ratio)
    else:
        return split_utterances_dataset(utt2spk, split_ratio)

def split_spkers_dataset(spk2utt, splits):
    train_spks_num = int(len(spk2utt.keys()) * 0.01 * int(splits[0]))
    train_utts_dataset = []
    
    for spk in list(spk2utt.keys())[:train_spks_num]:
        train_utts_dataset.extend(spk2utt[spk])

    dev_utts_dataset = []
    for spk in list(spk2utt.keys())[train_spks_num:]:
        dev_utts_dataset.extend(spk2utt[spk])

    return train_utts_dataset, dev_utts_dataset

def split_utterances_dataset(utt2spk, splits):
    train_utts_num = int(len(utt2spk.keys()) * 0.01 * int(split_ratio[0]))
    utts = list(utt2spk.keys())
    return utts[:train_utts_num], utts[train_utts_num:]

def get_utterance_info(utt_path):
    utt_info = utt_path.split("/")[-3:]
    utt_name = utt_info[-1].replace(".wav", "")
    utt = {}
    utt["wav"] = utt_path
    utt["spk_id"] = utt_info[0]

    # 获取音频的长度
    signal, fs = paddleaudio.load(utt_path)
    audio_duration = signal.shape[1] / SAMPLERATE
    utt["duration"] = audio_duration
    utt["start"] = 0
    utt["end"] = audio_duration
    utt["text"] = ""
    utt_name = "_".join([utt["spk_id"], utt_name])

    return utt_name, utt

def get_dataset_utterances(data_path):
    data_path = os.path.join(data_path, "wav", "**", "*.wav")
    spk2utt = {}
    utt2spk = {}
    voxceleb1 = {}
    for utt in glob.glob(data_path, recursive=True):
        spk_id = utt.split("/wav/")[1].split("/")[0]
        utt_name, utt = get_utterance_info(utt)
        spk2utt.setdefault(spk_id, []).append(utt_name)
        utt2spk[utt_name] = spk_id
        voxceleb1[utt_name] = utt
    voxceleb1 = json.dumps(voxceleb1, indent=4, sort_keys=False)

    return spk2utt, utt2spk, voxceleb1


def prepare_dataset_json(dataset, utt_list):
    utts_json = {}
    dataset = json.loads(dataset)
    for utt_id in utt_list:
        utt_info = dataset[utt_id]
        segments = get_chunks(utt_id, utt_info)
        utts_json.update(segments)
    utts_json = json.dumps(utts_json, indent=4, sort_keys=False)
    return utts_json

def get_chunks(utt_name, utt):
    seg_dur = args.seg_dur
    duration = utt["duration"]
    segments = {}
    if duration < args.amp_th:
        return segments

    num_chunks = max(int(duration / seg_dur), 1)
    for idx in range(num_chunks):
        seg_start = idx * seg_dur
        seg_end = min(duration, seg_start + seg_dur)
        seg_name = "_".join([utt_name, str(seg_start), str(seg_end)])
        segments[seg_name] = utt
        segments[seg_name]["start"] = seg_start
        segments[seg_name]["end"] = seg_end

    return segments

if __name__ == "__main__":
    """
    VoxCele1 dataset prepare script
    you need change the train or dev split ratio by yourself
    需要手动修改训练集和开发集的比例，第一个为train的比例，第二个为dev的比例
    """
    paddle.device.set_device("cpu")
    create_manifest()