#!/usr/bin/python3
#! coding:utf-8

"""
Generate enroll and test data
"""

import os
import glob
import json
import paddle
import paddleaudio
import argparse
SAMPLERATE = 16000 
def main(args):
    
    enroll_uttlist = set()
    test_uttlist = set()
    with open(args.trial, 'r') as f:
        for line in f:
            _, enroll_utt, test_utt = line.strip().split()
            enroll_uttlist.add(enroll_utt)
            test_uttlist.add(test_utt)
    
    # enroll_uttlist = list(enroll_uttlist)
    # test_uttlist = list(test_uttlist)
    
    enroll_dir = os.path.join(args.dir, "enroll")
    if not os.path.exists(enroll_dir):
        os.mkdir(enroll_dir)
    
    test_dir = os.path.join(args.dir, "test")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data_path = os.path.join(args.voxceleb1, "wav", "**", "*.wav")
    enroll_info = {}
    test_info = {}
    for utt in glob.glob(data_path, recursive=True):
        spk_id = "/".join(utt.strip().split("/")[-3:])
        if spk_id in enroll_uttlist:
            utt_name, utt_info = get_utterance_info(utt)
            utt_info["spk_id"] = spk_id
            enroll_info[spk_id] = utt_info
        
        if spk_id in test_uttlist:
            utt_name, utt_info = get_utterance_info(utt)
            utt_info["spk_id"] = spk_id
            test_info[spk_id] = utt_info
    
    with open(os.path.join(enroll_dir, "data.json"), 'w') as f:
        f.write(json.dumps(enroll_info, indent=4, sort_keys=False))

    with open(os.path.join(test_dir, "data.json"), 'w') as f:
        f.write(json.dumps(test_info, indent=4, sort_keys=False))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default="./data/trial", type=str, help="trial file")
    parser.add_argument("--voxceleb1", default="./voxceleb1", type=str, help="voxceleb root dir")
    parser.add_argument("--dir", default="./data", type=str, help="target directory of enroll and test")
    paddle.device.set_device("cpu")
    args = parser.parse_args()
    main(args)