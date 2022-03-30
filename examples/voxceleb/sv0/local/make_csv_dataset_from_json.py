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
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert the PaddleSpeech jsonline format to csv format
Currently, Speaker Identificaton Training process need csv format.
"""
import argparse
import os
import jsonlines
import collections
import json
import csv
from yacs.config import CfgNode
import tqdm
from paddleaudio import load as load_audio
import random
from paddlespeech.vector.training.seeding import seed_everything
# voxceleb meta info for each training utterance segment
# we extract a segment from a utterance to train 
# and the segment' period is between start and stop time point in the original wav file
# each field in the meta means as follows:
# id: the utterance segment name
# duration: utterance segment time
# wav: utterance file path
# start: start point in the original wav file
# stop: stop point in the original wav file
# spk_id: the utterance segment's speaker name
meta_info = collections.namedtuple(
        'META_INFO', ('id', 'duration', 'wav', 'start', 'stop', 'spk_id'))

def get_chunks(seg_dur, audio_id, audio_duration):
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds
    chunk_lst = [
            audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
            for i in range(num_chunks)
    ]
    return chunk_lst

def prepare_csv(wav_files, output_file, config, split_chunks=True):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    csv_lines = []
    header = ["id", "duration", "wav", "start", "stop", "spk_id"]
    for item in wav_files:
        item = json.loads(item.strip())
        audio_id = item['utt'].replace(".wav", "")
        audio_duration = item['feat_shape'][0]
        wav_file = item['feat']
        spk_id = audio_id.split('-')[0]
        waveform, sr = load_audio(wav_file)
        if split_chunks:
            uniq_chunks_list = get_chunks(config.chunk_duration, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]  # Timestamps of start and end
                start_sample = int(float(s) * sr)
                end_sample = int(float(e) * sr)
                # id, duration, wav, start, stop, spk_id
                csv_lines.append([
                    chunk, audio_duration, wav_file, start_sample, end_sample,
                    spk_id
                ])  
        else:
            csv_lines.append([audio_id, audio_duration, wav_file, 0, waveform.shape[0], spk_id])              

    
    with open(output_file, mode="w") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
        for line in csv_lines:
            csv_writer.writerow(line)

def get_enroll_test_list(filelist, verification_file):
    print(f"verification file: {verification_file}")
    enroll_audios = set()
    test_audios = set()
    with open(verification_file, 'r') as f:
        for line in f:
            _, enroll_file, test_file = line.strip().split(' ')
            enroll_audios.add('-'.join(enroll_file.split('/')))
            test_audios.add('-'.join(test_file.split('/')))
    
    enroll_files = []
    test_files = []
    for item in filelist:
        with open(item, 'r') as f:
            for line in f:
                audio_id = json.loads(line.strip())['utt']
                if audio_id in enroll_audios:
                    enroll_files.append(line)
                if audio_id in test_audios:
                    test_files.append(line)
            
    enroll_files = sorted(enroll_files)
    test_files = sorted(test_files)

    return enroll_files, test_files

def get_train_dev_list(filelist, target_dir, split_ratio):
    if not os.path.exists(os.path.join(target_dir, "meta")):
        os.makedirs(os.path.join(target_dir, "meta"))

    audio_files = []
    speakers = set()
    for item in filelist:
        with open(item, 'r') as f:
            for line in f:
                spk_id = json.loads(line.strip())['utt2spk']
                speakers.add(spk_id)
                audio_files.append(line.strip())
    
    speakers = sorted(speakers)
    with open(os.path.join(target_dir, "meta", "spk_id2label.txt"), 'w') as f:
        for label, spk_id in enumerate(speakers):
            f.write(f'{spk_id} {label}\n')
    split_idx = int(split_ratio * len(audio_files))
    random.shuffle(audio_files)
    train_files, dev_files = audio_files[:split_idx], audio_files[split_idx:]

    return train_files, dev_files
    
def prepare_data(args, config):

    paddle.set_device("cpu")
    seed_everything(config.seed)
    
    enroll_files, test_files = get_enroll_test_list([args.test], verification_file=config.verification_file)  
    prepare_csv(enroll_files, os.path.join(args.target_dir, "csv", "enroll.csv"), config, split_chunks=False)
    prepare_csv(test_files, os.path.join(args.target_dir, "csv", "test.csv"), config, split_chunks=False)
    
    train_files, dev_files = get_train_dev_list(args.train, target_dir=args.target_dir, split_ratio=config.split_ratio)
    prepare_csv(train_files, os.path.join(args.target_dir, "csv", "train.csv"), config)
    prepare_csv(dev_files, os.path.join(args.target_dir, "csv", "dev.csv"), config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        required=True,
        nargs='+',
        help="The jsonline files list for train")
    parser.add_argument(
        "--test", required=True, help="The jsonline file for test")
    parser.add_argument(
        "--target_dir",
        required=True,
        help="The target directory stores the csv files and meta file")
    parser.add_argument("--config",
                        default=None,
                        required=True,
                        type=str,
                        help="configuration file")
    args = parser.parse_args()

    # parse the yaml config file
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)

    prepare_data(args, config)