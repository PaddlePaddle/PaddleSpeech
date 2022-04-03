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
Convert the PaddleSpeech jsonline format data to csv format data in voxceleb experiment.
Currently, Speaker Identificaton Training process use csv format.
"""
import argparse
import csv
import json
import os
import random

import tqdm
from yacs.config import CfgNode

from paddleaudio import load as load_audio
from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.utils.vector_utils import get_chunks

logger = Log(__name__).getlog()


def prepare_csv(wav_files, output_file, config, split_chunks=True):
    """Prepare the csv file according the wav files

    Args:
        wav_files (list): all the audio list to prepare the csv file
        output_file (str): the output csv file
        config (CfgNode): yaml configuration content
        split_chunks (bool, optional): audio split flag. Defaults to True.
    """
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    csv_lines = []
    header = ["utt_id", "duration", "wav", "start", "stop", "label"]
    # voxceleb meta info for each training utterance segment
    # we extract a segment from a utterance to train 
    # and the segment' period is between start and stop time point in the original wav file
    # each field in the meta info means as follows:
    # utt_id: the utterance segment name, which is uniq in training dataset
    # duration: the total utterance time
    # wav: utterance file path, which should be absoulute path
    # start: start point in the original wav file sample point range
    # stop: stop point in the original wav file sample point range
    # label: the utterance segment's label name, 
    #        which is speaker name in speaker verification domain
    for item in tqdm.tqdm(wav_files, total=len(wav_files)):
        item = json.loads(item.strip())
        audio_id = item['utt'].replace(".wav",
                                       "")  # we remove the wav suffix name
        audio_duration = item['feat_shape'][0]
        wav_file = item['feat']
        label = audio_id.split('-')[
            0]  # speaker name in speaker verification domain
        waveform, sr = load_audio(wav_file)
        if split_chunks:
            uniq_chunks_list = get_chunks(config.chunk_duration, audio_id,
                                          audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]  # Timestamps of start and end
                start_sample = int(float(s) * sr)
                end_sample = int(float(e) * sr)
                # id, duration, wav, start, stop, label
                # in vector, the label in speaker id
                csv_lines.append([
                    chunk, audio_duration, wav_file, start_sample, end_sample,
                    label
                ])
        else:
            csv_lines.append([
                audio_id, audio_duration, wav_file, 0, waveform.shape[0], label
            ])

    with open(output_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
        for line in csv_lines:
            csv_writer.writerow(line)


def get_enroll_test_list(dataset_list, verification_file):
    """Get the enroll and test utterance list from all the voxceleb1 test utterance dataset.
       Generally, we get the enroll and test utterances from the verfification file.
       The verification file format as follows:
       target/nontarget enroll-utt test-utt,
       we set 0 as nontarget and 1 as target, eg:
       0 a.wav b.wav
       1 a.wav a.wav

    Args:
        dataset_list (list): all the dataset to get the test utterances
        verification_file (str): voxceleb1 trial file
    """
    logger.info(f"verification file: {verification_file}")
    enroll_audios = set()
    test_audios = set()
    with open(verification_file, 'r') as f:
        for line in f:
            _, enroll_file, test_file = line.strip().split(' ')
            enroll_audios.add('-'.join(enroll_file.split('/')))
            test_audios.add('-'.join(test_file.split('/')))

    enroll_files = []
    test_files = []
    for dataset in dataset_list:
        with open(dataset, 'r') as f:
            for line in f:
                # audio_id may be in enroll and test at the same time
                # eg: 1 a.wav a.wav
                # the audio a.wav is enroll and test file at the same time
                audio_id = json.loads(line.strip())['utt']
                if audio_id in enroll_audios:
                    enroll_files.append(line)
                if audio_id in test_audios:
                    test_files.append(line)

    enroll_files = sorted(enroll_files)
    test_files = sorted(test_files)

    return enroll_files, test_files


def get_train_dev_list(dataset_list, target_dir, split_ratio):
    """Get the train and dev utterance list from all the training utterance dataset.
       Generally, we use the split_ratio as the train dataset ratio,
       and the remaining utterance (ratio is 1 - split_ratio) is the dev dataset

    Args:
        dataset_list (list): all the dataset to get the all utterances
        target_dir (str): the target train and dev directory, 
                          we will create the csv directory to store the {train,dev}.csv file
        split_ratio (float): train dataset ratio in all utterance list
    """
    logger.info("start to get train and dev utt list")
    if not os.path.exists(os.path.join(target_dir, "meta")):
        os.makedirs(os.path.join(target_dir, "meta"))

    audio_files = []
    speakers = set()
    for dataset in dataset_list:
        with open(dataset, 'r') as f:
            for line in f:
                # the label is speaker name
                label_name = json.loads(line.strip())['utt2spk']
                speakers.add(label_name)
                audio_files.append(line.strip())
    speakers = sorted(speakers)
    logger.info(f"we get {len(speakers)} speakers from all the train dataset")

    with open(os.path.join(target_dir, "meta", "label2id.txt"), 'w') as f:
        for label_id, label_name in enumerate(speakers):
            f.write(f'{label_name} {label_id}\n')
    logger.info(
        f'we store the speakers to {os.path.join(target_dir, "meta", "label2id.txt")}'
    )

    # the split_ratio is for train dataset 
    # the remaining is for dev dataset
    split_idx = int(split_ratio * len(audio_files))
    audio_files = sorted(audio_files)
    random.shuffle(audio_files)
    train_files, dev_files = audio_files[:split_idx], audio_files[split_idx:]
    logger.info(
        f"we get train utterances: {len(train_files)}, dev utterance: {len(dev_files)}"
    )
    return train_files, dev_files


def prepare_data(args, config):
    """Convert the jsonline format to csv format

    Args:
        args (argparse.Namespace): scripts args
        config (CfgNode): yaml configuration content
    """
    # stage0: set the random seed
    random.seed(config.seed)

    # if external config set the skip_prep flat, we will do nothing
    if config.skip_prep:
        return

    # stage 1: prepare the enroll and test csv file
    #          And we generate the speaker to label file label2id.txt
    logger.info("start to prepare the data csv file")
    enroll_files, test_files = get_enroll_test_list(
        [args.test], verification_file=config.verification_file)
    prepare_csv(
        enroll_files,
        os.path.join(args.target_dir, "csv", "enroll.csv"),
        config,
        split_chunks=False)
    prepare_csv(
        test_files,
        os.path.join(args.target_dir, "csv", "test.csv"),
        config,
        split_chunks=False)

    # stage 2: prepare the train and dev csv file
    #          we get the train dataset ratio as config.split_ratio
    #          and the remaining is dev dataset
    logger.info("start to prepare the data csv file")
    train_files, dev_files = get_train_dev_list(
        args.train, target_dir=args.target_dir, split_ratio=config.split_ratio)
    prepare_csv(train_files,
                os.path.join(args.target_dir, "csv", "train.csv"), config)
    prepare_csv(dev_files,
                os.path.join(args.target_dir, "csv", "dev.csv"), config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        required=True,
        nargs='+',
        help="The jsonline files list for train.")
    parser.add_argument(
        "--test", required=True, help="The jsonline file for test")
    parser.add_argument(
        "--target_dir",
        default=None,
        required=True,
        help="The target directory stores the csv files and meta file.")
    parser.add_argument(
        "--config",
        default=None,
        required=True,
        type=str,
        help="configuration file")
    args = parser.parse_args()

    # parse the yaml config file
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)

    # prepare the csv file from jsonlines files
    prepare_data(args, config)
