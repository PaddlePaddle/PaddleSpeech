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
import os
from typing import List

import tqdm
from yacs.config import CfgNode

from paddleaudio.backends import soundfile_load as load_audio
from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.utils.vector_utils import get_chunks

logger = Log(__name__).getlog()


def get_chunks_list(wav_file: str,
                    split_chunks: bool,
                    base_path: str,
                    chunk_duration: float=3.0) -> List[List[str]]:
    """Get the single audio file info 

    Args:
        wav_file (list): the wav audio file and get this audio segment info list
        split_chunks (bool): audio split flag
        base_path (str): the audio base path 
        chunk_duration (float): the chunk duration. 
                                if set the split_chunks, we split the audio into multi-chunks segment.
    """
    waveform, sr = load_audio(wav_file)
    audio_id = wav_file.split("/rir_noise/")[-1].split(".")[0]
    audio_duration = waveform.shape[0] / sr

    ret = []
    if split_chunks and audio_duration > chunk_duration:  # Split into pieces of self.chunk_duration seconds.
        uniq_chunks_list = get_chunks(chunk_duration, audio_id, audio_duration)

        for idx, chunk in enumerate(uniq_chunks_list):
            s, e = chunk.split("_")[-2:]  # Timestamps of start and end
            start_sample = int(float(s) * sr)
            end_sample = int(float(e) * sr)

            # currently, all vector csv data format use one representation
            # id, duration, wav, start, stop, label
            # in rirs noise, all the label name is 'noise'
            # the label is string type and we will convert it to integer type in training
            ret.append([
                chunk, audio_duration, wav_file, start_sample, end_sample,
                "noise"
            ])
    else:  # Keep whole audio.
        ret.append(
            [audio_id, audio_duration, wav_file, 0, waveform.shape[0], "noise"])
    return ret


def generate_csv(wav_files,
                 output_file: str,
                 base_path: str,
                 split_chunks: bool=True):
    """Prepare the csv file according the wav files

    Args:
        wav_files (list): all the audio list to prepare the csv file
        output_file (str): the output csv file
        config (CfgNode): yaml configuration content
        split_chunks (bool): audio split flag
    """
    logger.info(f'Generating csv: {output_file}')
    header = ["utt_id", "duration", "wav", "start", "stop", "label"]
    csv_lines = []
    for item in tqdm.tqdm(wav_files):
        csv_lines.extend(
            get_chunks_list(
                item, base_path=base_path, split_chunks=split_chunks))

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
        for line in csv_lines:
            csv_writer.writerow(line)


def prepare_data(args, config):
    """Convert the jsonline format to csv format

    Args:
        args (argparse.Namespace): scripts args
        config (CfgNode): yaml configuration content
    """
    # if external config set the skip_prep flat, we will do nothing
    if config.skip_prep:
        return

    base_path = args.noise_dir
    wav_path = os.path.join(base_path, "RIRS_NOISES")
    logger.info(f"base path: {base_path}")
    logger.info(f"wav path: {wav_path}")
    rir_list = os.path.join(wav_path, "real_rirs_isotropic_noises", "rir_list")
    rir_files = []
    with open(rir_list, 'r') as f:
        for line in f.readlines():
            rir_file = line.strip().split(' ')[-1]
            rir_files.append(os.path.join(base_path, rir_file))

    noise_list = os.path.join(wav_path, "pointsource_noises", "noise_list")
    noise_files = []
    with open(noise_list, 'r') as f:
        for line in f.readlines():
            noise_file = line.strip().split(' ')[-1]
            noise_files.append(os.path.join(base_path, noise_file))

    csv_path = os.path.join(args.data_dir, 'csv')
    logger.info(f"csv path: {csv_path}")
    generate_csv(
        rir_files, os.path.join(csv_path, 'rir.csv'), base_path=base_path)
    generate_csv(
        noise_files, os.path.join(csv_path, 'noise.csv'), base_path=base_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noise_dir",
        default=None,
        required=True,
        help="The noise dataset dataset directory.")
    parser.add_argument(
        "--data_dir",
        default=None,
        required=True,
        help="The target directory stores the csv files")
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
