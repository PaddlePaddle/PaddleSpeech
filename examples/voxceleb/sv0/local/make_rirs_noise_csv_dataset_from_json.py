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

import paddle
import tqdm
from yacs.config import CfgNode

from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.training.seeding import seed_everything
logger = Log(__name__).getlog()
from paddleaudio import load as load_audio
from paddleaudio import save as save_wav


def get_chunks(seg_dur, audio_id, audio_duration):
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]
    return chunk_lst


def get_audio_info(wav_file: str,
                   split_chunks: bool,
                   base_path: str,
                   chunk_duration: float=3.0) -> List[List[str]]:
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
            new_wav_file = os.path.join(base_path,
                                        audio_id + f'_chunk_{idx+1:02}.wav')
            save_wav(waveform[start_sample:end_sample], sr, new_wav_file)
            # id, duration, new_wav
            ret.append([chunk, chunk_duration, new_wav_file])
    else:  # Keep whole audio.
        ret.append([audio_id, audio_duration, wav_file])
    return ret


def generate_csv(wav_files,
                 output_file: str,
                 base_path: str,
                 split_chunks: bool=True):
    print(f'Generating csv: {output_file}')
    header = ["id", "duration", "wav"]
    csv_lines = []
    for item in tqdm.tqdm(wav_files):
        csv_lines.extend(
            get_audio_info(
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
    # stage0: set the cpu device, 
    # all data prepare process will be done in cpu mode
    paddle.device.set_device("cpu")
    # set the random seed, it is a must for multiprocess training
    seed_everything(config.seed)
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
