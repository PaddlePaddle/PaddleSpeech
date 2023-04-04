# Copyright (c) 2023 speechbrain Authors. All Rights Reserved.
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#
# Modified from speechbrain 2023
# (https://github.com/speechbrain/speechbrain/blob/develop/recipes/AISHELL-1/aishell_prepare.py)
import argparse
import csv
import glob
import logging
import os

from paddlespeech.s2t.io.speechbrain.dataio import read_audio

logger = logging.getLogger(__name__)

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--data_folder",
    default=DATA_HOME + "/Aishell",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--save_folder",
    default="data/",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
parser.add_argument(
    "--skip_prep",
    default=False,
    type=bool,
    help="If True, skip data preparation. (default: %(default)s)")
args = parser.parse_args()


def prepare_aishell(data_folder, save_folder, skip_prep=False):
    """
    This function prepares the AISHELL-1 dataset.
    If the folder does not exist, the zip file will be extracted. If the zip file does not exist, it will be downloaded.
    data_folder : path to AISHELL-1 dataset.
    save_folder: path where to store the manifest csv files.
    skip_prep: If True, skip data preparation.
    """
    if skip_prep:
        return

    # Create filename-to-transcript dictionary
    filename2transcript = {}
    with open(
            os.path.join(data_folder,
                         "data_aishell/transcript/aishell_transcript_v0.8.txt"),
            "r",
    ) as f:
        lines = f.readlines()
        for line in lines:
            key = line.split()[0]
            value = " ".join(line.split()[1:])
            filename2transcript[key] = value

    splits = [
        "train",
        "dev",
        "test",
    ]
    ID_start = 0  # needed to have a unique ID for each audio
    for split in splits:
        new_filename = os.path.join(save_folder, split) + ".csv"
        if os.path.exists(new_filename):
            continue
        logger.info("Preparing %s..." % new_filename)

        csv_output = [["ID", "duration", "wav", "transcript"]]
        entry = []

        all_wavs = glob.glob(
            os.path.join(data_folder, "data_aishell/wav") + "/" + split +
            "/*/*.wav")
        for i in range(len(all_wavs)):
            filename = all_wavs[i].split("/")[-1].split(".wav")[0]
            if filename not in filename2transcript:
                continue
            signal = read_audio(all_wavs[i])
            duration = signal.shape[0] / 16000
            transcript_ = filename2transcript[filename]
            csv_line = [
                ID_start + i,
                str(duration),
                all_wavs[i],
                transcript_,
            ]
            entry.append(csv_line)

        csv_output = csv_output + entry

        with open(new_filename, mode="w") as csv_f:
            csv_writer = csv.writer(csv_f,
                                    delimiter=",",
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            for line in csv_output:
                csv_writer.writerow(line)

        msg = "\t%s successfully created!" % (new_filename)
        logger.info(msg)

        ID_start += len(all_wavs)


def main():
    if args.data_folder.startswith('~'):
        args.data_folder = os.path.expanduser(args.data_folder)

    prepare_aishell(args.data_folder, args.save_folder, skip_prep=False)

    print("Data csv prepare done!")


if __name__ == '__main__':
    main()
