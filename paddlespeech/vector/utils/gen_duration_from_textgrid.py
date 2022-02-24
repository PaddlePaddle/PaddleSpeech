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
import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import yaml
from praatio import textgrid
from yacs.config import CfgNode


def readtg(tg_path, sample_rate=24000, n_shift=300):
    alignment = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
    phones = []
    ends = []
    for interval in alignment.tierDict["phones"].entryList:
        phone = interval.label
        phones.append(phone)
        ends.append(interval.end)
    frame_pos = librosa.time_to_frames(ends, sr=sample_rate, hop_length=n_shift)
    durations = np.diff(frame_pos, prepend=0)
    assert len(durations) == len(phones)
    # merge  "" and sp in the end
    if phones[-1] == "" and len(phones) > 1 and phones[-2] == "sp":
        phones = phones[:-1]
        durations[-2] += durations[-1]
        durations = durations[:-1]
    # replace the last "sp" with "sil" in MFA1.x
    phones[-1] = "sil" if phones[-1] == "sp" else phones[-1]
    # replace the edge "" with "sil", replace the inner "" with "sp"
    new_phones = []
    for i, phn in enumerate(phones):
        if phn == "":
            if i in {0, len(phones) - 1}:
                new_phones.append("sil")
            else:
                new_phones.append("sp")
        else:
            new_phones.append(phn)
    phones = new_phones
    results = ""
    for (p, d) in zip(phones, durations):
        results += p + " " + str(d) + " "
    return results.strip()


# assume that the directory structure of inputdir is inputdir/speaker/*.TextGrid
# in MFA1.x, there are blank labels("") in the end, and maybe "sp" before it
# in MFA2.x, there are  blank labels("") in the begin and the end, while no "sp" and "sil" anymore
# we replace it with "sil"
def gen_duration_from_textgrid(inputdir, output, sample_rate=24000,
                               n_shift=300):
    # key: utt_id, value: (speaker, phn_durs)
    durations_dict = {}
    list_dir = os.listdir(inputdir)
    speakers = [dir for dir in list_dir if os.path.isdir(inputdir / dir)]
    for speaker in speakers:
        subdir = inputdir / speaker
        for file in os.listdir(subdir):
            if file.endswith(".TextGrid"):
                tg_path = subdir / file
                name = file.split(".")[0]
                durations_dict[name] = (speaker, readtg(
                    tg_path, sample_rate=sample_rate, n_shift=n_shift))
    with open(output, "w") as wf:
        for name in sorted(durations_dict.keys()):
            wf.write(name + "|" + durations_dict[name][0] + "|" +
                     durations_dict[name][1] + "\n")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")
    parser.add_argument(
        "--inputdir",
        default=None,
        type=str,
        help="directory to alignment files.")
    parser.add_argument(
        "--output", type=str, required=True, help="output duration file.")
    parser.add_argument("--sample-rate", type=int, help="the sample of wavs.")
    parser.add_argument(
        "--n-shift",
        type=int,
        help="the n_shift of time_to_freames, also called hop_length.")
    parser.add_argument(
        "--config", type=str, help="config file with fs and n_shift.")

    args = parser.parse_args()
    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))

    inputdir = Path(args.inputdir).expanduser()
    output = Path(args.output).expanduser()
    gen_duration_from_textgrid(inputdir, output, config.fs, config.n_shift)


if __name__ == "__main__":
    main()
