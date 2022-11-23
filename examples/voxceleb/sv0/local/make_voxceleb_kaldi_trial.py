#!/usr/bin/python3
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
Make VoxCeleb1 trial of kaldi format
this script creat the test trial from kaldi trial voxceleb1_test_v2.txt or official trial veri_test2.txt 
to kaldi trial format
"""
import argparse
import codecs
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--voxceleb_trial",
    default="voxceleb1_test_v2",
    type=str,
    help="VoxCeleb trial file. Default we use the kaldi trial voxceleb1_test_v2.txt"
)
parser.add_argument(
    "--trial",
    default="data/test/trial",
    type=str,
    help="Kaldi format trial file")
args = parser.parse_args()


def main(voxceleb_trial, trial):
    """
        VoxCeleb provide several trial file, which format is different with kaldi format.

        VoxCeleb format's meaning is as following:
        --------------------------------
        target_or_nontarget path1 path2
        --------------------------------
        target_or_nontarget is an integer: 1 target                 path1 is equal to path2
                                           0 nontarget              path1 is unequal to path2    
        path1: spkr_id/rec_id/name
        path2: spkr_id/rec_id/name

        Kaldi format's meaning is as following:
        ---------------------------------------
        utt_id1 utt_id2 target_or_nontarget
        ---------------------------------------
        utt_id1: utterance identification or speaker identification
        utt_id2: utterance identification or speaker identification
        target_or_nontarget is an string: 'target' utt_id1 is equal to  utt_id2
                                        'nontarget' utt_id2 is unequal to utt_id2
    """
    print("Start convert the voxceleb trial to kaldi format")
    if not os.path.exists(voxceleb_trial):
        raise RuntimeError(
            "{} does not exist. Pleas input the correct file path".format(
                voxceleb_trial))

    trial_dirname = os.path.dirname(trial)
    if not os.path.exists(trial_dirname):
        os.mkdir(trial_dirname)

    with codecs.open(voxceleb_trial, 'r', encoding='utf-8') as f, \
         codecs.open(trial, 'w', encoding='utf-8') as w:
        for line in f:
            target_or_nontarget, path1, path2 = line.strip().split()

            utt_id1 = "-".join(path1.split("/"))
            utt_id2 = "-".join(path2.split("/"))
            target = "nontarget"
            if int(target_or_nontarget):
                target = "target"
            w.write("{} {} {}\n".format(utt_id1, utt_id2, target))
    print("Convert the voxceleb trial to kaldi format successfully")


if __name__ == "__main__":
    main(args.voxceleb_trial, args.trial)
