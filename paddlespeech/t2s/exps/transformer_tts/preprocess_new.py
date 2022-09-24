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

import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.datasets.preprocess_utils import get_input_token
from paddlespeech.t2s.datasets.preprocess_utils import get_phn_dur
from paddlespeech.t2s.datasets.preprocess_utils import get_spk_id_map
from paddlespeech.t2s.datasets.preprocess_utils import merge_silence
from paddlespeech.t2s.utils import str2bool

#from concurrent.futures import ThreadPoolExecutor
#from operator import itemgetter
#from typing import Any
#from typing import Dict
#from typing import List
#import jsonlines
#import librosa
#import numpy as np
#import tqdm
#from paddlespeech.t2s.datasets.preprocess_utils import compare_duration_and_mel_length


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument(
        "--dataset",
        default="baker",
        type=str,
        help="name of dataset, should in {baker, aishell3, ljspeech, vctk} now")

    parser.add_argument(
        "--rootdir", default=None, type=str, help="directory to dataset.")

    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.")
    parser.add_argument(
        "--dur-file", default=None, type=str, help="path to durations.txt.")

    parser.add_argument("--config", type=str, help="transformer config file.")

    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")

    parser.add_argument(
        "--cut-sil",
        type=str2bool,
        default=True,
        help="whether cut sil in the edge of audio")

    parser.add_argument(
        "--spk_emb_dir",
        default=None,
        type=str,
        help="directory to speaker embedding files.")
    args = parser.parse_args()

    rootdir = Path(args.rootdir).expanduser()
    dumpdir = Path(args.dumpdir).expanduser()
    # use absolute path
    dumpdir = dumpdir.resolve()
    dumpdir.mkdir(parents=True, exist_ok=True)
    dur_file = Path(args.dur_file).expanduser()

    if args.spk_emb_dir:
        spk_emb_dir = Path(args.spk_emb_dir).expanduser().resolve()
    else:
        spk_emb_dir = None

    assert rootdir.is_dir()
    assert dur_file.is_file()

    with open(args.config, 'rt') as f:
        config = CfgNode(yaml.safe_load(f))

    sentences, speaker_set = get_phn_dur(dur_file)

    merge_silence(sentences)
    phone_id_map_path = dumpdir / "phone_id_map.txt"
    speaker_id_map_path = dumpdir / "speaker_id_map.txt"
    get_input_token(sentences, phone_id_map_path, args.dataset)
    get_spk_id_map(speaker_set, speaker_id_map_path)

    if args.dataset == "baker":
        wav_files = sorted(list((rootdir / "Wave").rglob("*.wav")))
        # split data into 3 sections
        num_train = 9800
        num_dev = 100
        train_wav_files = wav_files[:num_train]
        dev_wav_files = wav_files[num_train:num_train + num_dev]
        test_wav_files = wav_files[num_train + num_dev:]
    elif args.dataset == "aishell3":
        sub_num_dev = 5
        wav_dir = rootdir / "train" / "wav"
        train_wav_files = []
        dev_wav_files = []
        test_wav_files = []
        for speaker in os.listdir(wav_dir):
            wav_files = sorted(list((wav_dir / speaker).rglob("*.wav")))
            if len(wav_files) > 100:
                train_wav_files += wav_files[:-sub_num_dev * 2]
                dev_wav_files += wav_files[-sub_num_dev * 2:-sub_num_dev]
                test_wav_files += wav_files[-sub_num_dev:]
            else:
                train_wav_files += wav_files

    elif args.dataset == "ljspeech":
        wav_files = sorted(list((rootdir / "wavs").rglob("*.wav")))
        # split data into 3 sections
        num_train = 12900
        num_dev = 100
        train_wav_files = wav_files[:num_train]
        dev_wav_files = wav_files[num_train:num_train + num_dev]
        test_wav_files = wav_files[num_train + num_dev:]
    elif args.dataset == "vctk":
        sub_num_dev = 5
        wav_dir = rootdir / "wav48_silence_trimmed"
        train_wav_files = []
        dev_wav_files = []
        test_wav_files = []
        for speaker in os.listdir(wav_dir):
            wav_files = sorted(list((wav_dir / speaker).rglob("*_mic2.flac")))
            if len(wav_files) > 100:
                train_wav_files += wav_files[:-sub_num_dev * 2]
                dev_wav_files += wav_files[-sub_num_dev * 2:-sub_num_dev]
                test_wav_files += wav_files[-sub_num_dev:]
            else:
                train_wav_files += wav_files

    else:
        print("dataset should in {baker, aishell3, ljspeech, vctk} now!")

    train_dump_dir = dumpdir / "train" / "raw"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dumpdir / "dev" / "raw"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dumpdir / "test" / "raw"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    # Extractor
    mel_extractor = LogMelFBank(
        sr=config.fs,
        n_fft=config.n_fft,
        hop_length=config.n_shift,
        win_length=config.win_length,
        window=config.window,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax)

    # process for the 3 sections
    if train_wav_files:
        process_sentences(
            config=config,
            fps=train_wav_files,
            sentences=sentences,
            output_dir=train_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu,
            cut_sil=args.cut_sil,
            spk_emb_dir=spk_emb_dir)
    if dev_wav_files:
        process_sentences(
            config=config,
            fps=dev_wav_files,
            sentences=sentences,
            output_dir=dev_dump_dir,
            mel_extractor=mel_extractor,
            cut_sil=args.cut_sil,
            spk_emb_dir=spk_emb_dir)
    if test_wav_files:
        process_sentences(
            config=config,
            fps=test_wav_files,
            sentences=sentences,
            output_dir=test_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu,
            cut_sil=args.cut_sil,
            spk_emb_dir=spk_emb_dir)


if __name__ == "__main__":
    main()
