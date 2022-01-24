# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from multiprocessing import cpu_count
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tqdm
import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.data.get_feats import LogMelFBank
from paddlespeech.t2s.datasets import CSMSCMetaData
from paddlespeech.t2s.datasets import LJSpeechMetaData
from paddlespeech.t2s.datasets.vocoder_batch_fn import encode_mu_law
from paddlespeech.t2s.datasets.vocoder_batch_fn import float_2_label


class Transform(object):
    def __init__(self, output_dir: Path, config):
        self.fs = config.fs
        self.peak_norm = config.peak_norm
        self.bits = config.model.bits
        self.mode = config.model.mode
        self.mu_law = config.mu_law

        self.wav_dir = output_dir / "wav"
        self.mel_dir = output_dir / "mel"
        self.wav_dir.mkdir(exist_ok=True)
        self.mel_dir.mkdir(exist_ok=True)

        self.mel_extractor = LogMelFBank(
            sr=config.fs,
            n_fft=config.n_fft,
            hop_length=config.n_shift,
            win_length=config.win_length,
            window=config.window,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax)

        if self.mode != 'RAW' and self.mode != 'MOL':
            raise RuntimeError('Unknown mode value - ', self.mode)

    def __call__(self, example):
        wav_path, _, _ = example

        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        # print("self.sample_rate:",self.sample_rate)
        wav, _ = librosa.load(wav_path, sr=self.fs)
        peak = np.abs(wav).max()
        if self.peak_norm or peak > 1.0:
            wav /= peak

        mel = self.mel_extractor.get_log_mel_fbank(wav).T
        if self.mode == 'RAW':
            if self.mu_law:
                quant = encode_mu_law(wav, mu=2**self.bits)
            else:
                quant = float_2_label(wav, bits=self.bits)
        elif self.mode == 'MOL':
            quant = float_2_label(wav, bits=16)

        mel = mel.astype(np.float32)
        audio = quant.astype(np.int64)

        np.save(str(self.wav_dir / base_name), audio)
        np.save(str(self.mel_dir / base_name), mel)

        return base_name, mel.shape[-1], audio.shape[-1]


def create_dataset(config,
                   input_dir,
                   output_dir,
                   nprocs: int=1,
                   dataset_type: str="ljspeech"):
    input_dir = Path(input_dir).expanduser()
    '''
    LJSpeechMetaData.records: [filename, normalized text, speaker name(ljspeech)]
    CSMSCMetaData.records: [filename, normalized text, pinyin]
    '''
    if dataset_type == 'ljspeech':
        dataset = LJSpeechMetaData(input_dir)
    else:
        dataset = CSMSCMetaData(input_dir)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(exist_ok=True)

    transform = Transform(output_dir, config)

    file_names = []

    pool = Pool(processes=nprocs)

    for info in tqdm.tqdm(pool.imap(transform, dataset), total=len(dataset)):
        base_name, mel_len, audio_len = info
        file_names.append((base_name, mel_len, audio_len))

    meta_data = pd.DataFrame.from_records(file_names)
    meta_data.to_csv(
        str(output_dir / "metadata.csv"), sep="\t", index=None, header=None)
    print("saved meta data in to {}".format(
        os.path.join(output_dir, "metadata.csv")))

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create dataset")
    parser.add_argument(
        "--config", type=str, help="config file to overwrite default config.")

    parser.add_argument(
        "--input", type=str, help="path of the ljspeech dataset")
    parser.add_argument(
        "--output", type=str, help="path to save output dataset")
    parser.add_argument(
        "--num-cpu",
        type=int,
        default=cpu_count() // 2,
        help="number of process.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ljspeech",
        help="The dataset to preprocess, ljspeech or csmsc")

    args = parser.parse_args()

    with open(args.config, 'rt') as f:
        config = CfgNode(yaml.safe_load(f))

    if args.dataset != "ljspeech" and args.dataset != "csmsc":
        raise RuntimeError('Unknown dataset - ', args.dataset)

    create_dataset(
        config,
        input_dir=args.input,
        output_dir=args.output,
        nprocs=args.num_cpu,
        dataset_type=args.dataset)
