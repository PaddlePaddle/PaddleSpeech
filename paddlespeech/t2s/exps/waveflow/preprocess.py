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
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tqdm

from paddlespeech.t2s.audio import LogMagnitude
from paddlespeech.t2s.datasets import LJSpeechMetaData
from paddlespeech.t2s.exps.waveflow.config import get_cfg_defaults


class Transform(object):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels, fmin,
                 fmax):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        self.spec_normalizer = LogMagnitude(min=1e-5)

    def __call__(self, example):
        wav_path, _, _ = example

        sr = self.sample_rate
        n_fft = self.n_fft
        win_length = self.win_length
        hop_length = self.hop_length
        n_mels = self.n_mels
        fmin = self.fmin
        fmax = self.fmax

        wav, loaded_sr = librosa.load(wav_path, sr=None)
        assert loaded_sr == sr, "sample rate does not match, resampling applied"

        # Pad audio to the right size.
        frames = int(np.ceil(float(wav.size) / hop_length))
        fft_padding = (n_fft - hop_length) // 2  # sound
        desired_length = frames * hop_length + fft_padding * 2
        pad_amount = (desired_length - wav.size) // 2

        if wav.size % 2 == 0:
            wav = np.pad(wav, (pad_amount, pad_amount), mode='reflect')
        else:
            wav = np.pad(wav, (pad_amount, pad_amount + 1), mode='reflect')

        # Normalize audio.
        wav = wav / np.abs(wav).max() * 0.999

        # Compute mel-spectrogram.
        # Turn center to False to prevent internal padding.
        spectrogram = librosa.core.stft(wav,
                                        hop_length=hop_length,
                                        win_length=win_length,
                                        n_fft=n_fft,
                                        center=False)
        spectrogram_magnitude = np.abs(spectrogram)

        # Compute mel-spectrograms.
        mel_filter_bank = librosa.filters.mel(sr=sr,
                                              n_fft=n_fft,
                                              n_mels=n_mels,
                                              fmin=fmin,
                                              fmax=fmax)
        mel_spectrogram = np.dot(mel_filter_bank, spectrogram_magnitude)

        # log scale mel_spectrogram.
        mel_spectrogram = self.spec_normalizer.transform(mel_spectrogram)

        # Extract the center of audio that corresponds to mel spectrograms.
        audio = wav[fft_padding:-fft_padding]
        assert mel_spectrogram.shape[1] * hop_length == audio.size

        # there is no clipping here
        return audio, mel_spectrogram


def create_dataset(config, input_dir, output_dir):
    input_dir = Path(input_dir).expanduser()
    dataset = LJSpeechMetaData(input_dir)

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(exist_ok=True)

    transform = Transform(config.sample_rate, config.n_fft, config.win_length,
                          config.hop_length, config.n_mels, config.fmin,
                          config.fmax)
    file_names = []

    for example in tqdm.tqdm(dataset):
        fname, _, _ = example
        base_name = os.path.splitext(os.path.basename(fname))[0]
        wav_dir = output_dir / "wav"
        mel_dir = output_dir / "mel"
        wav_dir.mkdir(exist_ok=True)
        mel_dir.mkdir(exist_ok=True)

        audio, mel = transform(example)
        np.save(str(wav_dir / base_name), audio)
        np.save(str(mel_dir / base_name), mel)

        file_names.append((base_name, mel.shape[-1], audio.shape[-1]))

    meta_data = pd.DataFrame.from_records(file_names)
    meta_data.to_csv(str(output_dir / "metadata.csv"),
                     sep="\t",
                     index=None,
                     header=None)
    print("saved meta data in to {}".format(
        os.path.join(output_dir, "metadata.csv")))

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create dataset")
    parser.add_argument("--config",
                        type=str,
                        metavar="FILE",
                        help="extra config to overwrite the default config")
    parser.add_argument("--input",
                        type=str,
                        help="path of the ljspeech dataset")
    parser.add_argument("--output",
                        type=str,
                        help="path to save output dataset")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help=
        "options to overwrite --config file and the default config, passing in KEY VALUE pairs"
    )

    config = get_cfg_defaults()
    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()

    create_dataset(config.data, args.input, args.output)
