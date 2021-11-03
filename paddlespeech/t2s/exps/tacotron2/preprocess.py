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
import pickle
from pathlib import Path

import numpy as np
import tqdm

from paddlespeech.t2s.audio import AudioProcessor
from paddlespeech.t2s.audio import LogMagnitude
from paddlespeech.t2s.datasets import LJSpeechMetaData
from paddlespeech.t2s.exps.tacotron2.config import get_cfg_defaults
from paddlespeech.t2s.frontend import EnglishCharacter


def create_dataset(config, source_path, target_path, verbose=False):
    # create output dir
    target_path = Path(target_path).expanduser()
    mel_path = target_path / "mel"
    os.makedirs(mel_path, exist_ok=True)

    meta_data = LJSpeechMetaData(source_path)
    frontend = EnglishCharacter()
    processor = AudioProcessor(
        sample_rate=config.data.sample_rate,
        n_fft=config.data.n_fft,
        n_mels=config.data.n_mels,
        win_length=config.data.win_length,
        hop_length=config.data.hop_length,
        fmax=config.data.fmax,
        fmin=config.data.fmin)
    normalizer = LogMagnitude()

    records = []
    for (fname, text, _) in tqdm.tqdm(meta_data):
        wav = processor.read_wav(fname)
        mel = processor.mel_spectrogram(wav)
        mel = normalizer.transform(mel)
        ids = frontend(text)
        mel_name = os.path.splitext(os.path.basename(fname))[0]

        # save mel spectrogram
        records.append((mel_name, text, ids))
        np.save(mel_path / mel_name, mel)
    if verbose:
        print("save mel spectrograms into {}".format(mel_path))

    # save meta data as pickle archive
    with open(target_path / "metadata.pkl", 'wb') as f:
        pickle.dump(records, f)
        if verbose:
            print("saved metadata into {}".format(target_path / "metadata.pkl"))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create dataset")
    parser.add_argument(
        "--config",
        type=str,
        metavar="FILE",
        help="extra config to overwrite the default config")
    parser.add_argument(
        "--input", type=str, help="path of the ljspeech dataset")
    parser.add_argument(
        "--output", type=str, help="path to save output dataset")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="options to overwrite --config file and the default config, passing in KEY VALUE pairs"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print msg")

    config = get_cfg_defaults()
    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config.data)

    create_dataset(config, args.input, args.output, args.verbose)
