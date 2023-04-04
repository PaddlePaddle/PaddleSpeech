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
"""Normalize feature files and dump them."""
import argparse
import logging
from operator import itemgetter
from pathlib import Path

import jsonlines
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from paddlespeech.t2s.datasets.data_table import DataTable


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Normalize dumped raw features.")
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="directory including feature files to be normalized. "
        "you need to specify either *-scp or rootdir.")
    parser.add_argument("--dumpdir",
                        type=str,
                        required=True,
                        help="directory to dump normalized feature files.")
    parser.add_argument("--stats",
                        type=str,
                        required=True,
                        help="statistics file.")
    parser.add_argument("--skip-wav-copy",
                        default=False,
                        action="store_true",
                        help="whether to skip the copy of wav files.")

    args = parser.parse_args()

    dumpdir = Path(args.dumpdir).expanduser()
    # use absolute path
    dumpdir = dumpdir.resolve()
    dumpdir.mkdir(parents=True, exist_ok=True)

    # get dataset
    with jsonlines.open(args.metadata, 'r') as reader:
        metadata = list(reader)
    dataset = DataTable(metadata,
                        fields=["utt_id", "wave", "feats"],
                        converters={
                            'utt_id': None,
                            'wave': None if args.skip_wav_copy else np.load,
                            'feats': np.load,
                        })
    logging.info(f"The number of files = {len(dataset)}.")

    # restore scaler
    scaler = StandardScaler()
    scaler.mean_ = np.load(args.stats)[0]
    scaler.scale_ = np.load(args.stats)[1]

    # from version 0.23.0, this information is needed
    scaler.n_features_in_ = scaler.mean_.shape[0]

    # process each file
    output_metadata = []

    for item in tqdm(dataset):
        utt_id = item['utt_id']
        wave = item['wave']
        mel = item['feats']
        # normalize
        mel = scaler.transform(mel)

        # save
        mel_path = dumpdir / f"{utt_id}_feats.npy"
        np.save(mel_path, mel.astype(np.float32), allow_pickle=False)
        if not args.skip_wav_copy:
            wav_path = dumpdir / f"{utt_id}_wave.npy"
            np.save(wav_path, wave.astype(np.float32), allow_pickle=False)
        else:
            wav_path = wave
        output_metadata.append({
            'utt_id': utt_id,
            'wave': str(wav_path),
            'feats': str(mel_path),
        })
    output_metadata.sort(key=itemgetter('utt_id'))
    output_metadata_path = Path(args.dumpdir) / "metadata.jsonl"
    with jsonlines.open(output_metadata_path, 'w') as writer:
        for item in output_metadata:
            writer.write(item)
    logging.info(f"metadata dumped into {output_metadata_path}")


if __name__ == "__main__":
    main()
