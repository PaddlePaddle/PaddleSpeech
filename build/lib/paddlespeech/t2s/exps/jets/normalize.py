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
        description="Normalize dumped raw features (See detail in parallel_wavegan/bin/normalize.py)."
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="directory including feature files to be normalized. "
        "you need to specify either *-scp or rootdir.")

    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump normalized feature files.")
    parser.add_argument(
        "--feats-stats", type=str, required=True, help="feats statistics file.")
    parser.add_argument(
        "--pitch-stats", type=str, required=True, help="pitch statistics file.")
    parser.add_argument(
        "--energy-stats",
        type=str,
        required=True,
        help="energy statistics file.")
    parser.add_argument(
        "--phones-dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--speaker-dict", type=str, default=None, help="speaker id map file.")

    args = parser.parse_args()

    dumpdir = Path(args.dumpdir).expanduser()
    # use absolute path
    dumpdir = dumpdir.resolve()
    dumpdir.mkdir(parents=True, exist_ok=True)

    # get dataset
    with jsonlines.open(args.metadata, 'r') as reader:
        metadata = list(reader)
    dataset = DataTable(
        metadata,
        converters={
            "feats": np.load,
            "pitch": np.load,
            "energy": np.load,
            "wave": str,
        })
    logging.info(f"The number of files = {len(dataset)}.")

    # restore scaler
    feats_scaler = StandardScaler()
    feats_scaler.mean_ = np.load(args.feats_stats)[0]
    feats_scaler.scale_ = np.load(args.feats_stats)[1]
    feats_scaler.n_features_in_ = feats_scaler.mean_.shape[0]

    pitch_scaler = StandardScaler()
    pitch_scaler.mean_ = np.load(args.pitch_stats)[0]
    pitch_scaler.scale_ = np.load(args.pitch_stats)[1]
    pitch_scaler.n_features_in_ = pitch_scaler.mean_.shape[0]

    energy_scaler = StandardScaler()
    energy_scaler.mean_ = np.load(args.energy_stats)[0]
    energy_scaler.scale_ = np.load(args.energy_stats)[1]
    energy_scaler.n_features_in_ = energy_scaler.mean_.shape[0]

    vocab_phones = {}
    with open(args.phones_dict, 'rt') as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    for phn, id in phn_id:
        vocab_phones[phn] = int(id)

    vocab_speaker = {}
    with open(args.speaker_dict, 'rt') as f:
        spk_id = [line.strip().split() for line in f.readlines()]
    for spk, id in spk_id:
        vocab_speaker[spk] = int(id)

    # process each file
    output_metadata = []

    for item in tqdm(dataset):
        utt_id = item['utt_id']
        feats = item['feats']
        pitch = item['pitch']
        energy = item['energy']
        wave_path = item['wave']
        # normalize
        feats = feats_scaler.transform(feats)
        feats_dir = dumpdir / "data_feats"
        feats_dir.mkdir(parents=True, exist_ok=True)
        feats_path = feats_dir / f"{utt_id}_feats.npy"
        np.save(feats_path, feats.astype(np.float32), allow_pickle=False)

        pitch = pitch_scaler.transform(pitch)
        pitch_dir = dumpdir / "data_pitch"
        pitch_dir.mkdir(parents=True, exist_ok=True)
        pitch_path = pitch_dir / f"{utt_id}_pitch.npy"
        np.save(pitch_path, pitch.astype(np.float32), allow_pickle=False)

        energy = energy_scaler.transform(energy)
        energy_dir = dumpdir / "data_energy"
        energy_dir.mkdir(parents=True, exist_ok=True)
        energy_path = energy_dir / f"{utt_id}_energy.npy"
        np.save(energy_path, energy.astype(np.float32), allow_pickle=False)

        phone_ids = [vocab_phones[p] for p in item['phones']]
        spk_id = vocab_speaker[item["speaker"]]
        record = {
            "utt_id": item['utt_id'],
            "spk_id": spk_id,
            "text": phone_ids,
            "text_lengths": item['text_lengths'],
            "feats_lengths": item['feats_lengths'],
            "durations": item['durations'],
            "feats": str(feats_path),
            "pitch": str(pitch_path),
            "energy": str(energy_path),
            "wave": str(wave_path),
        }
        # add spk_emb for voice cloning
        if "spk_emb" in item:
            record["spk_emb"] = str(item["spk_emb"])

        output_metadata.append(record)
    output_metadata.sort(key=itemgetter('utt_id'))
    output_metadata_path = Path(args.dumpdir) / "metadata.jsonl"
    with jsonlines.open(output_metadata_path, 'w') as writer:
        for item in output_metadata:
            writer.write(item)
    logging.info(f"metadata dumped into {output_metadata_path}")


if __name__ == "__main__":
    main()
