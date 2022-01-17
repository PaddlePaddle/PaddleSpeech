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
from operator import itemgetter
from pathlib import Path

import jsonlines
import numpy as np
from tqdm import tqdm


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features .")

    parser.add_argument(
        "--old-dump-dir",
        default=None,
        type=str,
        help="directory to dump feature files.")
    parser.add_argument(
        "--dump-dir",
        type=str,
        required=True,
        help="directory to finetune dump feature files.")
    args = parser.parse_args()

    old_dump_dir = Path(args.old_dump_dir).expanduser()
    old_dump_dir = old_dump_dir.resolve()
    dump_dir = Path(args.dump_dir).expanduser()
    # use absolute path
    dump_dir = dump_dir.resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)

    assert old_dump_dir.is_dir()
    assert dump_dir.is_dir()

    for sub in ["train", "dev", "test"]:
        # 把 old_dump_dir 里面的 *-wave.npy 软连接到 dump_dir 的对应位置
        output_dir = dump_dir / sub
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        files = os.listdir(output_dir / "raw")
        for name in tqdm(files):
            utt_id = name.split("_feats.npy")[0]
            mel_path = output_dir / ("raw/" + name)
            gen_mel = np.load(mel_path)
            wave_name = utt_id + "_wave.npy"
            try:
                wav = np.load(old_dump_dir / sub / ("raw/" + wave_name))
                os.symlink(old_dump_dir / sub / ("raw/" + wave_name),
                           output_dir / ("raw/" + wave_name))
            except FileNotFoundError:
                print("delete " + name +
                      " because it cannot be found in the dump folder")
                os.remove(output_dir / "raw" / name)
                continue
            except FileExistsError:
                print("file " + name + " exists, skip.")
                continue
            num_sample = wav.shape[0]
            num_frames = gen_mel.shape[0]
            wav_path = output_dir / ("raw/" + wave_name)

            record = {
                "utt_id": utt_id,
                "num_samples": num_sample,
                "num_frames": num_frames,
                "feats": str(mel_path),
                "wave": str(wav_path),
            }
            results.append(record)

        results.sort(key=itemgetter("utt_id"))

        with jsonlines.open(output_dir / "raw/metadata.jsonl", 'w') as writer:
            for item in results:
                writer.write(item)


if __name__ == "__main__":
    main()
