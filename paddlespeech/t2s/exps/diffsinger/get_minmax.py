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
import argparse
import logging

import jsonlines
import numpy as np
from tqdm import tqdm

from paddlespeech.t2s.datasets.data_table import DataTable


def get_minmax(spec, min_spec, max_spec):
    # spec: [T, 80]
    for i in range(spec.shape[1]):
        min_value = np.min(spec[:, i])
        max_value = np.max(spec[:, i])
        min_spec[i] = min(min_value, min_spec[i])
        max_spec[i] = max(max_value, max_spec[i])

    return min_spec, max_spec


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=
        "Normalize dumped raw features (See detail in parallel_wavegan/bin/normalize.py)."
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="directory including feature files to be normalized. "
        "you need to specify either *-scp or rootdir.")

    parser.add_argument("--speech-stretchs",
                        type=str,
                        required=True,
                        help="min max spec file. only computer on train data")

    args = parser.parse_args()

    # get dataset
    with jsonlines.open(args.metadata, 'r') as reader:
        metadata = list(reader)
    dataset = DataTable(metadata, converters={
        "speech": np.load,
    })
    logging.info(f"The number of files = {len(dataset)}.")

    n_mel = 80
    min_spec = 100.0 * np.ones(shape=(n_mel), dtype=np.float32)
    max_spec = -100.0 * np.ones(shape=(n_mel), dtype=np.float32)

    for item in tqdm(dataset):
        spec = item['speech']
        min_spec, max_spec = get_minmax(spec, min_spec, max_spec)

    # Using min_spec=-6.0 training effect is better so far
    min_spec = -6.0 * np.ones(shape=(n_mel), dtype=np.float32)
    min_max_spec = np.stack([min_spec, max_spec], axis=0)
    np.save(str(args.speech_stretchs),
            min_max_spec.astype(np.float32),
            allow_pickle=False)


if __name__ == "__main__":
    main()
