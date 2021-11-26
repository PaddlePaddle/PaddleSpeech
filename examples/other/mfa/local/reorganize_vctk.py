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
"""Script to reorganize VCTK dataset so as to use Montreal Force
Aligner to align transcription and audio.

Please refer to https://montreal-forced-aligner.readthedocs.io/en/latest/data_prep.html
for more details about Montreal Force Aligner's requirements on cotpus.

For scripts to reorganize other corpus, please refer to 
 https://github.com/MontrealCorpusTools/MFA-reorganization-scripts
for more details.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import librosa
import soundfile as sf
from tqdm import tqdm


def resample_and_save(source, target, sr=16000):
    wav, _ = librosa.load(str(source), sr=sr)
    sf.write(str(target), wav, samplerate=sr, subtype='PCM_16')
    return target


def write_wav(root_dir: Union[str, Path], output_dir: Union[str, Path]):
    wav_paths = []
    new_wav_paths = []

    wav_dir = root_dir / 'wav48_silence_trimmed'
    new_dir = output_dir
    new_dir.mkdir(parents=True, exist_ok=True)

    for spk_dir in os.listdir(wav_dir):
        # no txt for p315
        # no mic2 for p280 and p362
        if spk_dir not in {'log.txt', 'p315', 'p280', 'p362'}:
            sub_dir = wav_dir / spk_dir
            new_sub_dir = new_dir / spk_dir
            new_sub_dir.mkdir(parents=True, exist_ok=True)
            for wav_name in os.listdir(sub_dir):
                # mic1 have very low frequency noises
                pre_wav_name = "_".join(wav_name.split("_")[:2])
                if "mic2" in wav_name:
                    wav_paths.append(str(sub_dir / wav_name))
                    # remove "_mic2" in wav_name and replace ".flac" with ".wav"
                    new_wav_name = pre_wav_name + ".wav"
                    new_wav_paths.append(str(new_sub_dir / new_wav_name))

    assert len(new_wav_paths) == len(wav_paths)

    with ThreadPoolExecutor(os.cpu_count()) as pool:
        with tqdm(total=len(wav_paths), desc="resampling") as pbar:
            futures = []
            for i, wav_path in enumerate(wav_paths):
                future = pool.submit(resample_and_save, wav_path,
                                     new_wav_paths[i])
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)

            results = []
            for ft in futures:
                results.append(ft.result())


def write_txt(root_dir: Union[str, Path], output_dir: Union[str, Path]):
    txt_dir = root_dir / 'txt'

    new_dir = output_dir
    new_dir.mkdir(parents=True, exist_ok=True)
    for spk_dir in os.listdir(txt_dir):
        # no txt for p315
        if spk_dir not in {'log.txt', 'p315', 'p280', 'p362'}:
            sub_dir = txt_dir / spk_dir
            new_sub_dir = new_dir / spk_dir
            for txt_name in os.listdir(sub_dir):
                rf = open(sub_dir / txt_name, "r")
                wf = open(new_sub_dir / txt_name, "w")
                for line in rf:
                    wf.write(line)


def reorganize_vctk(root_dir: Union[str, Path], output_dir: Union[str, Path]):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_wav(root_dir, output_dir)
    write_txt(root_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize VCTK-Corpus-0.92 dataset for MFA")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="",
        help="path to VCTK-Corpus-0.92 dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to save outputs (audio and transcriptions)")

    args = parser.parse_args()
    root_dir = Path(args.root_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    reorganize_vctk(root_dir, output_dir)
