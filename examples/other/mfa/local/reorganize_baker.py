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
"""Script to reorganize Baker dataset so as to use Montreal Force
Aligner to align transcription and audio.

Please refer to https://montreal-forced-aligner.readthedocs.io/en/latest/data_prep.html
for more details about Montreal Force Aligner's requirements on cotpus.

For scripts to reorganize other corpus, please refer to 
 https://github.com/MontrealCorpusTools/MFA-reorganization-scripts
for more details.
"""
import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import librosa
import soundfile as sf
from tqdm import tqdm


def get_transcripts(path: Union[str, Path]):
    transcripts = {}

    with open(path) as f:
        lines = f.readlines()

    for i in range(0, len(lines), 2):
        sentence_id = lines[i].split()[0]
        transcription = lines[i + 1].strip()
        transcripts[sentence_id] = transcription

    return transcripts


def resample_and_save(source, target, sr=16000):
    wav, _ = librosa.load(str(source), sr=sr)
    sf.write(str(target), wav, samplerate=sr, subtype='PCM_16')
    return target


def reorganize_baker(root_dir: Union[str, Path],
                     output_dir: Union[str, Path]=None,
                     resample_audio=False):
    root_dir = Path(root_dir).expanduser()
    transcript_path = root_dir / "ProsodyLabeling" / "000001-010000.txt"
    transcriptions = get_transcripts(transcript_path)

    wave_dir = root_dir / "Wave"
    wav_paths = sorted(list(wave_dir.glob("*.wav")))
    output_dir = Path(output_dir).expanduser()
    assert wave_dir != output_dir, "Don't use an the original wav's directory as output_dir"

    output_dir.mkdir(parents=True, exist_ok=True)

    if resample_audio:
        with ThreadPoolExecutor(os.cpu_count()) as pool:
            with tqdm(total=len(wav_paths), desc="resampling") as pbar:
                futures = []
                for wav_path in wav_paths:
                    future = pool.submit(resample_and_save, wav_path,
                                         output_dir / wav_path.name)
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    results.append(ft.result())
    else:
        for wav_path in tqdm(wav_paths, desc="copying"):
            shutil.copyfile(wav_path, output_dir / wav_path.name)

    for sentence_id, transcript in tqdm(
            transcriptions.items(), desc="transcription process"):
        with open(output_dir / (sentence_id + ".lab"), 'wt') as f:
            f.write(transcript)
            f.write('\n')
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize Baker dataset for MFA")
    parser.add_argument("--root-dir", type=str, help="path to baker dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to save outputs (audio and transcriptions)")
    parser.add_argument(
        "--resample-audio",
        action="store_true",
        help="To resample audio files or just copy them")
    args = parser.parse_args()

    reorganize_baker(args.root_dir, args.output_dir, args.resample_audio)
