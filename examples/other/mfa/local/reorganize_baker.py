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
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import librosa
import soundfile as sf
from tqdm import tqdm

repalce_dict = {
    "；": "",
    "。": "",
    "：": "",
    "—": "",
    "）": "",
    "，": "",
    "“": "",
    "（": "",
    "、": "",
    "…": "",
    "！": "",
    "？": "",
    "”": ""
}


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
                     resample_audio=False,
                     rhy_dur=False):
    root_dir = Path(root_dir).expanduser()
    if rhy_dur:
        transcript_path = root_dir / "ProsodyLabeling" / "000001-010000_rhy.txt"
    else:
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


def insert_rhy(sentence_first, sentence_second):
    sub = '#'
    return_words = []
    sentence_first = sentence_first.translate(str.maketrans(repalce_dict))
    rhy_idx = [substr.start() for substr in re.finditer(sub, sentence_first)]
    re_rhy_idx = []
    sentence_first_ = sentence_first.replace("#1", "").replace(
        "#2", "").replace("#3", "").replace("#4", "")
    sentence_seconds = sentence_second.split(" ")
    for i, w in enumerate(rhy_idx):
        re_rhy_idx.append(w - i * 2)
    i = 0
    # print("re_rhy_idx: ", re_rhy_idx)
    for sentence_s in (sentence_seconds):
        return_words.append(sentence_s)
        if i < len(re_rhy_idx) and len(return_words) - i == re_rhy_idx[i]:
            return_words.append("sp" + sentence_first[rhy_idx[i] + 1:rhy_idx[i]
                                                      + 2])
            i = i + 1
    return return_words


def normalize_rhy(root_dir: Union[str, Path]):
    root_dir = Path(root_dir).expanduser()
    transcript_path = root_dir / "ProsodyLabeling" / "000001-010000.txt"
    target_transcript_path = root_dir / "ProsodyLabeling" / "000001-010000_rhy.txt"

    with open(transcript_path) as f:
        lines = f.readlines()

    with open(target_transcript_path, 'wt') as f:
        for i in range(0, len(lines), 2):
            sentence_first = lines[i]  #第一行直接保存
            f.write(sentence_first)
            transcription = lines[i + 1].strip()
            f.write("\t" + " ".join(
                insert_rhy(sentence_first.split('\t')[1], transcription)) +
                    "\n")


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
    parser.add_argument(
        "--rhy-with-duration",
        action="store_true", )
    args = parser.parse_args()

    if args.rhy_with_duration:
        normalize_rhy(args.root_dir)
    reorganize_baker(args.root_dir, args.output_dir, args.resample_audio,
                     args.rhy_with_duration)
