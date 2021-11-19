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
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from paddlespeech.vector.exps.ge2e.audio_processor import SpeakerVerificationPreprocessor


def _process_utterance(path_pair, processor: SpeakerVerificationPreprocessor):
    # Load and preprocess the waveform
    input_path, output_path = path_pair
    wav = processor.preprocess_wav(input_path)
    if len(wav) == 0:
        return

    # Create the mel spectrogram, discard those that are too short
    frames = processor.melspectrogram(wav)
    if len(frames) < processor.partial_n_frames:
        return

    np.save(output_path, frames)


def _process_speaker(speaker_dir: Path,
                     processor: SpeakerVerificationPreprocessor,
                     datasets_root: Path,
                     output_dir: Path,
                     pattern: str,
                     skip_existing: bool=False):
    # datastes root: a reference path to compute speaker_name
    # we prepand dataset name to speaker_id becase we are mixing serveal
    # multispeaker datasets together
    speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)
    speaker_output_dir = output_dir / speaker_name
    speaker_output_dir.mkdir(parents=True, exist_ok=True)

    # load exsiting file set
    sources_fpath = speaker_output_dir / "_sources.txt"
    if sources_fpath.exists():
        try:
            with sources_fpath.open("rt") as sources_file:
                existing_names = {line.split(",")[0] for line in sources_file}
        except Exception as e:
            existing_names = {}
    else:
        existing_names = {}

    sources_file = sources_fpath.open("at" if skip_existing else "wt")
    for in_fpath in speaker_dir.rglob(pattern):
        out_name = "_".join(
            in_fpath.relative_to(speaker_dir).with_suffix(".npy").parts)
        if skip_existing and out_name in existing_names:
            continue
        out_fpath = speaker_output_dir / out_name
        _process_utterance((in_fpath, out_fpath), processor)
        sources_file.write(f"{out_name},{in_fpath}\n")

    sources_file.close()


def _process_dataset(processor: SpeakerVerificationPreprocessor,
                     datasets_root: Path,
                     speaker_dirs: List[Path],
                     dataset_name: str,
                     output_dir: Path,
                     pattern: str,
                     skip_existing: bool=False):
    print(
        f"{dataset_name}: Preprocessing data for {len(speaker_dirs)} speakers.")

    _func = partial(
        _process_speaker,
        processor=processor,
        datasets_root=datasets_root,
        output_dir=output_dir,
        pattern=pattern,
        skip_existing=skip_existing)

    with mp.Pool(16) as pool:
        list(
            tqdm(
                pool.imap(_func, speaker_dirs),
                dataset_name,
                len(speaker_dirs),
                unit="speakers"))
    print(f"Done preprocessing {dataset_name}.")


def process_librispeech(processor,
                        datasets_root,
                        output_dir,
                        skip_existing=False):
    dataset_name = "LibriSpeech/train-other-500"
    dataset_root = datasets_root / dataset_name
    speaker_dirs = list(dataset_root.glob("*"))
    _process_dataset(processor, datasets_root, speaker_dirs, dataset_name,
                     output_dir, "*.flac", skip_existing)


def process_voxceleb1(processor, datasets_root, output_dir,
                      skip_existing=False):
    dataset_name = "VoxCeleb1"
    dataset_root = datasets_root / dataset_name

    anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]
    with dataset_root.joinpath("vox1_meta.csv").open("rt") as metafile:
        metadata = [line.strip().split("\t") for line in metafile][1:]

    # speaker id -> nationality
    nationalities = {line[0]: line[3] for line in metadata if line[-1] == "dev"}
    keep_speaker_ids = [
        speaker_id for speaker_id, nationality in nationalities.items()
        if nationality.lower() in anglophone_nationalites
    ]
    print(
        "VoxCeleb1: using samples from {} (presumed anglophone) speakers out of {}."
        .format(len(keep_speaker_ids), len(nationalities)))

    speaker_dirs = list((dataset_root / "wav").glob("*"))
    speaker_dirs = [
        speaker_dir for speaker_dir in speaker_dirs
        if speaker_dir.name in keep_speaker_ids
    ]
    _process_dataset(processor, datasets_root, speaker_dirs, dataset_name,
                     output_dir, "*.wav", skip_existing)


def process_voxceleb2(processor, datasets_root, output_dir,
                      skip_existing=False):
    dataset_name = "VoxCeleb2"
    dataset_root = datasets_root / dataset_name
    # There is no nationality in meta data for VoxCeleb2
    speaker_dirs = list((dataset_root / "wav").glob("*"))
    _process_dataset(processor, datasets_root, speaker_dirs, dataset_name,
                     output_dir, "*.wav", skip_existing)


def process_aidatatang_200zh(processor,
                             datasets_root,
                             output_dir,
                             skip_existing=False):
    dataset_name = "aidatatang_200zh/train"
    dataset_root = datasets_root / dataset_name

    speaker_dirs = list((dataset_root).glob("*"))
    _process_dataset(processor, datasets_root, speaker_dirs, dataset_name,
                     output_dir, "*.wav", skip_existing)


def process_magicdata(processor, datasets_root, output_dir,
                      skip_existing=False):
    dataset_name = "magicdata/train"
    dataset_root = datasets_root / dataset_name

    speaker_dirs = list((dataset_root).glob("*"))
    _process_dataset(processor, datasets_root, speaker_dirs, dataset_name,
                     output_dir, "*.wav", skip_existing)
