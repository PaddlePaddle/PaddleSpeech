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
import random
from pathlib import Path

import numpy as np
from paddle.io import BatchSampler
from paddle.io import Dataset

from paddlespeech.vector.exps.ge2e.random_cycle import random_cycle


class MultiSpeakerMelDataset(Dataset):
    """A 2 layer directory thatn contains mel spectrograms in *.npy format.
    An Example file structure tree is shown below. We prefer to preprocess
    raw datasets and organized them like this.

    dataset_root/
      speaker1/
        utterance1.npy
        utterance2.npy
        utterance3.npy
      speaker2/
        utterance1.npy
        utterance2.npy
        utterance3.npy
    """

    def __init__(self, dataset_root: Path):
        self.root = Path(dataset_root).expanduser()
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]

        speaker_utterances = {
            speaker_dir: list(speaker_dir.glob("*.npy"))
            for speaker_dir in speaker_dirs
        }

        self.speaker_dirs = speaker_dirs
        self.speaker_to_utterances = speaker_utterances

        # meta data
        self.num_speakers = len(self.speaker_dirs)
        self.num_utterances = np.sum(
            len(utterances)
            for speaker, utterances in self.speaker_to_utterances.items())

    def get_example_by_index(self, speaker_index, utterance_index):
        speaker_dir = self.speaker_dirs[speaker_index]
        fpath = self.speaker_to_utterances[speaker_dir][utterance_index]
        return self[fpath]

    def __getitem__(self, fpath):
        return np.load(fpath)

    def __len__(self):
        return int(self.num_utterances)


class MultiSpeakerSampler(BatchSampler):
    """A multi-stratal sampler designed for speaker verification task.
    First, N speakers from all speakers are sampled randomly. Then, for each
    speaker, randomly sample M utterances from their corresponding utterances.
    """

    def __init__(self,
                 dataset: MultiSpeakerMelDataset,
                 speakers_per_batch: int,
                 utterances_per_speaker: int):
        self._speakers = list(dataset.speaker_dirs)
        self._speaker_to_utterances = dataset.speaker_to_utterances

        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker

    def __iter__(self):
        # yield list of Paths
        speaker_generator = iter(random_cycle(self._speakers))
        speaker_utterances_generator = {
            s: iter(random_cycle(us))
            for s, us in self._speaker_to_utterances.items()
        }

        while True:
            speakers = []
            for _ in range(self.speakers_per_batch):
                speakers.append(next(speaker_generator))

            utterances = []
            for s in speakers:
                us = speaker_utterances_generator[s]
                for _ in range(self.utterances_per_speaker):
                    utterances.append(next(us))
            yield utterances


class RandomClip(object):
    def __init__(self, frames):
        self.frames = frames

    def __call__(self, spec):
        # spec [T, C]
        T = spec.shape[0]
        start = random.randint(0, T - self.frames)
        return spec[start:start + self.frames, :]


class Collate(object):
    def __init__(self, num_frames):
        self.random_crop = RandomClip(num_frames)

    def __call__(self, examples):
        frame_clips = [self.random_crop(mel) for mel in examples]
        batced_clips = np.stack(frame_clips)
        return batced_clips
