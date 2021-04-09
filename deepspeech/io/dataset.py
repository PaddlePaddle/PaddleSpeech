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

import math
import random
import tarfile
import logging
import numpy as np
from collections import namedtuple
from functools import partial

from paddle.io import Dataset

from deepspeech.frontend.utility import read_manifest
from deepspeech.frontend.augmentor.augmentation import AugmentationPipeline
from deepspeech.frontend.featurizer.speech_featurizer import SpeechFeaturizer
from deepspeech.frontend.speech import SpeechSegment
from deepspeech.frontend.normalizer import FeatureNormalizer

logger = logging.getLogger(__name__)

__all__ = [
    "ManifestDataset",
]


class ManifestDataset(Dataset):
    def __init__(self,
                 manifest_path,
                 unit_type,
                 vocab_filepath,
                 mean_std_filepath,
                 spm_model_prefix=None,
                 augmentation_config='{}',
                 max_duration=float('inf'),
                 min_duration=0.0,
                 stride_ms=10.0,
                 window_ms=20.0,
                 n_fft=None,
                 max_freq=None,
                 target_sample_rate=16000,
                 specgram_type='linear',
                 feat_dim=None,
                 delta_delta=False,
                 use_dB_normalization=True,
                 target_dB=-20,
                 random_seed=0,
                 keep_transcription_text=False):
        """Manifest Dataset

        Args:
            manifest_path (str): manifest josn file path
            unit_type(str): token unit type, e.g. char, word, spm
            vocab_filepath (str): vocab file path.
            mean_std_filepath (str): mean and std file path, which suffix is *.npy
            spm_model_prefix (str): spm model prefix, need if `unit_type` is spm.
            augmentation_config (str, optional): augmentation json str. Defaults to '{}'.
            max_duration (float, optional): audio length in seconds must less than this. Defaults to float('inf').
            min_duration (float, optional): audio length is seconds must greater than this. Defaults to 0.0.
            stride_ms (float, optional): stride size in ms. Defaults to 10.0.
            window_ms (float, optional): window size in ms. Defaults to 20.0.
            n_fft (int, optional): fft points for rfft. Defaults to None.
            max_freq (int, optional): max cut freq. Defaults to None.
            target_sample_rate (int, optional): target sample rate which used for training. Defaults to 16000.
            specgram_type (str, optional): 'linear', 'mfcc' or 'fbank'. Defaults to 'linear'.
            feat_dim (int, optional): audio feature dim, using by 'mfcc' or 'fbank'. Defaults to None.
            delta_delta (bool, optional): audio feature with delta-delta, using by 'fbank' or 'mfcc'. Defaults to False.
            use_dB_normalization (bool, optional): do dB normalization. Defaults to True.
            target_dB (int, optional): target dB. Defaults to -20.
            random_seed (int, optional): for random generator. Defaults to 0.
            keep_transcription_text (bool, optional): True, when not in training mode, will not do tokenizer; Defaults to False.
        """
        super().__init__()

        self._max_duration = max_duration
        self._min_duration = min_duration
        self._normalizer = FeatureNormalizer(mean_std_filepath)
        self._audio_augmentation_pipeline = AugmentationPipeline(
            augmentation_config=augmentation_config, random_seed=random_seed)
        self._speech_featurizer = SpeechFeaturizer(
            unit_type=unit_type,
            vocab_filepath=vocab_filepath,
            spm_model_prefix=spm_model_prefix,
            specgram_type=specgram_type,
            feat_dim=feat_dim,
            delta_delta=delta_delta,
            stride_ms=stride_ms,
            window_ms=window_ms,
            n_fft=n_fft,
            max_freq=max_freq,
            target_sample_rate=target_sample_rate,
            use_dB_normalization=use_dB_normalization,
            target_dB=target_dB)
        self._rng = random.Random(random_seed)
        self._keep_transcription_text = keep_transcription_text
        # for caching tar files info
        self._local_data = namedtuple('local_data', ['tar2info', 'tar2object'])
        self._local_data.tar2info = {}
        self._local_data.tar2object = {}

        # read manifest
        self._manifest = read_manifest(
            manifest_path=manifest_path,
            max_duration=self._max_duration,
            min_duration=self._min_duration)
        self._manifest.sort(key=lambda x: x["duration"])

    @property
    def manifest(self):
        return self._manifest

    @property
    def vocab_size(self):
        """Return the vocabulary size.

        :return: Vocabulary size.
        :rtype: int
        """
        return self._speech_featurizer.vocab_size

    @property
    def vocab_list(self):
        """Return the vocabulary in list.

        :return: Vocabulary in list.
        :rtype: list
        """
        return self._speech_featurizer.vocab_list

    @property
    def feature_size(self):
        return self._speech_featurizer.feature_size

    def _parse_tar(self, file):
        """Parse a tar file to get a tarfile object
        and a map containing tarinfoes
        """
        result = {}
        f = tarfile.open(file)
        for tarinfo in f.getmembers():
            result[tarinfo.name] = tarinfo
        return f, result

    def _subfile_from_tar(self, file):
        """Get subfile object from tar.

        It will return a subfile object from tar file
        and cached tar file info for next reading request.
        """
        tarpath, filename = file.split(':', 1)[1].split('#', 1)
        if 'tar2info' not in self._local_data.__dict__:
            self._local_data.tar2info = {}
        if 'tar2object' not in self._local_data.__dict__:
            self._local_data.tar2object = {}
        if tarpath not in self._local_data.tar2info:
            object, infoes = self._parse_tar(tarpath)
            self._local_data.tar2info[tarpath] = infoes
            self._local_data.tar2object[tarpath] = object
        return self._local_data.tar2object[tarpath].extractfile(
            self._local_data.tar2info[tarpath][filename])

    def process_utterance(self, audio_file, transcript):
        """Load, augment, featurize and normalize for speech data.

        :param audio_file: Filepath or file object of audio file.
        :type audio_file: str | file
        :param transcript: Transcription text.
        :type transcript: str
        :return: Tuple of audio feature tensor and data of transcription part,
                 where transcription part could be token ids or text.
        :rtype: tuple of (2darray, list)
        """
        if isinstance(audio_file, str) and audio_file.startswith('tar:'):
            speech_segment = SpeechSegment.from_file(
                self._subfile_from_tar(audio_file), transcript)
        else:
            speech_segment = SpeechSegment.from_file(audio_file, transcript)
        self._audio_augmentation_pipeline.transform_audio(speech_segment)
        specgram, transcript_part = self._speech_featurizer.featurize(
            speech_segment, self._keep_transcription_text)
        specgram = self._normalizer.apply(specgram)
        return specgram, transcript_part

    def _instance_reader_creator(self, manifest):
        """
        Instance reader creator. Create a callable function to produce
        instances of data.

        Instance: a tuple of ndarray of audio spectrogram and a list of
        token indices for transcript.
        """

        def reader():
            for instance in manifest:
                inst = self.process_utterance(instance["feat"],
                                              instance["text"])
                yield inst

        return reader

    def __len__(self):
        return len(self._manifest)

    def __getitem__(self, idx):
        instance = self._manifest[idx]
        return self.process_utterance(instance["feat"], instance["text"])
