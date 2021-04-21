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
import io
import tarfile
import time
from collections import namedtuple
from typing import Optional

import numpy as np
from paddle.io import Dataset
from yacs.config import CfgNode

from deepspeech.frontend.augmentor.augmentation import AugmentationPipeline
from deepspeech.frontend.featurizer.speech_featurizer import SpeechFeaturizer
from deepspeech.frontend.normalizer import FeatureNormalizer
from deepspeech.frontend.speech import SpeechSegment
from deepspeech.frontend.utility import read_manifest
from deepspeech.utils.log import Log

__all__ = [
    "ManifestDataset",
]

logger = Log(__name__).getlog()


class ManifestDataset(Dataset):
    @classmethod
    def params(cls, config: Optional[CfgNode]=None) -> CfgNode:
        default = CfgNode(
            dict(
                train_manifest="",
                dev_manifest="",
                test_manifest="",
                manifest="",
                unit_type="char",
                vocab_filepath="",
                spm_model_prefix="",
                mean_std_filepath="",
                augmentation_config="",
                max_input_len=27.0,
                min_input_len=0.0,
                max_output_len=float('inf'),
                min_output_len=0.0,
                max_output_input_ratio=float('inf'),
                min_output_input_ratio=0.0,
                stride_ms=10.0,  # ms
                window_ms=20.0,  # ms
                n_fft=None,  # fft points
                max_freq=None,  # None for samplerate/2
                raw_wav=True,  # use raw_wav or kaldi feature
                specgram_type='linear',  # 'linear', 'mfcc', 'fbank'
                feat_dim=0,  # 'mfcc', 'fbank'
                delta_delta=False,  # 'mfcc', 'fbank'
                target_sample_rate=16000,  # target sample rate
                use_dB_normalization=True,
                target_dB=-20,
                random_seed=0,
                keep_transcription_text=False,
                batch_size=32,  # batch size
                num_workers=0,  # data loader workers
                sortagrad=False,  # sorted in first epoch when True
                shuffle_method="batch_shuffle",  # 'batch_shuffle', 'instance_shuffle'
            ))

        if config is not None:
            config.merge_from_other_cfg(default)
        return default

    @classmethod
    def from_config(cls, config):
        """Build a ManifestDataset object from a config.

        Args:
            config (yacs.config.CfgNode): configs object.

        Returns:
            ManifestDataset: dataet object.
        """
        assert 'manifest' in config.data
        assert config.data.manifest
        assert 'keep_transcription_text' in config.data

        if isinstance(config.data.augmentation_config, (str, bytes)):
            if config.data.augmentation_config:
                aug_file = io.open(
                    config.data.augmentation_config, mode='r', encoding='utf8')
            else:
                aug_file = io.StringIO(initial_value='{}', newline='')
        else:
            aug_file = config.data.augmentation_config
            assert isinstance(aug_file, io.StringIO)

        dataset = cls(
            manifest_path=config.data.manifest,
            unit_type=config.data.unit_type,
            vocab_filepath=config.data.vocab_filepath,
            mean_std_filepath=config.data.mean_std_filepath,
            spm_model_prefix=config.data.spm_model_prefix,
            augmentation_config=aug_file.read(),
            max_input_len=config.data.max_input_len,
            min_input_len=config.data.min_input_len,
            max_output_len=config.data.max_output_len,
            min_output_len=config.data.min_output_len,
            max_output_input_ratio=config.data.max_output_input_ratio,
            min_output_input_ratio=config.data.min_output_input_ratio,
            stride_ms=config.data.stride_ms,
            window_ms=config.data.window_ms,
            n_fft=config.data.n_fft,
            max_freq=config.data.max_freq,
            target_sample_rate=config.data.target_sample_rate,
            specgram_type=config.data.specgram_type,
            feat_dim=config.data.feat_dim,
            delta_delta=config.data.delta_delta,
            use_dB_normalization=config.data.use_dB_normalization,
            target_dB=config.data.target_dB,
            random_seed=config.data.random_seed,
            keep_transcription_text=config.data.keep_transcription_text)
        return dataset

    def __init__(self,
                 manifest_path,
                 unit_type,
                 vocab_filepath,
                 mean_std_filepath,
                 spm_model_prefix=None,
                 augmentation_config='{}',
                 max_input_len=float('inf'),
                 min_input_len=0.0,
                 max_output_len=float('inf'),
                 min_output_len=0.0,
                 max_output_input_ratio=float('inf'),
                 min_output_input_ratio=0.0,
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
            max_input_len ([type], optional): maximum output seq length, in seconds for raw wav, in frame numbers for feature data. Defaults to float('inf').
            min_input_len (float, optional): minimum input seq length, in seconds for raw wav, in frame numbers for feature data. Defaults to 0.0.
            max_output_len (float, optional): maximum input seq length, in modeling units. Defaults to 500.0.
            min_output_len (float, optional): minimum input seq length, in modeling units. Defaults to 0.0.
            max_output_input_ratio (float, optional): maximum output seq length/output seq length ratio. Defaults to 10.0.
            min_output_input_ratio (float, optional): minimum output seq length/output seq length ratio. Defaults to 0.05.
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
        self._max_input_len = max_input_len,
        self._min_input_len = min_input_len,
        self._max_output_len = max_output_len,
        self._min_output_len = min_output_len,
        self._max_output_input_ratio = max_output_input_ratio,
        self._min_output_input_ratio = min_output_input_ratio,

        self._normalizer = FeatureNormalizer(
            mean_std_filepath) if mean_std_filepath else None
        self._augmentation_pipeline = AugmentationPipeline(
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

        self._rng = np.random.RandomState(random_seed)
        self._keep_transcription_text = keep_transcription_text
        # for caching tar files info
        self._local_data = namedtuple('local_data', ['tar2info', 'tar2object'])
        self._local_data.tar2info = {}
        self._local_data.tar2object = {}

        # read manifest
        self._manifest = read_manifest(
            manifest_path=manifest_path,
            max_input_len=max_input_len,
            min_input_len=min_input_len,
            max_output_len=max_output_len,
            min_output_len=min_output_len,
            max_output_input_ratio=max_output_input_ratio,
            min_output_input_ratio=min_output_input_ratio)
        self._manifest.sort(key=lambda x: x["feat_shape"][0])

    @property
    def manifest(self):
        return self._manifest

    @property
    def vocab_size(self):
        return self._speech_featurizer.vocab_size

    @property
    def vocab_list(self):
        return self._speech_featurizer.vocab_list

    @property
    def vocab_dict(self):
        return self._speech_featurizer.vocab_dict

    @property
    def text_feature(self):
        return self._speech_featurizer.text_feature

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
        start_time = time.time()
        if isinstance(audio_file, str) and audio_file.startswith('tar:'):
            speech_segment = SpeechSegment.from_file(
                self._subfile_from_tar(audio_file), transcript)
        else:
            speech_segment = SpeechSegment.from_file(audio_file, transcript)
        load_wav_time = time.time() - start_time
        #logger.debug(f"load wav time: {load_wav_time}")

        # audio augment
        start_time = time.time()
        self._augmentation_pipeline.transform_audio(speech_segment)
        audio_aug_time = time.time() - start_time
        #logger.debug(f"audio augmentation time: {audio_aug_time}")

        start_time = time.time()
        specgram, transcript_part = self._speech_featurizer.featurize(
            speech_segment, self._keep_transcription_text)
        if self._normalizer:
            specgram = self._normalizer.apply(specgram)
        feature_time = time.time() - start_time
        #logger.debug(f"audio & test feature time: {feature_time}")

        # specgram augment
        start_time = time.time()
        specgram = self._augmentation_pipeline.transform_feature(specgram)
        feature_aug_time = time.time() - start_time
        #logger.debug(f"audio feature augmentation time: {feature_aug_time}")
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
