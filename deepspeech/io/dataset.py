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
            )
        return dataset

    
    def _read_vocab(self, vocab_filepath):
        """Load vocabulary from file."""
        vocab_lines = []
        with open(vocab_filepath, 'r', encoding='utf-8') as file:
            vocab_lines.extend(file.readlines())
        vocab_list = [line[:-1] for line in vocab_lines]
        return vocab_list


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
                 min_output_input_ratio=0.0):
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

        # self._rng = np.random.RandomState(random_seed)

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

        self._vocab_list = self._read_vocab(vocab_filepath)

    @property
    def manifest(self):
        return self._manifest
    
    @property
    def vocab_size(self):
        """Return the vocabulary size.

        Returns:
            int: Vocabulary size.
        """
        return len(self._vocab_list)

    @property
    def vocab_list(self):
        """Return the vocabulary in list.

        Returns:
            List[str]: 
        """
        return self._vocab_list

    @property
    def vocab_dict(self):
        """Return the vocabulary in dict.

        Returns:
            Dict[str, int]: 
        """
        vocab_dict = dict(
            [(token, idx) for (idx, token) in enumerate(self._vocab_list)])
        return vocab_dict

    @property
    def feature_size(self):
        """Return the audio feature size.

        Returns:
            int: audio feature size.
        """
        return self._manifest[0]["feat_shape"][-1]

    

    def __len__(self):
        return len(self._manifest)

    def __getitem__(self, idx):
        instance = self._manifest[idx]
        return instance["utt"], instance["feat"], instance["text"]
