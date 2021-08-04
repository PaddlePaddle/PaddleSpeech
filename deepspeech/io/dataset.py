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
from typing import Optional

from paddle.io import Dataset
from yacs.config import CfgNode

from deepspeech.frontend.utility import read_manifest
from deepspeech.utils.log import Log

__all__ = ["ManifestDataset", "TripletManifestDataset"]

logger = Log(__name__).getlog()


class ManifestDataset(Dataset):
    @classmethod
    def params(cls, config: Optional[CfgNode]=None) -> CfgNode:
        default = CfgNode(
            dict(
                manifest="",
                max_input_len=27.0,
                min_input_len=0.0,
                max_output_len=float('inf'),
                min_output_len=0.0,
                max_output_input_ratio=float('inf'),
                min_output_input_ratio=0.0, ))

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

        dataset = cls(
            manifest_path=config.data.manifest,
            max_input_len=config.data.max_input_len,
            min_input_len=config.data.min_input_len,
            max_output_len=config.data.max_output_len,
            min_output_len=config.data.min_output_len,
            max_output_input_ratio=config.data.max_output_input_ratio,
            min_output_input_ratio=config.data.min_output_input_ratio, )
        return dataset

    def __init__(self,
                 manifest_path,
                 max_input_len=float('inf'),
                 min_input_len=0.0,
                 max_output_len=float('inf'),
                 min_output_len=0.0,
                 max_output_input_ratio=float('inf'),
                 min_output_input_ratio=0.0):
        """Manifest Dataset

        Args:
            manifest_path (str): manifest josn file path
            max_input_len ([type], optional): maximum output seq length, in seconds for raw wav, in frame numbers for feature data. Defaults to float('inf').
            min_input_len (float, optional): minimum input seq length, in seconds for raw wav, in frame numbers for feature data. Defaults to 0.0.
            max_output_len (float, optional): maximum input seq length, in modeling units. Defaults to 500.0.
            min_output_len (float, optional): minimum input seq length, in modeling units. Defaults to 0.0.
            max_output_input_ratio (float, optional): maximum output seq length/output seq length ratio. Defaults to 10.0.
            min_output_input_ratio (float, optional): minimum output seq length/output seq length ratio. Defaults to 0.05.
        
        """
        super().__init__()

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

    def __len__(self):
        return len(self._manifest)

    def __getitem__(self, idx):
        instance = self._manifest[idx]
        return instance["utt"], instance["feat"], instance["text"]


class TripletManifestDataset(ManifestDataset):
    """
        For Joint Training of Speech Translation and ASR.
        text: translation,
        text1: transcript.
    """

    def __getitem__(self, idx):
        instance = self._manifest[idx]
        return instance["utt"], instance["feat"], instance["text"], instance[
            "text1"]
