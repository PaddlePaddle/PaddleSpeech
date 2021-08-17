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
from deepspeech.io.utility import pad_list
from deepspeech.utils.log import Log

__all__ = [
    "ManifestDataset", "TripletManifestDataset", "TransformDataset",
    "CustomConverter"
]

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
            max_input_len ([type], optional): maximum output seq length, 
                in seconds for raw wav, in frame numbers for feature data. Defaults to float('inf').
            min_input_len (float, optional): minimum input seq length, 
                in seconds for raw wav, in frame numbers for feature data. Defaults to 0.0.
            max_output_len (float, optional): maximum input seq length, 
                in modeling units. Defaults to 500.0.
            min_output_len (float, optional): minimum input seq length, 
                in modeling units. Defaults to 0.0.
            max_output_input_ratio (float, optional): maximum output seq length/output seq length ratio. 
                Defaults to 10.0.
            min_output_input_ratio (float, optional): minimum output seq length/output seq length ratio.
                Defaults to 0.05.
        
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


class CustomConverter():
    """Custom batch converter.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (np.dtype): Data type to convert.
        
    """

    def __init__(self, subsampling_factor=1, dtype=np.float32):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype

    def __call__(self, batch):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.

        Returns:
            tuple(paddle.Tensor, paddle.Tensor, paddle.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        (xs, ys), utts = batch[0]

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        if xs[0].dtype.kind == "c":
            xs_pad_real = pad_list([x.real for x in xs], 0).astype(self.dtype)
            xs_pad_imag = pad_list([x.imag for x in xs], 0).astype(self.dtype)
            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it E2E here
            # because torch.nn.DataParellel can't handle it.
            xs_pad = {"real": xs_pad_real, "imag": xs_pad_imag}
        else:
            xs_pad = pad_list(xs, 0).astype(self.dtype)

        # NOTE: this is for multi-output (e.g., speech translation)
        ys_pad = pad_list(
            [np.array(y[0][:]) if isinstance(y, tuple) else y for y in ys],
            self.ignore_id)

        olens = np.array(
            [y[0].shape[0] if isinstance(y, tuple) else y.shape[0] for y in ys])
        return utts, xs_pad, ilens, ys_pad, olens


class TransformDataset(Dataset):
    """Transform Dataset.

    Args:
        data: list object from make_batchset
        transfrom: transform function

    """

    def __init__(self, data, transform):
        """Init function."""
        super().__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        """Len function."""
        return len(self.data)

    def __getitem__(self, idx):
        """[] operator."""
        return self.transform(self.data[idx])
