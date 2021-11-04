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
# Modified from espnet(https://github.com/espnet/espnet)
import numpy as np

from paddlespeech.s2t.io.utility import pad_list
from paddlespeech.s2t.utils.log import Log

__all__ = ["CustomConverter"]

logger = Log(__name__).getlog()


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
            tuple(np.ndarray, nn.ndarray, nn.ndarray)

        """
        # batch should be located in list
        assert len(batch) == 1
        (xs, ys), utts = batch[0]
        assert xs[0] is not None, "please check Reader and Augmentation impl."

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
