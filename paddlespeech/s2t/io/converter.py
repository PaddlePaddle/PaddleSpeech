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
    def __init__(self,
                 subsampling_factor=1,
                 dtype=np.float32,
                 load_aux_input=False,
                 load_aux_output=False):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype
        self.load_aux_input = load_aux_input
        self.load_aux_output = load_aux_output

    def __call__(self, batch):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.

        Returns:
            tuple(np.ndarray, nn.ndarray, nn.ndarray)

        """
        # batch should be located in list
        assert len(batch) == 1
        data, utts = batch[0]
        xs_data, ys_data = [], []
        for ud in data:
            if ud[0].ndim > 1:
                # speech data (input): (speech_len, feat_dim)
                xs_data.append(ud)
            else:
                # text data (output): (text_len, )
                ys_data.append(ud)

        assert xs_data[0][
            0] is not None, "please check Reader and Augmentation impl."

        xs_pad, ilens = [], []
        for xs in xs_data:
            # perform subsampling
            if self.subsampling_factor > 1:
                xs = [x[::self.subsampling_factor, :] for x in xs]

            # get batch of lengths of input sequences
            ilens.append(np.array([x.shape[0] for x in xs]))

            # perform padding and convert to tensor
            # currently only support real number
            xs_pad.append(pad_list(xs, 0).astype(self.dtype))

            if not self.load_aux_input:
                xs_pad, ilens = xs_pad[0], ilens[0]
                break

        # NOTE: this is for multi-output (e.g., speech translation)
        ys_pad, olens = [], []

        for ys in ys_data:
            ys_pad.append(
                pad_list([
                    np.array(y[0][:]) if isinstance(y, tuple) else y for y in ys
                ], self.ignore_id))

            olens.append(
                np.array([
                    y[0].shape[0] if isinstance(y, tuple) else y.shape[0]
                    for y in ys
                ]))

            if not self.load_aux_output:
                ys_pad, olens = ys_pad[0], olens[0]
                break

        return utts, xs_pad, ilens, ys_pad, olens
