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
import numpy as np
import paddle


class Clip(object):
    """Collate functor for training vocoders.
    """

    def __init__(
            self,
            batch_max_steps=20480,
            hop_size=256,
            aux_context_window=0, ):
        """Initialize customized collater for DataLoader.

        Parameters
        ----------
        batch_max_steps : int
            The maximum length of input signal in batch.
        hop_size : int
            Hop size of auxiliary features.
        aux_context_window : int
            Context window size for auxiliary feature conv.

        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window

        # set useful values in random cutting
        self.start_offset = aux_context_window
        self.end_offset = -(self.batch_max_frames + aux_context_window)
        self.mel_threshold = self.batch_max_frames + 2 * aux_context_window

    def __call__(self, examples):
        """Convert into batch tensors.

        Parameters
        ----------
        batch : list
            list of tuple of the pair of audio and features. Audio shape (T, ), features shape(T', C).

        Returns
        ----------
        Tensor
            Auxiliary feature batch (B, C, T'), where
            T = (T' - 2 * aux_context_window) * hop_size.
        Tensor
            Target signal batch (B, 1, T).

        """
        # check length
        examples = [
            self._adjust_length(b['wave'], b['feats']) for b in examples
            if b['feats'].shape[0] > self.mel_threshold
        ]
        xs, cs = [b[0] for b in examples], [b[1] for b in examples]

        # make batch with random cut
        c_lengths = [c.shape[0] for c in cs]
        start_frames = np.array([
            np.random.randint(self.start_offset, cl + self.end_offset)
            for cl in c_lengths
        ])
        x_starts = start_frames * self.hop_size
        x_ends = x_starts + self.batch_max_steps

        c_starts = start_frames - self.aux_context_window
        c_ends = start_frames + self.batch_max_frames + self.aux_context_window
        y_batch = np.stack(
            [x[start:end] for x, start, end in zip(xs, x_starts, x_ends)])
        c_batch = np.stack(
            [c[start:end] for c, start, end in zip(cs, c_starts, c_ends)])

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = paddle.to_tensor(
            y_batch, dtype=paddle.float32).unsqueeze(1)  # (B, 1, T)
        c_batch = paddle.to_tensor(
            c_batch, dtype=paddle.float32).transpose([0, 2, 1])  # (B, C, T')

        return y_batch, c_batch

    def _adjust_length(self, x, c):
        """Adjust the audio and feature lengths.

        Note
        -------
        Basically we assume that the length of x and c are adjusted
        through preprocessing stage, but if we use other library processed
        features, this process will be needed.

        """
        if len(x) < c.shape[0] * self.hop_size:
            x = np.pad(x, (0, c.shape[0] * self.hop_size - len(x)), mode="edge")
        elif len(x) > c.shape[0] * self.hop_size:
            # print(
            #     f"wave length: ({len(x)}), mel length: ({c.shape[0]}), hop size: ({self.hop_size })"
            # )
            x = x[:c.shape[0] * self.hop_size]

        # check the legnth is valid
        assert len(x) == c.shape[
            0] * self.hop_size, f"wave length: ({len(x)}), mel length: ({c.shape[0]})"

        return x, c
