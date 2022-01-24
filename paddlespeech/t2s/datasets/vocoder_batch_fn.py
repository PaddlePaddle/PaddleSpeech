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
from pathlib import Path

import numpy as np
import paddle
from paddle.io import Dataset


def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)


def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    # TODO: get rid of log2 - makes no sense
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = paddle.sign(y) / mu * ((1 + mu)**paddle.abs(y) - 1)
    return x


class WaveRNNDataset(Dataset):
    """A simple dataset adaptor for the processed ljspeech dataset."""

    def __init__(self, root):
        self.root = Path(root).expanduser()

        records = []

        with open(self.root / "metadata.csv", 'r') as rf:

            for line in rf:
                name = line.split("\t")[0]
                mel_path = str(self.root / "mel" / (str(name) + ".npy"))
                wav_path = str(self.root / "wav" / (str(name) + ".npy"))
                records.append((mel_path, wav_path))

        self.records = records

    def __getitem__(self, i):
        mel_name, wav_name = self.records[i]
        mel = np.load(mel_name)
        wav = np.load(wav_name)
        return mel, wav

    def __len__(self):
        return len(self.records)


class WaveRNNClip(object):
    def __init__(self,
                 mode: str='RAW',
                 batch_max_steps: int=4500,
                 hop_size: int=300,
                 aux_context_window: int=2,
                 bits: int=9):
        self.mode = mode
        self.mel_win = batch_max_steps // hop_size + 2 * aux_context_window
        self.batch_max_steps = batch_max_steps
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        if self.mode == 'MOL':
            self.bits = 16
        else:
            self.bits = bits

    def __call__(self, batch):
        # batch: [mel, quant]
        # voc_pad = 2  this will pad the input so that the resnet can 'see' wider than input length
        # max_offsets = n_frames - 2 - (mel_win + 2 * hp.voc_pad) = n_frames - 15
        max_offsets = [
            x[0].shape[-1] - 2 - (self.mel_win + 2 * self.aux_context_window)
            for x in batch
        ]
        # the slice point of mel selecting randomly 
        mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        # the slice point of wav selecting randomly, which is behind 2(=pad) frames 
        sig_offsets = [(offset + self.aux_context_window) * self.hop_size
                       for offset in mel_offsets]
        # mels.sape[1] = voc_seq_len // hop_length + 2 * voc_pad
        mels = [
            x[0][:, mel_offsets[i]:mel_offsets[i] + self.mel_win]
            for i, x in enumerate(batch)
        ]
        # label.shape[1] = voc_seq_len + 1
        labels = [
            x[1][sig_offsets[i]:sig_offsets[i] + self.batch_max_steps + 1]
            for i, x in enumerate(batch)
        ]

        mels = np.stack(mels).astype(np.float32)
        labels = np.stack(labels).astype(np.int64)

        mels = paddle.to_tensor(mels)
        labels = paddle.to_tensor(labels, dtype='int64')

        # x is input, y is label
        x = labels[:, :self.batch_max_steps]
        y = labels[:, 1:]
        '''
        mode = RAW:
            mu_law = True:
                quant: bits = 9   0, 1, 2, ..., 509, 510, 511  int
            mu_law = False
                quant bits = 9    [0， 511]  float
        mode = MOL:
            quant: bits = 16  [0. 65536]  float
        '''
        # x should be normalizes in.[0, 1] in RAW mode
        x = label_2_float(paddle.cast(x, dtype='float32'), self.bits)
        # y should be normalizes in.[0, 1] in MOL mode
        if self.mode == 'MOL':
            y = label_2_float(paddle.cast(y, dtype='float32'), self.bits)

        return x, y, mels


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
