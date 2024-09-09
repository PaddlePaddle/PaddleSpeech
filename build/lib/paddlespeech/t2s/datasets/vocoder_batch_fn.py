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

from paddlespeech.t2s.audio.codec import encode_mu_law
from paddlespeech.t2s.audio.codec import float_2_label
from paddlespeech.t2s.audio.codec import label_2_float


class Clip(object):
    """Collate functor for training vocoders.
    """

    def __init__(
            self,
            batch_max_steps=20480,
            hop_size=256,
            aux_context_window=0, ):
        """Initialize customized collater for DataLoader.
        Args:

            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.

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

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features. Audio shape (T, ), features shape(T', C).

        Returns:
            Tensor:
                Target signal batch (B, 1, T).
            Tensor:
                Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
        """
        # check length
        batch = [
            self._adjust_length(b['wave'], b['feats']) for b in batch
            if b['feats'].shape[0] > self.mel_threshold
        ]
        xs, cs = [b[0] for b in batch], [b[1] for b in batch]

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

        # convert each batch to tensor, assume that each item in batch has the same length
        y_batch = paddle.to_tensor(
            y_batch, dtype=paddle.float32).unsqueeze(1)  # (B, 1, T)
        c_batch = paddle.to_tensor(
            c_batch, dtype=paddle.float32).transpose([0, 2, 1])  # (B, C, T')

        return y_batch, c_batch

    def _adjust_length(self, x, c):
        """Adjust the audio and feature lengths.

        Note:
            Basically we assume that the length of x and c are adjusted
            through preprocessing stage, but if we use other library processed
            features, this process will be needed.

        """
        if len(x) < c.shape[0] * self.hop_size:
            x = np.pad(x, (0, c.shape[0] * self.hop_size - len(x)), mode="edge")
        elif len(x) > c.shape[0] * self.hop_size:
            x = x[:c.shape[0] * self.hop_size]
        # check the legnth is valid
        assert len(x) == c.shape[
            0] * self.hop_size, f"wave length: ({len(x)}), mel length: ({c.shape[0]})"

        return x, c


class WaveRNNClip(Clip):
    def __init__(self,
                 mode: str='RAW',
                 batch_max_steps: int=4500,
                 hop_size: int=300,
                 aux_context_window: int=2,
                 bits: int=9,
                 mu_law: bool=True):
        self.mode = mode
        self.mel_win = batch_max_steps // hop_size + 2 * aux_context_window
        self.batch_max_steps = batch_max_steps
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.mu_law = mu_law
        self.batch_max_frames = batch_max_steps // hop_size
        self.mel_threshold = self.batch_max_frames + 2 * aux_context_window
        if self.mode == 'MOL':
            self.bits = 16
        else:
            self.bits = bits

    def to_quant(self, wav):
        if self.mode == 'RAW':
            if self.mu_law:
                quant = encode_mu_law(wav, mu=2**self.bits)
            else:
                quant = float_2_label(wav, bits=self.bits)
        elif self.mode == 'MOL':
            quant = float_2_label(wav, bits=16)
        quant = quant.astype(np.int64)
        return quant

    def __call__(self, batch):
        # voc_pad = 2  this will pad the input so that the resnet can 'see' wider than input length
        # max_offsets = n_frames - 2 - (mel_win + 2 * hp.voc_pad) = n_frames - 15
        """Convert into batch tensors.
        Args:
            batch (list): list of tuple of the pair of audio and features. Audio shape (T, ), features shape(T', C).

        Returns:
            Tensor: Input signal batch (B, 1, T).
            Tensor: Target signal batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), 
                where T = (T' - 2 * aux_context_window) * hop_size.

        """
        # check length
        batch = [
            self._adjust_length(b['wave'], b['feats']) for b in batch
            if b['feats'].shape[0] > self.mel_threshold
        ]
        wav, mel = [b[0] for b in batch], [b[1] for b in batch]
        # mel 此处需要转置
        mel = [x.T for x in mel]
        max_offsets = [
            x.shape[-1] - 2 - (self.mel_win + 2 * self.aux_context_window)
            for x in mel
        ]
        # the slice point of mel selecting randomly 
        mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        # the slice point of wav selecting randomly, which is behind 2(=pad) frames 
        sig_offsets = [(offset + self.aux_context_window) * self.hop_size
                       for offset in mel_offsets]
        # mels.shape[1] = voc_seq_len // hop_length + 2 * voc_pad
        mels = [
            x[:, mel_offsets[i]:mel_offsets[i] + self.mel_win]
            for i, x in enumerate(mel)
        ]
        # label.shape[1] = voc_seq_len + 1
        wav = [self.to_quant(x) for x in wav]

        labels = [
            x[sig_offsets[i]:sig_offsets[i] + self.batch_max_steps + 1]
            for i, x in enumerate(wav)
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


# for paddleslim


class Clip_static(Clip):
    """Collate functor for training vocoders.
    """

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features. Audio shape (T, ), features shape(T', C).

        Returns: 
            Dict[str, np.array]:
                Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
        """
        # check length
        batch = [
            self._adjust_length(b['wave'], b['feats']) for b in batch
            if b['feats'].shape[0] > self.mel_threshold
        ]
        xs, cs = [b[0] for b in batch], [b[1] for b in batch]

        # make batch with random cut
        c_lengths = [c.shape[0] for c in cs]
        start_frames = np.array([
            np.random.randint(self.start_offset, cl + self.end_offset)
            for cl in c_lengths
        ])

        c_starts = start_frames - self.aux_context_window
        c_ends = start_frames + self.batch_max_frames + self.aux_context_window
        c_batch = np.stack(
            [c[start:end] for c, start, end in zip(cs, c_starts, c_ends)])
        # infer axis (T',C) is different with train axis (B, C, T')
        # c_batch = c_batch.transpose([0, 2, 1])  # (B, C, T')
        # do not need batch axis in infer
        c_batch = c_batch[0]
        batch = {"logmel": c_batch}
        return batch
