# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from https://github.com/fatchord/WaveRNN
import sys
import time
from typing import List

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

from paddlespeech.t2s.audio.codec import decode_mu_law
from paddlespeech.t2s.modules.losses import sample_from_discretized_mix_logistic
from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.upsample import Stretch2D


class ResBlock(nn.Layer):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1D(dims, dims, kernel_size=1, bias_attr=False)
        self.conv2 = nn.Conv1D(dims, dims, kernel_size=1, bias_attr=False)
        self.batch_norm1 = nn.BatchNorm1D(dims)
        self.batch_norm2 = nn.BatchNorm1D(dims)

    def forward(self, x):
        '''
        conv -> bn -> relu -> conv -> bn + residual connection
        '''
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Layer):
    def __init__(self,
                 res_blocks: int=10,
                 compute_dims: int=128,
                 res_out_dims: int=128,
                 aux_channels: int=80,
                 aux_context_window: int=0):
        super().__init__()
        k_size = aux_context_window * 2 + 1
        # pay attention here, the dim reduces aux_context_window * 2
        self.conv_in = nn.Conv1D(
            aux_channels, compute_dims, kernel_size=k_size, bias_attr=False)
        self.batch_norm = nn.BatchNorm1D(compute_dims)
        self.layers = nn.LayerList()
        for _ in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1D(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        '''
        Args:
            x (Tensor): Input tensor (B, in_dims, T).
        Returns:
            Tensor: Output tensor (B, res_out_dims, T).
        '''

        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class UpsampleNetwork(nn.Layer):
    def __init__(self,
                 aux_channels: int=80,
                 upsample_scales: List[int]=[4, 5, 3, 5],
                 compute_dims: int=128,
                 res_blocks: int=10,
                 res_out_dims: int=128,
                 aux_context_window: int=2):
        super().__init__()
        # total_scale is the total Up sampling multiple
        total_scale = np.prod(upsample_scales)
        # TODO pad*total_scale is numpy.int64
        self.indent = int(aux_context_window * total_scale)
        self.resnet = MelResNet(
            res_blocks=res_blocks,
            aux_channels=aux_channels,
            compute_dims=compute_dims,
            res_out_dims=res_out_dims,
            aux_context_window=aux_context_window)
        self.resnet_stretch = Stretch2D(total_scale, 1)
        self.up_layers = nn.LayerList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2D(scale, 1)

            conv = nn.Conv2D(
                1, 1, kernel_size=k_size, padding=padding, bias_attr=False)
            weight_ = paddle.full_like(conv.weight, 1. / k_size[1])
            conv.weight.set_value(weight_)
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        '''
        Args:
            c (Tensor): Input tensor (B, C_aux, T).
        Returns:
            Tensor: Output tensor (B, (T - 2 * pad) *  prob(upsample_scales), C_aux).
            Tensor: Output tensor (B, (T - 2 * pad) *  prob(upsample_scales), res_out_dims).
        '''
        # aux: [B, C_aux, T] 
        # -> [B, res_out_dims, T - 2 * aux_context_window]
        # -> [B, 1, res_out_dims, T - 2 * aux_context_window]
        aux = self.resnet(m).unsqueeze(1)
        # aux: [B, 1, res_out_dims, T - 2 * aux_context_window]
        # -> [B, 1, res_out_dims, (T - 2 * pad) *  prob(upsample_scales)]
        aux = self.resnet_stretch(aux)
        # aux: [B, 1, res_out_dims, T * prob(upsample_scales)] 
        # -> [B, res_out_dims, T * prob(upsample_scales)]
        aux = aux.squeeze(1)
        # m: [B, C_aux, T] -> [B, 1, C_aux, T]
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        # m: [B, 1, C_aux, T*prob(upsample_scales)]
        # -> [B, C_aux, T * prob(upsample_scales)]
        # -> [B, C_aux, (T - 2 * pad) * prob(upsample_scales)]
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        # m: [B, (T - 2 * pad) * prob(upsample_scales), C_aux]
        # aux: [B, (T - 2 * pad) * prob(upsample_scales), res_out_dims]
        return m.transpose([0, 2, 1]), aux.transpose([0, 2, 1])


class WaveRNN(nn.Layer):
    def __init__(
            self,
            rnn_dims: int=512,
            fc_dims: int=512,
            bits: int=9,
            aux_context_window: int=2,
            upsample_scales: List[int]=[4, 5, 3, 5],
            aux_channels: int=80,
            compute_dims: int=128,
            res_out_dims: int=128,
            res_blocks: int=10,
            hop_length: int=300,
            sample_rate: int=24000,
            mode='RAW',
            init_type: str="xavier_uniform", ):
        '''
        Args:
            rnn_dims (int, optional): Hidden dims of RNN Layers.
            fc_dims (int, optional): Dims of FC Layers.
            bits (int, optional): bit depth of signal.
            aux_context_window (int, optional): The context window size of the first convolution applied to the 
                auxiliary input, by default 2
            upsample_scales (List[int], optional): Upsample scales of the upsample network.
            aux_channels (int, optional): Auxiliary channel of the residual blocks.
            compute_dims (int, optional): Dims of Conv1D in MelResNet.
            res_out_dims (int, optional): Dims of output in MelResNet.
            res_blocks (int, optional): Number of residual blocks.
            mode (str, optional): Output mode of the WaveRNN vocoder. 
                `MOL` for Mixture of Logistic Distribution, and `RAW` for quantized bits as the model's output.
            init_type (str): How to initialize parameters.
        '''
        super().__init__()
        self.mode = mode
        self.aux_context_window = aux_context_window
        if self.mode == 'RAW':
            self.n_classes = 2**bits
        elif self.mode == 'MOL':
            self.n_classes = 10 * 3
        else:
            RuntimeError('Unknown model mode value - ', self.mode)

        # List of rnns to call 'flatten_parameters()' on
        self._to_flatten = []

        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # initialize parameters
        initialize(self, init_type)

        self.upsample = UpsampleNetwork(
            aux_channels=aux_channels,
            upsample_scales=upsample_scales,
            compute_dims=compute_dims,
            res_blocks=res_blocks,
            res_out_dims=res_out_dims,
            aux_context_window=aux_context_window)
        self.I = nn.Linear(aux_channels + self.aux_dims + 1, rnn_dims)

        self.rnn1 = nn.GRU(rnn_dims, rnn_dims)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims)

        self._to_flatten += [self.rnn1, self.rnn2]

        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

        nn.initializer.set_global_initializer(None)

    def forward(self, x, c):
        '''
        Args:
            x (Tensor): wav sequence, [B, T]
            c (Tensor): mel spectrogram [B, C_aux, T']

            T = (T' - 2 * aux_context_window ) * hop_length
        Returns:
            Tensor: [B, T, n_classes]
        '''
        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        bsize = paddle.shape(x)[0]
        h1 = paddle.zeros([1, bsize, self.rnn_dims])
        h2 = paddle.zeros([1, bsize, self.rnn_dims])
        # c: [B, T, C_aux]
        # aux: [B, T, res_out_dims]
        c, aux = self.upsample(c)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = paddle.concat([x.unsqueeze(-1), c, a1], axis=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = paddle.concat([x, a2], axis=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = paddle.concat([x, a3], axis=2)
        x = F.relu(self.fc1(x))

        x = paddle.concat([x, a4], axis=2)
        x = F.relu(self.fc2(x))

        return self.fc3(x)

    @paddle.no_grad()
    def generate(self,
                 c,
                 batched: bool=True,
                 target: int=12000,
                 overlap: int=600,
                 mu_law: bool=True,
                 gen_display: bool=False):
        """
        Args:
            c(Tensor): input mels, (T', C_aux)
            batched(bool): generate in batch or not
            target(int): target number of samples to be generated in each batch entry
            overlap(int): number of samples for crossfading between batches
            mu_law(bool)
        Returns: 
            wav sequence: Output (T' * prod(upsample_scales), out_channels, C_out).
        """

        self.eval()

        mu_law = mu_law if self.mode == 'RAW' else False

        output = []
        start = time.time()

        # pseudo batch
        # (T, C_aux) -> (1, C_aux, T)
        c = paddle.transpose(c, [1, 0]).unsqueeze(0)
        T = paddle.shape(c)[-1]
        wave_len = T * self.hop_length
        # TODO remove two transpose op by modifying function pad_tensor
        c = self.pad_tensor(
            c.transpose([0, 2, 1]), pad=self.aux_context_window,
            side='both').transpose([0, 2, 1])

        c, aux = self.upsample(c)

        if batched:
            # (num_folds, target + 2 * overlap, features)
            c = self.fold_with_overlap(c, target, overlap)
            aux = self.fold_with_overlap(aux, target, overlap)

        # for dygraph to static graph, if use seq_len of `b_size, seq_len, _ = paddle.shape(c)` in for
        # will not get TensorArray
        # see https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/case_analysis_cn.html#list-lodtensorarray
        # b_size, seq_len, _ = paddle.shape(c)
        b_size = paddle.shape(c)[0]
        seq_len = paddle.shape(c)[1]

        h1 = paddle.zeros([b_size, self.rnn_dims])
        h2 = paddle.zeros([b_size, self.rnn_dims])
        x = paddle.zeros([b_size, 1])

        d = self.aux_dims
        aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

        for i in range(seq_len):
            m_t = c[:, i, :]
            # for dygraph to static graph
            # a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)
            a1_t = aux_split[0][:, i, :]
            a2_t = aux_split[1][:, i, :]
            a3_t = aux_split[2][:, i, :]
            a4_t = aux_split[3][:, i, :]
            x = paddle.concat([x, m_t, a1_t], axis=1)
            x = self.I(x)
            # use GRUCell here
            h1, _ = self.rnn1[0].cell(x, h1)
            x = x + h1
            inp = paddle.concat([x, a2_t], axis=1)
            # use GRUCell here
            h2, _ = self.rnn2[0].cell(inp, h2)

            x = x + h2
            x = paddle.concat([x, a3_t], axis=1)
            x = F.relu(self.fc1(x))

            x = paddle.concat([x, a4_t], axis=1)
            x = F.relu(self.fc2(x))

            logits = self.fc3(x)

            if self.mode == 'MOL':
                sample = sample_from_discretized_mix_logistic(
                    logits.unsqueeze(0).transpose([0, 2, 1]))
                output.append(sample.reshape([-1]))
                x = sample.transpose([1, 0, 2])

            elif self.mode == 'RAW':
                posterior = F.softmax(logits, axis=1)
                distrib = paddle.distribution.Categorical(posterior)
                # corresponding operate [np.floor((fx + 1) / 2 * mu + 0.5)] in enocde_mu_law
                # distrib.sample([1])[0].cast('float32'): [0, 2**bits-1]
                # sample: [-1, 1]
                sample = 2 * distrib.sample([1])[0].cast('float32') / (
                    self.n_classes - 1.) - 1.
                output.append(sample)
                x = sample.unsqueeze(-1)
            else:
                raise RuntimeError('Unknown model mode value - ', self.mode)

            if gen_display:
                if i % 1000 == 0:
                    self.gen_display(i, int(seq_len), int(b_size), start)

        output = paddle.stack(output).transpose([1, 0])

        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)

        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = paddle.linspace(1, 0, 10 * self.hop_length)
        output = output[:wave_len]
        output[-10 * self.hop_length:] *= fade_out

        self.train()

        # 增加 C_out 维度
        return output.unsqueeze(-1)

    def _flatten_parameters(self):
        [m.flatten_parameters() for m in self._to_flatten]

    def pad_tensor(self, x, pad, side='both'):
        '''
        Args:
            x(Tensor): mel, [1, n_frames, 80]
            pad(int): 
            side(str, optional):  (Default value = 'both')

        Returns:
            Tensor
        '''
        b, t, _ = paddle.shape(x)
        # for dygraph to static graph
        c = x.shape[-1]
        total = t + 2 * pad if side == 'both' else t + pad
        padded = paddle.zeros([b, total, c])
        if side == 'before' or side == 'both':
            padded[:, pad:pad + t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap):
        '''
        Fold the tensor with overlap for quick batched inference.
        Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x(Tensor): Upsampled conditioning features. mels or aux
                shape=(1, T, features)
                mels: [1, T, 80]
                aux: [1, T, 128]
            target(int): Target timesteps for each index of batch
            overlap(int): Timesteps for both xfade and rnn warmup

        Returns:
            Tensor: 
                shape=(num_folds, target + 2 * overlap, features)
                num_flods = (time_seq - overlap) // (target + overlap)
                mel: [num_folds, target + 2 * overlap, 80]
                aux: [num_folds, target + 2 * overlap, 128]

        Details:
            x = [[h1, h2, ... hn]]
            Where each h is a vector of conditioning features
            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                    [h4, h5, h6, h7],
                    [h7, h8, h9, h10]]
        '''

        _, total_len, features = paddle.shape(x)

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')

        folded = paddle.zeros([num_folds, target + 2 * overlap, features])

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[0][start:end, :]
        return folded

    def xfade_and_unfold(self, y, target: int=12000, overlap: int=600):
        ''' Applies a crossfade and unfolds into a 1d array.

        Args:
            y (Tensor): 
                Batched sequences of audio samples
                shape=(num_folds, target + 2 * overlap)
                dtype=paddle.float32
            overlap (int): Timesteps for both xfade and rnn warmup

        Returns:
            Tensor
                audio samples in a 1d array
                shape=(total_len)
                dtype=paddle.float32

        Details:
            y = [[seq1],
                [seq2],
                [seq3]]

            Apply a gain envelope at both ends of the sequences

            y = [[seq1_in, seq1_target, seq1_out],
                [seq2_in, seq2_target, seq2_out],
                [seq3_in, seq3_target, seq3_out]]

            Stagger and add up the groups of samples:

            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

        '''
        # num_folds = (total_len - overlap) // (target + overlap)
        num_folds, length = paddle.shape(y)
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the run warmup
        slience_len = 0
        linear_len = slience_len
        fade_len = overlap - slience_len
        slience = paddle.zeros([slience_len], dtype=paddle.float32)
        linear = paddle.ones([linear_len], dtype=paddle.float32)

        # Equal power crossfade
        # fade_in increase from 0 to 1, fade_out reduces from 1 to 0
        sigmoid_scale = 2.3
        t = paddle.linspace(
            -sigmoid_scale, sigmoid_scale, fade_len, dtype=paddle.float32)
        # sigmoid 曲线应该更好
        fade_in = paddle.nn.functional.sigmoid(t)
        fade_out = 1 - paddle.nn.functional.sigmoid(t)
        # Concat the silence to the fades
        fade_out = paddle.concat([linear, fade_out])
        fade_in = paddle.concat([slience, fade_in])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = paddle.zeros([total_len], dtype=paddle.float32)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def gen_display(self, i, seq_len, b_size, start):
        gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
        pbar = self.progbar(i, seq_len)
        msg = f'| {pbar} {i*b_size}/{seq_len*b_size} | Batch Size: {b_size} | Gen Rate: {gen_rate:.1f}kHz | '
        sys.stdout.write(f"\r{msg}")

    def progbar(self, i, n, size=16):
        done = int(i * size) // n
        bar = ''
        for i in range(size):
            bar += '█' if i <= done else '░'
        return bar


class WaveRNNInference(nn.Layer):
    def __init__(self, normalizer, wavernn):
        super().__init__()
        self.normalizer = normalizer
        self.wavernn = wavernn

    def forward(self,
                logmel,
                batched: bool=True,
                target: int=12000,
                overlap: int=600,
                mu_law: bool=True,
                gen_display: bool=False):
        normalized_mel = self.normalizer(logmel)

        wav = self.wavernn.generate(
            normalized_mel, )
        # batched=batched,
        # target=target,
        # overlap=overlap,
        # mu_law=mu_law,
        # gen_display=gen_display)

        return wav
