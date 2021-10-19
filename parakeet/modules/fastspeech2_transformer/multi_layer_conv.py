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
"""Layer modules for FFT block in FastSpeech (Feed-forward Transformer)."""
import paddle


class MultiLayeredConv1d(paddle.nn.Layer):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed
    to replace positionwise feed-forward network
    in Transforner block, which is introduced in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize MultiLayeredConv1d module.

        Parameters
        ----------
        in_chans : int
            Number of input channels.
        hidden_chans : int
            Number of hidden channels.
        kernel_size : int
            Kernel size of conv1d.
        dropout_rate : float
            Dropout rate.

        """
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = paddle.nn.Conv1D(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2, )
        self.w_2 = paddle.nn.Conv1D(
            hidden_chans,
            in_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2, )
        self.dropout = paddle.nn.Dropout(dropout_rate)
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        """Calculate forward propagation.

        Parameters
        ----------
        x : paddle.Tensor
            Batch of input tensors (B, T, in_chans).

        Returns
        ----------
        paddle.Tensor
            Batch of output tensors (B, T, in_chans).
        """
        x = self.relu(self.w_1(x.transpose([0, 2, 1]))).transpose([0, 2, 1])
        return self.w_2(self.dropout(x).transpose([0, 2, 1])).transpose(
            [0, 2, 1])


class Conv1dLinear(paddle.nn.Layer):
    """Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize Conv1dLinear module.

        Parameters
        ----------
        in_chans : int
            Number of input channels.
        hidden_chans : int
            Number of hidden channels.
        kernel_size : int
            Kernel size of conv1d.
        dropout_rate : float
            Dropout rate.
        """
        super(Conv1dLinear, self).__init__()
        self.w_1 = paddle.nn.Conv1D(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2, )
        self.w_2 = paddle.nn.Linear(hidden_chans, in_chans, bias_attr=True)
        self.dropout = paddle.nn.Dropout(dropout_rate)
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        """Calculate forward propagation.

        Parameters
        ----------
        x : paddle.Tensor
        Batch of input tensors (B, T, in_chans).

        Returns
        ----------
        paddle.Tensor
            Batch of output tensors (B, T, in_chans).

        """
        x = self.relu(self.w_1(x.transpose([0, 2, 1]))).transpose([0, 2, 1])

        return self.w_2(self.dropout(x))
