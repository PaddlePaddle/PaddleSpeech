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

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

from paddlespeech.s2t.modules.activation import brelu
from paddlespeech.s2t.modules.mask import make_non_pad_mask
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ['RNNStack']


class RNNCell(nn.RNNCellBase):
    r"""
    Elman RNN (SimpleRNN) cell. Given the inputs and previous states, it
    computes the outputs and updates states.
    The formula used is as follows:
    .. math::
        h_{t} & = act(x_{t} + b_{ih} + W_{hh}h_{t-1} + b_{hh})
        y_{t} & = h_{t}

    where :math:`act` is for :attr:`activation`.
    """

    def __init__(self,
                 hidden_size: int,
                 activation="tanh",
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=None,
                 bias_hh_attr=None,
                 name=None):
        super().__init__()
        std = 1.0 / math.sqrt(hidden_size)
        self.weight_hh = self.create_parameter(
            (hidden_size, hidden_size),
            weight_hh_attr,
            default_initializer=I.Uniform(-std, std))
        self.bias_ih = None
        self.bias_hh = self.create_parameter(
            (hidden_size, ),
            bias_hh_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))

        self.hidden_size = hidden_size
        if activation not in ["tanh", "relu", "brelu"]:
            raise ValueError(
                "activation for SimpleRNNCell should be tanh or relu, "
                "but get {}".format(activation))
        self.activation = activation
        self._activation_fn = paddle.tanh \
            if activation == "tanh" \
            else F.relu
        if activation == 'brelu':
            self._activation_fn = brelu

    def forward(self, inputs, states=None):
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)
        pre_h = states
        i2h = inputs
        if self.bias_ih is not None:
            i2h += self.bias_ih
        h2h = paddle.matmul(pre_h, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            h2h += self.bias_hh
        h = self._activation_fn(i2h + h2h)
        return h, h

    @property
    def state_shape(self):
        return (self.hidden_size, )


class GRUCell(nn.RNNCellBase):
    r"""
    Gated Recurrent Unit (GRU) RNN cell. Given the inputs and previous states,
    it computes the outputs and updates states.
    The formula for GRU used is as follows:
    ..  math::
        r_{t} & = \sigma(W_{ir}x_{t} + b_{ir} + W_{hr}h_{t-1} + b_{hr})
        z_{t} & = \sigma(W_{iz}x_{t} + b_{iz} + W_{hz}h_{t-1} + b_{hz})
        \widetilde{h}_{t} & = \tanh(W_{ic}x_{t} + b_{ic} + r_{t} * (W_{hc}h_{t-1} + b_{hc}))
        h_{t} & = z_{t} * h_{t-1} + (1 - z_{t}) * \widetilde{h}_{t}
        y_{t} & = h_{t}

    where :math:`\sigma` is the sigmoid fucntion, and * is the elemetwise
    multiplication operator.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=None,
                 bias_hh_attr=None,
                 name=None):
        super().__init__()
        std = 1.0 / math.sqrt(hidden_size)
        self.weight_hh = self.create_parameter(
            (3 * hidden_size, hidden_size),
            weight_hh_attr,
            default_initializer=I.Uniform(-std, std))
        self.bias_ih = None
        self.bias_hh = self.create_parameter(
            (3 * hidden_size, ),
            bias_hh_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))

        self.hidden_size = hidden_size
        self.input_size = input_size
        self._gate_activation = F.sigmoid
        self._activation = paddle.tanh

    def forward(self, inputs, states=None):
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)

        pre_hidden = states
        x_gates = inputs
        if self.bias_ih is not None:
            x_gates = x_gates + self.bias_ih
        h_gates = paddle.matmul(pre_hidden, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            h_gates = h_gates + self.bias_hh

        x_r, x_z, x_c = paddle.split(x_gates, num_or_sections=3, axis=1)
        h_r, h_z, h_c = paddle.split(h_gates, num_or_sections=3, axis=1)

        r = self._gate_activation(x_r + h_r)
        z = self._gate_activation(x_z + h_z)
        c = self._activation(x_c + r * h_c)  # apply reset gate after mm
        h = (pre_hidden - c) * z + c
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/layers/dynamic_gru_cn.html#dynamic-gru

        return h, h

    @property
    def state_shape(self):
        r"""
        The `state_shape` of GRUCell is a shape `[hidden_size]` (-1 for batch
        size would be automatically inserted into shape). The shape corresponds
        to the shape of :math:`h_{t-1}`.
        """
        return (self.hidden_size, )


class BiRNNWithBN(nn.Layer):
    """Bidirectonal simple rnn layer with sequence-wise batch normalization.
    The batch normalization is only performed on input-state weights.

    :param size: Dimension of RNN cells.
    :type size: int
    :param share_weights: Whether to share input-hidden weights between
                          forward and backward directional RNNs.
    :type share_weights: bool
    :return: Bidirectional simple rnn layer.
    :rtype: Variable
    """

    def __init__(self, i_size: int, h_size: int, share_weights: bool):
        super().__init__()
        self.share_weights = share_weights
        if self.share_weights:
            #input-hidden weights shared between bi-directional rnn.
            self.fw_fc = nn.Linear(i_size, h_size, bias_attr=False)
            # batch norm is only performed on input-state projection
            self.fw_bn = nn.BatchNorm1D(
                h_size, bias_attr=None, data_format='NLC')
            self.bw_fc = self.fw_fc
            self.bw_bn = self.fw_bn
        else:
            self.fw_fc = nn.Linear(i_size, h_size, bias_attr=False)
            self.fw_bn = nn.BatchNorm1D(
                h_size, bias_attr=None, data_format='NLC')
            self.bw_fc = nn.Linear(i_size, h_size, bias_attr=False)
            self.bw_bn = nn.BatchNorm1D(
                h_size, bias_attr=None, data_format='NLC')

        self.fw_cell = RNNCell(hidden_size=h_size, activation='brelu')
        self.bw_cell = RNNCell(hidden_size=h_size, activation='brelu')
        self.fw_rnn = nn.RNN(
            self.fw_cell, is_reverse=False, time_major=False)  #[B, T, D]
        self.bw_rnn = nn.RNN(
            self.fw_cell, is_reverse=True, time_major=False)  #[B, T, D]

    def forward(self, x: paddle.Tensor, x_len: paddle.Tensor):
        # x, shape [B, T, D]
        fw_x = self.fw_bn(self.fw_fc(x))
        bw_x = self.bw_bn(self.bw_fc(x))
        fw_x, _ = self.fw_rnn(inputs=fw_x, sequence_length=x_len)
        bw_x, _ = self.bw_rnn(inputs=bw_x, sequence_length=x_len)
        x = paddle.concat([fw_x, bw_x], axis=-1)
        return x, x_len


class BiGRUWithBN(nn.Layer):
    """Bidirectonal gru layer with sequence-wise batch normalization.
    The batch normalization is only performed on input-state weights.

    :param name: Name of the layer.
    :type name: string
    :param input: Input layer.
    :type input: Variable
    :param size: Dimension of GRU cells.
    :type size: int
    :param act: Activation type.
    :type act: string
    :return: Bidirectional GRU layer.
    :rtype: Variable
    """

    def __init__(self, i_size: int, h_size: int):
        super().__init__()
        hidden_size = h_size * 3

        self.fw_fc = nn.Linear(i_size, hidden_size, bias_attr=False)
        self.fw_bn = nn.BatchNorm1D(
            hidden_size, bias_attr=None, data_format='NLC')
        self.bw_fc = nn.Linear(i_size, hidden_size, bias_attr=False)
        self.bw_bn = nn.BatchNorm1D(
            hidden_size, bias_attr=None, data_format='NLC')

        self.fw_cell = GRUCell(input_size=hidden_size, hidden_size=h_size)
        self.bw_cell = GRUCell(input_size=hidden_size, hidden_size=h_size)
        self.fw_rnn = nn.RNN(
            self.fw_cell, is_reverse=False, time_major=False)  #[B, T, D]
        self.bw_rnn = nn.RNN(
            self.fw_cell, is_reverse=True, time_major=False)  #[B, T, D]

    def forward(self, x, x_len):
        # x, shape [B, T, D]
        fw_x = self.fw_bn(self.fw_fc(x))
        bw_x = self.bw_bn(self.bw_fc(x))
        fw_x, _ = self.fw_rnn(inputs=fw_x, sequence_length=x_len)
        bw_x, _ = self.bw_rnn(inputs=bw_x, sequence_length=x_len)
        x = paddle.concat([fw_x, bw_x], axis=-1)
        return x, x_len


class RNNStack(nn.Layer):
    """RNN group with stacked bidirectional simple RNN or GRU layers.

    :param input: Input layer.
    :type input: Variable
    :param size: Dimension of RNN cells in each layer.
    :type size: int
    :param num_stacks: Number of stacked rnn layers.
    :type num_stacks: int
    :param use_gru: Use gru if set True. Use simple rnn if set False.
    :type use_gru: bool
    :param share_rnn_weights: Whether to share input-hidden weights between
                              forward and backward directional RNNs.
                              It is only available when use_gru=False.
    :type share_weights: bool
    :return: Output layer of the RNN group.
    :rtype: Variable
    """

    def __init__(self,
                 i_size: int,
                 h_size: int,
                 num_stacks: int,
                 use_gru: bool,
                 share_rnn_weights: bool):
        super().__init__()
        rnn_stacks = []
        for i in range(num_stacks):
            if use_gru:
                #default:GRU using tanh
                rnn_stacks.append(BiGRUWithBN(i_size=i_size, h_size=h_size))
            else:
                rnn_stacks.append(
                    BiRNNWithBN(
                        i_size=i_size,
                        h_size=h_size,
                        share_weights=share_rnn_weights))
            i_size = h_size * 2

        self.rnn_stacks = nn.LayerList(rnn_stacks)

    def forward(self, x: paddle.Tensor, x_len: paddle.Tensor):
        """
        x: shape [B, T, D]
        x_len: shpae [B]
        """
        for i, rnn in enumerate(self.rnn_stacks):
            x, x_len = rnn(x, x_len)
            masks = make_non_pad_mask(x_len)  #[B, T]
            masks = masks.unsqueeze(-1)  # [B, T, 1]
            # TODO(Hui Zhang): not support bool multiply
            masks = masks.astype(x.dtype)
            x = x.multiply(masks)

        return x, x_len
