# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F
from paddlespeech.s2t.utils.log import Log 
logger = Log(__name__).getlog()
def length_to_mask(length, max_len=None, dtype=None):
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().astype(
            'int').item()  # using arange to generate mask
    mask = paddle.arange(
        max_len, dtype=length.dtype).expand(
            (len(length), max_len)) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    mask = paddle.to_tensor(mask, dtype=dtype)
    return mask


class Conv1d(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="same",
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="reflect", ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode

        self.conv = nn.Conv1D(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=groups,
            bias_attr=bias, )

    def forward(self, x):
        if self.padding == "same":
            x = self._manage_padding(x, self.kernel_size, self.dilation,
                                     self.stride)
        else:
            raise ValueError("Padding must be 'same'. Got {self.padding}")

        return self.conv(x)

    def _manage_padding(self, x, kernel_size: int, dilation: int, stride: int):
        L_in = x.shape[-1]  # Detecting input shape
        padding = self._get_padding_elem(L_in, stride, kernel_size,
                                         dilation)  # Time padding
        x = F.pad(
            x, padding, mode=self.padding_mode,
            data_format="NCL")  # Applying padding
        return x

    def _get_padding_elem(self,
                          L_in: int,
                          stride: int,
                          kernel_size: int,
                          dilation: int):
        if stride > 1:
            n_steps = math.ceil(((L_in - kernel_size * dilation) / stride) + 1)
            L_out = stride * (n_steps - 1) + kernel_size * dilation
            padding = [kernel_size // 2, kernel_size // 2]
        else:
            L_out = (L_in - dilation * (kernel_size - 1) - 1) // stride + 1

            padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]

        return padding


class BatchNorm1d(nn.Layer):
    def __init__(
            self,
            input_size,
            eps=1e-05,
            momentum=0.9,
            weight_attr=None,
            bias_attr=None,
            data_format='NCL',
            use_global_stats=None, ):
        super().__init__()

        self.norm = nn.BatchNorm1D(
            input_size,
            epsilon=eps,
            momentum=momentum,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format,
            use_global_stats=use_global_stats, )

    def forward(self, x):
        x_n = self.norm(x)
        return x_n


class TDNNBlock(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            activation=nn.ReLU, ):
        super().__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation, )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))


class Res2NetBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, scale=8, dilation=1):
        super().__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.LayerList([
            TDNNBlock(
                in_channel, hidden_channel, kernel_size=3, dilation=dilation)
            for i in range(scale - 1)
        ])
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(paddle.chunk(x, self.scale, axis=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = paddle.concat(y, axis=1)
        return y


class SEBlock(nn.Layer):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1)
        self.relu = paddle.nn.ReLU()
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L)
            mask = mask.unsqueeze(1)
            total = mask.sum(axis=2, keepdim=True)
            s = (x * mask).sum(axis=2, keepdim=True) / total
        else:
            s = x.mean(axis=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x


class AttentiveStatisticsPooling(nn.Layer):
    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1)

    def forward(self, x, lengths=None):
        C, L = x.shape[1], x.shape[2]  # KP: (N, C, L)

        def _compute_statistics(x, m, axis=2, eps=self.eps):
            mean = (m * x).sum(axis)
            std = paddle.sqrt(
                (m * (x - mean.unsqueeze(axis)).pow(2)).sum(axis).clip(eps))
            return mean, std

        if lengths is None:
            lengths = paddle.ones([x.shape[0]])

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            total = mask.sum(axis=2, keepdim=True).astype('float32')
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).tile((1, 1, L))
            std = std.unsqueeze(2).tile((1, 1, L))
            attn = paddle.concat([x, mean, std], axis=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = paddle.where(
            mask.tile((1, C, 1)) == 0,
            paddle.ones_like(attn) * float("-inf"), attn)

        attn = F.softmax(attn, axis=2)
        mean, std = _compute_statistics(x, attn)

        # Append mean and std of the batch
        pooled_stats = paddle.concat((mean, std), axis=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SERes2NetBlock(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            res2net_scale=8,
            se_channels=128,
            kernel_size=1,
            dilation=1,
            activation=nn.ReLU, ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation, )
        self.res2net_block = Res2NetBlock(out_channels, out_channels,
                                          res2net_scale, dilation)
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation, )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1, )

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual


class EcapaTdnn(nn.Layer):
    def __init__(
            self,
            input_size: int,
            lin_neurons=192,
            activation=nn.ReLU,
            channels=[512, 512, 512, 512, 1536],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            res2net_scale=8,
            se_channels=128,
            global_context=True, ):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.LayerList()
        self.emb_size = lin_neurons
        logger.info("input_size: {}".format(input_size))
        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation, ))

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation, ))

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation, )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context, )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=self.emb_size,
            kernel_size=1, )

    def forward(self, x, lengths=None):
        """
        Compute embeddings.

        Args:
            x (paddle.Tensor): Input log-fbanks with shape (N, n_mels, T).
            lengths (paddle.Tensor, optional): Length proportions of batch length with shape (N). Defaults to None.

        Returns:
            paddle.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = paddle.concat(xl[1:], axis=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        return x
    
    @classmethod
    def init_from_config(cls, config: dict):
        # logger.info("ecapa config: {}".format(config))
        # input_size = config["input_size"]
        # lin_neurons = config.get("lin_neurons", 192)
        # logger.info("input size: {}".format(input_size))
        # logger.info("lin neurons: {}".format(lin_neurons))
        model_conf = config["model_conf"]
        logger.info("model_conf: {}".format(model_conf))
        model = cls(**model_conf)

        return model

class CosClassifier(nn.Layer):
    def __init__(self,
                 input_size,
                 device="cpu",
                 lin_block=0,
                 lin_neurons=192,
                 out_neurons=1211):
        super().__init__()

        self.fc = nn.Linear(in_features=input_size, out_features=out_neurons)
    
    def forward(self, x):
        
        # x 的维度是 [batch, dim, 1]
        x = paddle.squeeze(x, axis=2)
        logits = self.fc(x)
        # logger.info("classifier logits shape: {}".format(logits.shape))
        return logits

    @classmethod
    def init_from_config(cls, config):

        classifier_conf = config["classifier_conf"]
        logger.info("classfier conf: {}".format(classifier_conf))
        classifier = cls(**classifier_conf)

        return classifier

class LogSoftmaxWrapper(nn.Layer):
    """
    Arguments
    ---------
    Returns
    ---------
    loss : torch.Tensor
        Learning loss
    predictions : torch.Tensor
        Log probabilities
    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outputs = outputs.unsqueeze(1)
    >>> targets = torch.tensor([ [0], [1], [0], [1] ])
    >>> log_prob = LogSoftmaxWrapper(nn.Identity())
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> log_prob = LogSoftmaxWrapper(AngularMargin(margin=0.2, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> log_prob = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.3, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    """

    def __init__(self, loss_fn):
        super(LogSoftmaxWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.criterion = paddle.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, length=None):
        """
        Arguments
        ---------
        outputs : torch.Tensor
            Network output tensor, of shape
            [batch, 1, outdim].
        targets : torch.Tensor
            Target tensor, of shape [batch, 1].

        Returns
        -------
        loss: torch.Tensor
            Loss for current examples.
        """
        logger.info("output shape: {}".format(outputs))
        logger.info("target shape: {}".format(targets))
        # outputs = outputs.squeeze(1)
        # targets = targets.squeeze(1)
        targets = nn.functional.one_hot(targets, outputs.shape[1]).float()
        logger.info("one host target: {}".format(targets))
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)
        
        predictions = nn.functional.log_softmax(predictions)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss

class AngularMargin(nn.Layer):
    """
    An implementation of Angular Margin (AM) proposed in the following
    paper: '''Margin Matters: Towards More Discriminative Deep Neural Network
    Embeddings for Speaker Recognition''' (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similiarity
    scale : float
        The scale for cosine similiarity

    Return
    ---------
    predictions : torch.Tensor

    Example
    -------
    >>> pred = AngularMargin()
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0):
        super(AngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets):
        """Compute AM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Return
        ---------
        predictions : torch.Tensor
        """
        outputs = outputs - self.margin * targets
        return self.scale * outputs


class AdditiveAngularMargin(AngularMargin):
    """
    An implementation of Additive Angular Margin (AAM) proposed
    in the following paper: '''Margin Matters: Towards More Discriminative Deep
    Neural Network Embeddings for Speaker Recognition'''
    (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similiarity.
    scale: float
        The scale for cosine similiarity.

    Returns
    -------
    predictions : torch.Tensor
        Tensor.
    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> pred = AdditiveAngularMargin()
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        super(AdditiveAngularMargin, self).__init__(margin, scale)
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        """
        Compute AAM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Return
        ---------
        predictions : torch.Tensor
        """
        cosine = outputs.float()
        sine = paddle.sqrt(1.0 - paddle.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            phi = paddle.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs