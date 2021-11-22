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
from paddle import nn

from paddlespeech.t2s.modules.fastspeech2_transformer.attention import MultiHeadedAttention
from paddlespeech.t2s.modules.fastspeech2_transformer.embedding import PositionalEncoding
from paddlespeech.t2s.modules.fastspeech2_transformer.encoder_layer import EncoderLayer
from paddlespeech.t2s.modules.fastspeech2_transformer.multi_layer_conv import Conv1dLinear
from paddlespeech.t2s.modules.fastspeech2_transformer.multi_layer_conv import MultiLayeredConv1d
from paddlespeech.t2s.modules.fastspeech2_transformer.positionwise_feed_forward import PositionwiseFeedForward
from paddlespeech.t2s.modules.fastspeech2_transformer.repeat import repeat


class Encoder(nn.Layer):
    """Transformer encoder module.

    Parameters
    ----------
    idim : int
        Input dimension.
    attention_dim : int
        Dimention of attention.
    attention_heads : int
        The number of heads of multi head attention.
    linear_units : int
        The number of units of position-wise feed forward.
    num_blocks : int
        The number of decoder blocks.
    dropout_rate : float
        Dropout rate.
    positional_dropout_rate : float
        Dropout rate after adding positional encoding.
    attention_dropout_rate : float
        Dropout rate in attention.
    input_layer : Union[str, paddle.nn.Layer]
        Input layer type.
    pos_enc_class : paddle.nn.Layer
        Positional encoding module class.
        `PositionalEncoding `or `ScaledPositionalEncoding`
    normalize_before : bool
        Whether to use layer_norm before the first block.
    concat_after : bool
        Whether to concat attention layer's input and output.
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    positionwise_layer_type : str
        "linear", "conv1d", or "conv1d-linear".
    positionwise_conv_kernel_size : int
        Kernel size of positionwise conv1d layer.
    selfattention_layer_type : str
        Encoder attention layer type.
    padding_idx : int
        Padding idx for input_layer=embed.
    """

    def __init__(
            self,
            idim,
            attention_dim=256,
            attention_heads=4,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.0,
            input_layer="conv2d",
            pos_enc_class=PositionalEncoding,
            normalize_before=True,
            concat_after=False,
            positionwise_layer_type="linear",
            positionwise_conv_kernel_size=1,
            selfattention_layer_type="selfattn",
            padding_idx=-1, ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self.conv_subsampling_factor = 1
        if input_layer == "linear":
            self.embed = nn.Sequential(
                nn.Linear(idim, attention_dim, bias_attr=True),
                nn.LayerNorm(attention_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif input_layer == "embed":
            self.embed = nn.Sequential(
                nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif isinstance(input_layer, nn.Layer):
            self.embed = nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif input_layer is None:
            self.embed = nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate))
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.normalize_before = normalize_before
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size, )
        if selfattention_layer_type in [
                "selfattn",
                "rel_selfattn",
                "legacy_rel_selfattn",
        ]:
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = [
                (attention_heads, attention_dim, attention_dropout_rate, )
            ] * num_blocks

        else:
            raise NotImplementedError(selfattention_layer_type)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args[lnum]),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after, ), )
        if self.normalize_before:
            self.after_norm = nn.LayerNorm(attention_dim)

    def get_positionwise_layer(
            self,
            positionwise_layer_type="linear",
            attention_dim=256,
            linear_units=2048,
            dropout_rate=0.1,
            positionwise_conv_kernel_size=1, ):
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units,
                                       dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (attention_dim, linear_units,
                                       positionwise_conv_kernel_size,
                                       dropout_rate, )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (attention_dim, linear_units,
                                       positionwise_conv_kernel_size,
                                       dropout_rate, )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        return positionwise_layer, positionwise_layer_args

    def forward(self, xs, masks):
        """Encode input sequence.

        Parameters
        ----------
        xs : paddle.Tensor
            Input tensor (#batch, time, idim).
        masks : paddle.Tensor
            Mask tensor (#batch, time).

        Returns
        ----------
        paddle.Tensor
            Output tensor (#batch, time, attention_dim).
        paddle.Tensor
            Mask tensor (#batch, time).
        """

        xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        Parameters
        ----------
        xs : paddle.Tensor
            Input tensor.
        masks : paddle.Tensor
            Mask tensor.
        cache : List[paddle.Tensor]
            List of cache tensors.

        Returns
        ----------
        paddle.Tensor
            Output tensor.
        paddle.Tensor
            Mask tensor.
        List[paddle.Tensor]
            List of new cache tensors.
        """

        xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
