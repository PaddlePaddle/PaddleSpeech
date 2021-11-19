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
"""Encoder definition."""
import logging

import paddle

from paddlespeech.t2s.modules.conformer.convolution import ConvolutionModule
from paddlespeech.t2s.modules.conformer.encoder_layer import EncoderLayer
from paddlespeech.t2s.modules.layer_norm import LayerNorm
from paddlespeech.t2s.modules.nets_utils import get_activation
from paddlespeech.t2s.modules.transformer.attention import LegacyRelPositionMultiHeadedAttention
from paddlespeech.t2s.modules.transformer.attention import MultiHeadedAttention
from paddlespeech.t2s.modules.transformer.attention import RelPositionMultiHeadedAttention
from paddlespeech.t2s.modules.transformer.embedding import LegacyRelPositionalEncoding
from paddlespeech.t2s.modules.transformer.embedding import PositionalEncoding
from paddlespeech.t2s.modules.transformer.embedding import RelPositionalEncoding
from paddlespeech.t2s.modules.transformer.embedding import ScaledPositionalEncoding
from paddlespeech.t2s.modules.transformer.multi_layer_conv import Conv1dLinear
from paddlespeech.t2s.modules.transformer.multi_layer_conv import MultiLayeredConv1d
from paddlespeech.t2s.modules.transformer.positionwise_feed_forward import PositionwiseFeedForward
from paddlespeech.t2s.modules.transformer.repeat import repeat
from paddlespeech.t2s.modules.transformer.subsampling import Conv2dSubsampling


class Encoder(paddle.nn.Layer):
    """Conformer encoder module.
    Parameters
    ----------
    idim : int
        Input dimension.
    attention_dim : int
        Dimension of attention.
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
    macaron_style : bool
        Whether to use macaron style for positionwise layer.
    pos_enc_layer_type : str
        Encoder positional encoding layer type.
    selfattention_layer_type : str
        Encoder attention layer type.
    activation_type : str
        Encoder activation function type.
    use_cnn_module : bool
        Whether to use convolution module.
    zero_triu : bool
        Whether to zero the upper triangular part of attention matrix.
    cnn_module_kernel : int
        Kernerl size of convolution module.
    padding_idx : int
        Padding idx for input_layer=embed.
    stochastic_depth_rate : float
        Maximum probability to skip the encoder layer.
    intermediate_layers : Union[List[int], None]
        indices of intermediate CTC layer.
        indices start from 1.
        if not None, intermediate outputs are returned (which changes return type
        signature.)
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
            normalize_before=True,
            concat_after=False,
            positionwise_layer_type="linear",
            positionwise_conv_kernel_size=1,
            macaron_style=False,
            pos_enc_layer_type="abs_pos",
            selfattention_layer_type="selfattn",
            activation_type="swish",
            use_cnn_module=False,
            zero_triu=False,
            cnn_module_kernel=31,
            padding_idx=-1,
            stochastic_depth_rate=0.0,
            intermediate_layers=None, ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            pos_enc_class = LegacyRelPositionalEncoding
            assert selfattention_layer_type == "legacy_rel_selfattn"
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        self.conv_subsampling_factor = 1
        if input_layer == "linear":
            self.embed = paddle.nn.Sequential(
                paddle.nn.Linear(idim, attention_dim),
                paddle.nn.LayerNorm(attention_dim),
                paddle.nn.Dropout(dropout_rate),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate), )
            self.conv_subsampling_factor = 4

        elif input_layer == "embed":
            self.embed = paddle.nn.Sequential(
                paddle.nn.Embedding(
                    idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif isinstance(input_layer, paddle.nn.Layer):
            self.embed = paddle.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif input_layer is None:
            self.embed = paddle.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate))
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads, attention_dim,
                                           attention_dropout_rate, )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads, attention_dim,
                                           attention_dropout_rate, )
        elif selfattention_layer_type == "rel_selfattn":
            logging.info(
                "encoder self-attention layer type = relative self-attention")
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads, attention_dim,
                                           attention_dropout_rate, zero_triu, )
        else:
            raise ValueError("unknown encoder_attn_layer: " +
                             selfattention_layer_type)

        # feed-forward module definition
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units,
                                       dropout_rate, activation, )
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

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks, ), )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        self.intermediate_layers = intermediate_layers

    def forward(self, xs, masks):
        """Encode input sequence.
        Parameters
        ----------
        xs : paddle.Tensor
            Input tensor (#batch, time, idim).
            masks (paddle.Tensor): Mask tensor (#batch, 1, time).
        Returns
        ----------
        paddle.Tensor
            Output tensor (#batch, time, attention_dim).
        paddle.Tensor
            Mask tensor (#batch, time).
        """
        if isinstance(self.embed, (Conv2dSubsampling)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        if self.intermediate_layers is None:
            xs, masks = self.encoders(xs, masks)
        else:
            intermediate_outputs = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs, masks = encoder_layer(xs, masks)

                if (self.intermediate_layers is not None and
                        layer_idx + 1 in self.intermediate_layers):
                    # intermediate branches also require normalization.
                    encoder_output = xs
                    if isinstance(encoder_output, tuple):
                        encoder_output = encoder_output[0]
                        if self.normalize_before:
                            encoder_output = self.after_norm(encoder_output)
                    intermediate_outputs.append(encoder_output)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.intermediate_layers is not None:
            return xs, masks, intermediate_outputs
        return xs, masks
