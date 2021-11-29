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
from typing import List
from typing import Union

from paddle import nn

from paddlespeech.t2s.modules.activation import get_activation
from paddlespeech.t2s.modules.conformer.convolution import ConvolutionModule
from paddlespeech.t2s.modules.conformer.encoder_layer import EncoderLayer as ConformerEncoderLayer
from paddlespeech.t2s.modules.layer_norm import LayerNorm
from paddlespeech.t2s.modules.transformer.attention import MultiHeadedAttention
from paddlespeech.t2s.modules.transformer.attention import RelPositionMultiHeadedAttention
from paddlespeech.t2s.modules.transformer.embedding import PositionalEncoding
from paddlespeech.t2s.modules.transformer.embedding import RelPositionalEncoding
from paddlespeech.t2s.modules.transformer.embedding import ScaledPositionalEncoding
from paddlespeech.t2s.modules.transformer.encoder_layer import EncoderLayer
from paddlespeech.t2s.modules.transformer.multi_layer_conv import Conv1dLinear
from paddlespeech.t2s.modules.transformer.multi_layer_conv import MultiLayeredConv1d
from paddlespeech.t2s.modules.transformer.positionwise_feed_forward import PositionwiseFeedForward
from paddlespeech.t2s.modules.transformer.repeat import repeat
from paddlespeech.t2s.modules.transformer.subsampling import Conv2dSubsampling


class BaseEncoder(nn.Layer):
    """Base Encoder module.

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
    input_layer : Union[str, nn.Layer]
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
    encoder_type: str
         "transformer", or "conformer".
    """

    def __init__(self,
                 idim: int,
                 attention_dim: int=256,
                 attention_heads: int=4,
                 linear_units: int=2048,
                 num_blocks: int=6,
                 dropout_rate: float=0.1,
                 positional_dropout_rate: float=0.1,
                 attention_dropout_rate: float=0.0,
                 input_layer: str="conv2d",
                 normalize_before: bool=True,
                 concat_after: bool=False,
                 positionwise_layer_type: str="linear",
                 positionwise_conv_kernel_size: int=1,
                 macaron_style: bool=False,
                 pos_enc_layer_type: str="abs_pos",
                 selfattention_layer_type: str="selfattn",
                 activation_type: str="swish",
                 use_cnn_module: bool=False,
                 zero_triu: bool=False,
                 cnn_module_kernel: int=31,
                 padding_idx: int=-1,
                 stochastic_depth_rate: float=0.0,
                 intermediate_layers: Union[List[int], None]=None,
                 encoder_type: str="transformer"):
        """Construct an Base Encoder object."""
        super().__init__()
        activation = get_activation(activation_type)
        pos_enc_class = self.get_pos_enc_class(pos_enc_layer_type,
                                               selfattention_layer_type)
        self.encoder_type = encoder_type

        self.conv_subsampling_factor = 1
        self.embed = self.get_embed(
            idim=idim,
            input_layer=input_layer,
            attention_dim=attention_dim,
            pos_enc_class=pos_enc_class,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            padding_idx=padding_idx)

        self.normalize_before = normalize_before

        # self-attention module definition
        encoder_selfattn_layer, encoder_selfattn_layer_args = self.get_encoder_selfattn_layer(
            selfattention_layer_type=selfattention_layer_type,
            attention_heads=attention_heads,
            attention_dim=attention_dim,
            attention_dropout_rate=attention_dropout_rate,
            zero_triu=zero_triu,
            pos_enc_layer_type=pos_enc_layer_type)
        # feed-forward module definition
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type, attention_dim, linear_units, dropout_rate,
            positionwise_conv_kernel_size, activation)

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        if self.encoder_type == "transformer":
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after, ), )

        elif self.encoder_type == "conformer":
            self.encoders = repeat(
                num_blocks,
                lambda lnum: ConformerEncoderLayer(
                    attention_dim,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                    convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    stochastic_depth_rate * float(1 + lnum) / num_blocks, ), )
            self.intermediate_layers = intermediate_layers
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def get_positionwise_layer(self,
                               positionwise_layer_type: str="linear",
                               attention_dim: int=256,
                               linear_units: int=2048,
                               dropout_rate: float=0.1,
                               positionwise_conv_kernel_size: int=1,
                               activation: nn.Layer=nn.ReLU()):
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units,
                                       dropout_rate, activation)
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

    def get_encoder_selfattn_layer(self,
                                   selfattention_layer_type: str="selfattn",
                                   attention_heads: int=4,
                                   attention_dim: int=256,
                                   attention_dropout_rate: float=0.0,
                                   zero_triu: bool=False,
                                   pos_enc_layer_type: str="abs_pos"):
        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads, attention_dim,
                                           attention_dropout_rate, )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads, attention_dim,
                                           attention_dropout_rate, zero_triu, )
        else:
            raise ValueError("unknown encoder_attn_layer: " +
                             selfattention_layer_type)
        return encoder_selfattn_layer, encoder_selfattn_layer_args

    def get_pos_enc_class(self,
                          pos_enc_layer_type: str="abs_pos",
                          selfattention_layer_type: str="selfattn"):
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
        return pos_enc_class

    def get_embed(self,
                  idim,
                  input_layer="conv2d",
                  attention_dim: int=256,
                  pos_enc_class=PositionalEncoding,
                  dropout_rate: int=0.1,
                  positional_dropout_rate: int=0.1,
                  padding_idx: int=-1):

        if input_layer == "linear":
            embed = nn.Sequential(
                nn.Linear(idim, attention_dim),
                nn.LayerNorm(attention_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif input_layer == "conv2d":
            embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate), )
            self.conv_subsampling_factor = 4
        elif input_layer == "embed":
            embed = nn.Sequential(
                nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif isinstance(input_layer, nn.Layer):
            embed = nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif input_layer is None:
            embed = nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate))
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        return embed

    def forward(self, xs, masks):
        """Encode input sequence.

        Parameters
        ----------
        xs : paddle.Tensor
            Input tensor (#batch, time, idim).
        masks : paddle.Tensor
            Mask tensor (#batch, 1, time).

        Returns
        ----------
        paddle.Tensor
            Output tensor (#batch, time, attention_dim).
        paddle.Tensor
            Mask tensor (#batch, 1, time).
        """
        xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks


class TransformerEncoder(BaseEncoder):
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
    pos_enc_layer_type : str
        Encoder positional encoding layer type.
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
    activation_type : str
        Encoder activation function type.
    padding_idx : int
        Padding idx for input_layer=embed.
    """

    def __init__(
            self,
            idim,
            attention_dim: int=256,
            attention_heads: int=4,
            linear_units: int=2048,
            num_blocks: int=6,
            dropout_rate: float=0.1,
            positional_dropout_rate: float=0.1,
            attention_dropout_rate: float=0.0,
            input_layer: str="conv2d",
            pos_enc_layer_type: str="abs_pos",
            normalize_before: bool=True,
            concat_after: bool=False,
            positionwise_layer_type: str="linear",
            positionwise_conv_kernel_size: int=1,
            selfattention_layer_type: str="selfattn",
            activation_type: str="relu",
            padding_idx: int=-1, ):
        """Construct an Transformer Encoder object."""
        super().__init__(
            idim,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer=input_layer,
            pos_enc_layer_type=pos_enc_layer_type,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            selfattention_layer_type=selfattention_layer_type,
            activation_type=activation_type,
            padding_idx=padding_idx,
            encoder_type="transformer")

    def forward(self, xs, masks):
        """Encode input sequence.

        Parameters
        ----------
        xs : paddle.Tensor
            Input tensor (#batch, time, idim).
        masks : paddle.Tensor
            Mask tensor (#batch, 1, time).

        Returns
        ----------
        paddle.Tensor
            Output tensor (#batch, time, attention_dim).
        paddle.Tensor
            Mask tensor (#batch, 1, time).
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


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module.
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
    input_layer : Union[str, nn.Layer]
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
            idim: int,
            attention_dim: int=256,
            attention_heads: int=4,
            linear_units: int=2048,
            num_blocks: int=6,
            dropout_rate: float=0.1,
            positional_dropout_rate: float=0.1,
            attention_dropout_rate: float=0.0,
            input_layer: str="conv2d",
            normalize_before: bool=True,
            concat_after: bool=False,
            positionwise_layer_type: str="linear",
            positionwise_conv_kernel_size: int=1,
            macaron_style: bool=False,
            pos_enc_layer_type: str="rel_pos",
            selfattention_layer_type: str="rel_selfattn",
            activation_type: str="swish",
            use_cnn_module: bool=False,
            zero_triu: bool=False,
            cnn_module_kernel: int=31,
            padding_idx: int=-1,
            stochastic_depth_rate: float=0.0,
            intermediate_layers: Union[List[int], None]=None, ):
        """Construct an Conformer Encoder object."""
        super().__init__(
            idim=idim,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer=input_layer,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=macaron_style,
            pos_enc_layer_type=pos_enc_layer_type,
            selfattention_layer_type=selfattention_layer_type,
            activation_type=activation_type,
            use_cnn_module=use_cnn_module,
            zero_triu=zero_triu,
            cnn_module_kernel=cnn_module_kernel,
            padding_idx=padding_idx,
            stochastic_depth_rate=stochastic_depth_rate,
            intermediate_layers=intermediate_layers,
            encoder_type="conformer")

    def forward(self, xs, masks):
        """Encode input sequence.
        Parameters
        ----------
        xs : paddle.Tensor
            Input tensor (#batch, time, idim).
        masks : paddle.Tensor
            Mask tensor (#batch, 1, time).
        Returns
        ----------
        paddle.Tensor
            Output tensor (#batch, time, attention_dim).
        paddle.Tensor
            Mask tensor (#batch, 1, time).
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
