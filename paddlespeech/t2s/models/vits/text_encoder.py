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
"""Text encoder module in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""
import math
from typing import Tuple

import paddle
from paddle import nn

from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask
from paddlespeech.t2s.modules.transformer.encoder import ConformerEncoder as Encoder
from paddlespeech.utils.initialize import normal_


class TextEncoder(nn.Layer):
    """Text encoder module in VITS.

    This is a module of text encoder described in `Conditional Variational Autoencoder
    with Adversarial Learning for End-to-End Text-to-Speech`_.

    Instead of the relative positional Transformer, we use conformer architecture as
    the encoder module, which contains additional convolution layers.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    """

    def __init__(
            self,
            vocabs: int,
            attention_dim: int=192,
            attention_heads: int=2,
            linear_units: int=768,
            blocks: int=6,
            positionwise_layer_type: str="conv1d",
            positionwise_conv_kernel_size: int=3,
            positional_encoding_layer_type: str="rel_pos",
            self_attention_layer_type: str="rel_selfattn",
            activation_type: str="swish",
            normalize_before: bool=True,
            use_macaron_style: bool=False,
            use_conformer_conv: bool=False,
            conformer_kernel_size: int=7,
            dropout_rate: float=0.1,
            positional_dropout_rate: float=0.0,
            attention_dropout_rate: float=0.0, ):
        """Initialize TextEncoder module.

        Args:
            vocabs (int):
                Vocabulary size.
            attention_dim (int):
                Attention dimension.
            attention_heads (int):
                Number of attention heads.
            linear_units (int):
                Number of linear units of positionwise layers.
            blocks (int):
                Number of encoder blocks.
            positionwise_layer_type (str):
                Positionwise layer type.
            positionwise_conv_kernel_size (int):
                Positionwise layer's kernel size.
            positional_encoding_layer_type (str):
                Positional encoding layer type.
            self_attention_layer_type (str):
                Self-attention layer type.
            activation_type (str):
                Activation function type.
            normalize_before (bool):
                Whether to apply LayerNorm before attention.
            use_macaron_style (bool):
                Whether to use macaron style components.
            use_conformer_conv (bool):
                Whether to use conformer conv layers.
            conformer_kernel_size (int):
                Conformer's conv kernel size.
            dropout_rate (float):
                Dropout rate.
            positional_dropout_rate (float):
                Dropout rate for positional encoding.
            attention_dropout_rate (float):
                Dropout rate for attention.

        """
        super().__init__()
        # store for forward
        self.attention_dim = attention_dim

        # define modules
        self.emb = nn.Embedding(vocabs, attention_dim)

        self.encoder = Encoder(
            idim=-1,
            input_layer=None,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            normalize_before=normalize_before,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            macaron_style=use_macaron_style,
            pos_enc_layer_type=positional_encoding_layer_type,
            selfattention_layer_type=self_attention_layer_type,
            activation_type=activation_type,
            use_cnn_module=use_conformer_conv,
            cnn_module_kernel=conformer_kernel_size, )
        self.proj = nn.Conv1D(attention_dim, attention_dim * 2, 1)

        self.reset_parameters()

    def forward(
            self,
            x: paddle.Tensor,
            x_lengths: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Calculate forward propagation.

        Args:
            x (Tensor):
                Input index tensor (B, T_text).
            x_lengths (Tensor):
                Length tensor (B,).

        Returns:
            Tensor:
                Encoded hidden representation (B, attention_dim, T_text).
            Tensor:
                Projected mean tensor (B, attention_dim, T_text).
            Tensor:
                Projected scale tensor (B, attention_dim, T_text).
            Tensor:
                Mask tensor for input tensor (B, 1, T_text).

        """
        x = self.emb(x) * math.sqrt(self.attention_dim)
        x_mask = make_non_pad_mask(x_lengths).unsqueeze(1)
        x_mask = x_mask.astype(x.dtype)
        # encoder assume the channel last (B, T_text, attention_dim)
        # but mask shape shoud be (B, 1, T_text)
        x, _ = self.encoder(x, x_mask)

        # convert the channel first (B, attention_dim, T_text)
        x = paddle.transpose(x, [0, 2, 1])
        stats = self.proj(x) * x_mask
        m, logs = paddle.split(stats, 2, axis=1)

        return x, m, logs, x_mask

    def reset_parameters(self):
        normal_(self.emb.weight, mean=0.0, std=self.attention_dim**-0.5)
        if self.emb._padding_idx is not None:
            with paddle.no_grad():
                self.emb.weight[self.emb._padding_idx] = 0
