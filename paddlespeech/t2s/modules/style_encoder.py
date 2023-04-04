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
"""Style encoder of GST-Tacotron."""
from typing import Sequence

import paddle
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.t2s.modules.transformer.attention import MultiHeadedAttention as BaseMultiHeadedAttention


class StyleEncoder(nn.Layer):
    """Style encoder.

    This module is style encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017
    
    Args:
        idim (int, optional): 
            Dimension of the input mel-spectrogram.
        gst_tokens (int, optional): 
            The number of GST embeddings.
        gst_token_dim (int, optional): 
            Dimension of each GST embedding.
        gst_heads (int, optional): 
            The number of heads in GST multihead attention.
        conv_layers (int, optional): 
            The number of conv layers in the reference encoder.
        conv_chans_list (Sequence[int], optional): 
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional): 
            Kernal size of conv layers in the reference encoder.
        conv_stride (int, optional): 
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): 
            The number of GRU layers in the reference encoder.
        gru_units (int, optional):
            The number of GRU units in the reference encoder.

    Todo:
        * Support manual weight specification in inference.

    """
    def __init__(
        self,
        idim: int = 80,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initilize global style encoder module."""
        assert check_argument_types()
        super().__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )
        self.stl = StyleTokenLayer(
            ref_embed_dim=gru_units,
            gst_tokens=gst_tokens,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
        )

    def forward(self, speech: paddle.Tensor) -> paddle.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): 
                Batch of padded target features (B, Lmax, odim).

        Returns: 
            Tensor: Style token embeddings (B, token_dim).

        """
        ref_embs = self.ref_enc(speech)
        style_embs = self.stl(ref_embs)

        return style_embs


class ReferenceEncoder(nn.Layer):
    """Reference encoder module.

    This module is refernece encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017
    
    Args:
        idim (int, optional): 
            Dimension of the input mel-spectrogram.
        conv_layers (int, optional): 
            The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional): 
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional): 
            Kernal size of conv layers in the reference encoder.
        conv_stride (int, optional): 
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): 
            The number of GRU layers in the reference encoder.
        gru_units (int, optional): 
            The number of GRU units in the reference encoder.

    """
    def __init__(
        self,
        idim=80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initilize reference encoder module."""
        assert check_argument_types()
        super().__init__()

        # check hyperparameters are valid
        assert conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert (
            len(conv_chans_list) == conv_layers
        ), "the number of conv layers and length of channels list must be the same."

        convs = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [
                nn.Conv2D(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias_attr=False,
                ),
                nn.BatchNorm2D(conv_out_chans),
                nn.ReLU(),
            ]
        self.convs = nn.Sequential(*convs)

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        # get the number of GRU input units
        gru_in_units = idim
        for i in range(conv_layers):
            gru_in_units = (gru_in_units - conv_kernel_size +
                            2 * padding) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = nn.GRU(gru_in_units, gru_units, gru_layers, time_major=False)

    def forward(self, speech: paddle.Tensor) -> paddle.Tensor:
        """Calculate forward propagation.
        Args:
            speech (Tensor): 
                Batch of padded target features (B, Lmax, idim).

        Returns:
            Tensor: Reference embedding (B, gru_units)

        """
        batch_size = speech.shape[0]
        # (B, 1, Lmax, idim)
        xs = speech.unsqueeze(1)
        # (B, Lmax', conv_out_chans, idim')
        hs = self.convs(xs).transpose([0, 2, 1, 3])
        time_length = hs.shape[1]
        # (B, Lmax', gru_units)
        hs = hs.reshape(shape=[batch_size, time_length, -1])
        self.gru.flatten_parameters()
        # (gru_layers, batch_size, gru_units)
        _, ref_embs = self.gru(hs)
        # (batch_size, gru_units)
        ref_embs = ref_embs[-1]

        return ref_embs


class StyleTokenLayer(nn.Layer):
    """Style token layer module.

    This module is style token layer introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017
    Args:
        ref_embed_dim (int, optional): 
            Dimension of the input reference embedding.
        gst_tokens (int, optional): 
            The number of GST embeddings.
        gst_token_dim (int, optional): 
            Dimension of each GST embedding.
        gst_heads (int, optional): 
            The number of heads in GST multihead attention.
        dropout_rate (float, optional): 
            Dropout rate in multi-head attention.

    """
    def __init__(
        self,
        ref_embed_dim: int = 128,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initilize style token layer module."""
        assert check_argument_types()
        super().__init__()

        gst_embs = paddle.randn(shape=[gst_tokens, gst_token_dim // gst_heads])
        self.gst_embs = paddle.create_parameter(
            shape=gst_embs.shape,
            dtype=str(gst_embs.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(gst_embs))
        self.mha = MultiHeadedAttention(
            q_dim=ref_embed_dim,
            k_dim=gst_token_dim // gst_heads,
            v_dim=gst_token_dim // gst_heads,
            n_head=gst_heads,
            n_feat=gst_token_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, ref_embs: paddle.Tensor) -> paddle.Tensor:
        """Calculate forward propagation.

        Args:
            ref_embs (Tensor):
                Reference embeddings (B, ref_embed_dim).

        Returns: 
            Tensor: Style token embeddings (B, gst_token_dim).

        """
        batch_size = ref_embs.shape[0]
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        gst_embs = paddle.tanh(self.gst_embs).unsqueeze(0).expand(
            [batch_size, -1, -1])
        # (batch_size, 1 ,ref_embed_dim)
        ref_embs = ref_embs.unsqueeze(1)
        style_embs = self.mha(ref_embs, gst_embs, gst_embs, None)

        return style_embs.squeeze(1)


class MultiHeadedAttention(BaseMultiHeadedAttention):
    """Multi head attention module with different input dimension."""
    def __init__(self, q_dim, k_dim, v_dim, n_head, n_feat, dropout_rate=0.0):
        """Initialize multi head attention module."""
        # Do not use super().__init__() here since we want to
        # overwrite BaseMultiHeadedAttention.__init__() method.
        nn.Layer.__init__(self)
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(q_dim, n_feat)
        self.linear_k = nn.Linear(k_dim, n_feat)
        self.linear_v = nn.Linear(v_dim, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
