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
import paddle
from paddle.fluid.layers import sequence_mask
from paddle.nn import functional as F

__all__ = [
    "guided_attention_loss",
    "weighted_mean",
    "masked_l1_loss",
    "masked_softmax_with_cross_entropy",
]


def attention_guide(dec_lens, enc_lens, N, T, g, dtype=None):
    """Build that W matrix. shape(B, T_dec, T_enc)
    W[i, n, t] = 1 - exp(-(n/dec_lens[i] - t/enc_lens[i])**2 / (2g**2)) 

    See also:
    Tachibana, Hideyuki, Katsuya Uenoyama, and Shunsuke Aihara. 2017. “Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.” ArXiv:1710.08969 [Cs, Eess], October. http://arxiv.org/abs/1710.08969.
    """
    dtype = dtype or paddle.get_default_dtype()
    dec_pos = paddle.arange(0, N).astype(dtype) / dec_lens.unsqueeze(
        -1)  # n/N # shape(B, T_dec)
    enc_pos = paddle.arange(0, T).astype(dtype) / enc_lens.unsqueeze(
        -1)  # t/T # shape(B, T_enc)
    W = 1 - paddle.exp(-(dec_pos.unsqueeze(-1) - enc_pos.unsqueeze(1))**2 /
                       (2 * g**2))

    dec_mask = sequence_mask(dec_lens, maxlen=N)
    enc_mask = sequence_mask(enc_lens, maxlen=T)
    mask = dec_mask.unsqueeze(-1) * enc_mask.unsqueeze(1)
    mask = paddle.cast(mask, W.dtype)

    W *= mask
    return W


def guided_attention_loss(attention_weight, dec_lens, enc_lens, g):
    """Guided attention loss, masked to excluded padding parts."""
    _, N, T = attention_weight.shape
    W = attention_guide(dec_lens, enc_lens, N, T, g, attention_weight.dtype)

    total_tokens = (dec_lens * enc_lens).astype(W.dtype)
    loss = paddle.mean(paddle.sum(W * attention_weight, [1, 2]) / total_tokens)
    return loss


def weighted_mean(input, weight):
    """Weighted mean. It can also be used as masked mean.

    Parameters
    -----------
    input : Tensor 
        The input tensor.
    weight : Tensor
        The weight tensor with broadcastable shape with the input.

    Returns
    ----------
    Tensor [shape=(1,)]
        Weighted mean tensor with the same dtype as input.
    """
    weight = paddle.cast(weight, input.dtype)
    broadcast_ratio = input.size / weight.size
    return paddle.sum(input * weight) / (paddle.sum(weight) * broadcast_ratio)


def masked_l1_loss(prediction, target, mask):
    """Compute maksed L1 loss.

    Parameters
    ----------
    prediction : Tensor
        The prediction.
    target : Tensor
        The target. The shape should be broadcastable to ``prediction``.
    mask : Tensor
        The mask. The shape should be broadcatable to the broadcasted shape of
        ``prediction`` and ``target``.

    Returns
    -------
    Tensor [shape=(1,)]
        The masked L1 loss.
    """
    abs_error = F.l1_loss(prediction, target, reduction='none')
    loss = weighted_mean(abs_error, mask)
    return loss


def masked_softmax_with_cross_entropy(logits, label, mask, axis=-1):
    """Compute masked softmax with cross entropy loss.

    Parameters
    ----------
    logits : Tensor
        The logits. The ``axis``-th axis is the class dimension.
    label : Tensor [dtype: int]
        The label. The size of the ``axis``-th axis should be 1.
    mask : Tensor 
        The mask. The shape should be broadcastable to ``label``.
    axis : int, optional
        The index of the class dimension in the shape of ``logits``, by default
        -1.

    Returns
    -------
    Tensor [shape=(1,)]
        The masked softmax with cross entropy loss.
    """
    ce = F.softmax_with_cross_entropy(logits, label, axis=axis)
    loss = weighted_mean(ce, mask)
    return loss
