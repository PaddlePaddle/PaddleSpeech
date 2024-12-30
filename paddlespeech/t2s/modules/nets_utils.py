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
import math
from typing import Tuple

import numpy as np
import paddle
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.utils.initialize import _calculate_fan_in_and_fan_out
from paddlespeech.utils.initialize import kaiming_uniform_
from paddlespeech.utils.initialize import normal_
from paddlespeech.utils.initialize import ones_
from paddlespeech.utils.initialize import uniform_
from paddlespeech.utils.initialize import zeros_


# default init method of torch
# copy from https://github.com/PaddlePaddle/PaddleSpeech/blob/9cf8c1985a98bb380c183116123672976bdfe5c9/paddlespeech/t2s/models/vits/vits.py#L506
def _reset_parameters(module):
    if isinstance(module, (nn.Conv1D, nn.Conv1DTranspose, nn.Conv2D,
                           nn.Conv2DTranspose)):
        kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(module.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                uniform_(module.bias, -bound, bound)

    if isinstance(module,
                  (nn.BatchNorm1D, nn.BatchNorm2D, nn.GroupNorm, nn.LayerNorm)):
        ones_(module.weight)
        zeros_(module.bias)

    if isinstance(module, nn.Linear):
        kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            uniform_(module.bias, -bound, bound)

    if isinstance(module, nn.Embedding):
        normal_(module.weight)
        if module._padding_idx is not None:
            with paddle.no_grad():
                module.weight[module._padding_idx] = 0


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List[Tensor]): 
            List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): 
            Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [paddle.ones([4]), paddle.ones([2]), paddle.ones([1])]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    max_len = max(x.shape[0] for x in xs)
    pad = paddle.full(
        [n_batch, max_len, *xs[0].shape[1:]], pad_value, dtype=xs[0].dtype)

    for i in range(n_batch):
        pad[i, :xs[i].shape[0]] = xs[i]

    return pad


def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (Tensor(int64)): 
            Batch of lengths (B,).
        xs (Tensor, optional): 
            The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): 
            Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor(bool): Mask tensor containing indices of padded part bool.

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = paddle.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]])
        >>> xs = paddle.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]])

        With the reference tensor and dimension indicator.

        >>> xs = paddle.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]])
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]],)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    # check if lengths is 0-dim tensor, if so, add a dimension
    if lengths.ndim == 0:
        bs = paddle.shape(lengths.unsqueeze(0))
    else:
        bs = paddle.shape(lengths)

    if xs is None:
        maxlen = paddle.cast(lengths.max(), dtype=bs.dtype)
    else:
        maxlen = paddle.shape(xs)[length_dim]

    seq_range = paddle.arange(0, maxlen, dtype=paddle.int64)
    # VITS 最后一个 expand 的位置
    seq_range_expand = seq_range.unsqueeze(0).expand([bs, maxlen])
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand.cast(seq_range_expand.dtype)

    if xs is not None:
        assert paddle.shape(xs)[0] == bs, (paddle.shape(xs)[0], bs)
        if length_dim < 0:
            length_dim = len(paddle.shape(xs)) + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None
            for i in range(len(paddle.shape(xs))))
        mask = paddle.expand(mask[ind], paddle.shape(xs))
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (Tensor(int64) or List): 
            Batch of lengths (B,).
        xs (Tensor, optional): 
            The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): 
            Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor(bool): 
            mask tensor containing indices of padded part bool.

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = paddle.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]])
        >>> xs = paddle.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]])

        With the reference tensor and dimension indicator.

        >>> xs = paddle.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]])
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]])

    """
    return paddle.logical_not(make_pad_mask(lengths, xs, length_dim))


def initialize(model: nn.Layer, init: str):
    """Initialize weights of a neural network module.

    Parameters are initialized using the given method or distribution.

    Custom initialization routines can be implemented into submodules

    Args:
        model (nn.Layer): 
            Target.
        init (str):
            Method of initialization.
    """
    assert check_argument_types()

    if init == "xavier_uniform":
        nn.initializer.set_global_initializer(nn.initializer.XavierUniform(),
                                              nn.initializer.Constant())
    elif init == "xavier_normal":
        nn.initializer.set_global_initializer(nn.initializer.XavierNormal(),
                                              nn.initializer.Constant())
    elif init == "kaiming_uniform":
        nn.initializer.set_global_initializer(nn.initializer.KaimingUniform(),
                                              nn.initializer.Constant())
    elif init == "kaiming_normal":
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(),
                                              nn.initializer.Constant())
    else:
        raise ValueError("Unknown initialization: " + init)


# for VITS
def get_random_segments(
        x: paddle.paddle,
        x_lengths: paddle.Tensor,
        segment_size: int, ) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Get random segments.
    Args:
        x (Tensor): 
            Input tensor (B, C, T).
        x_lengths (Tensor): 
            Length tensor (B,).
        segment_size (int): 
            Segment size.
    Returns:
        Tensor: 
            Segmented tensor (B, C, segment_size).
        Tensor: 
            Start index tensor (B,).
    """
    b, c, t = paddle.shape(x)
    max_start_idx = x_lengths - segment_size
    rand_number = paddle.rand([b])
    start_idxs = paddle.cast(rand_number *
                             max_start_idx.astype(rand_number.dtype), 'int64')
    segments = get_segments(x, start_idxs, segment_size)

    return segments, start_idxs


def get_segments(
        x: paddle.Tensor,
        start_idxs: paddle.Tensor,
        segment_size: int, ) -> paddle.Tensor:
    """Get segments.
    Args:
        x (Tensor): 
            Input tensor (B, C, T).
        start_idxs (Tensor): 
            Start index tensor (B,).
        segment_size (int): 
            Segment size.
    Returns:
        Tensor: Segmented tensor (B, C, segment_size).
    """
    b, c, t = paddle.shape(x)
    segments = paddle.zeros([b, c, segment_size], dtype=x.dtype)
    for i, start_idx in enumerate(start_idxs):
        segments[i] = x[i, :, start_idx:start_idx + segment_size]
    return segments


# see https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/pytorch_project_convertor/API_docs/ops/torch.gather.md
def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


# for ERNIE SAT
# mask phones
def phones_masking(xs_pad: paddle.Tensor,
                   src_mask: paddle.Tensor,
                   align_start: paddle.Tensor,
                   align_end: paddle.Tensor,
                   align_start_lens: paddle.Tensor,
                   mlm_prob: float=0.8,
                   mean_phn_span: int=8,
                   span_bdy: paddle.Tensor=None):
    '''
    Args:
        xs_pad (paddle.Tensor): 
            input speech (B, Tmax, D).
        src_mask (paddle.Tensor): 
            mask of speech (B, 1, Tmax).
        align_start (paddle.Tensor): 
            frame level phone alignment start (B, Tmax2).
        align_end (paddle.Tensor): 
            frame level phone alignment end (B, Tmax2).
        align_start_lens (paddle.Tensor): 
            length of align_start (B, ).
        mlm_prob (float):
        mean_phn_span (int):
        span_bdy (paddle.Tensor): 
            masked mel boundary of input speech (B, 2).
    Returns:
        paddle.Tensor[bool]: masked position of input speech (B, Tmax).
    '''
    bz, sent_len, _ = paddle.shape(xs_pad)
    masked_pos = paddle.zeros((bz, sent_len))
    if mlm_prob == 1.0:
        masked_pos += 1
    elif mean_phn_span == 0:
        # only speech
        length = sent_len
        mean_phn_span = min(length * mlm_prob // 3, 50)
        masked_phn_idxs = random_spans_noise_mask(
            length=length, mlm_prob=mlm_prob,
            mean_phn_span=mean_phn_span).nonzero()
        masked_pos[:, masked_phn_idxs] = 1
    else:
        for idx in range(bz):
            # for inference
            if span_bdy is not None:
                for s, e in zip(span_bdy[idx][::2], span_bdy[idx][1::2]):
                    masked_pos[idx, s:e] = 1
            # for training
            else:
                length = align_start_lens[idx]
                if length < 2:
                    continue
                masked_phn_idxs = random_spans_noise_mask(
                    length=length,
                    mlm_prob=mlm_prob,
                    mean_phn_span=mean_phn_span).nonzero()
                masked_start = align_start[idx][masked_phn_idxs].tolist()
                masked_end = align_end[idx][masked_phn_idxs].tolist()
                for s, e in zip(masked_start, masked_end):
                    masked_pos[idx, s:e] = 1
    non_eos_mask = paddle.reshape(src_mask, paddle.shape(xs_pad)[:2])
    masked_pos = masked_pos * non_eos_mask.astype(masked_pos.dtype)
    masked_pos = paddle.cast(masked_pos, 'bool')

    return masked_pos


# mask speech and phones
def phones_text_masking(xs_pad: paddle.Tensor,
                        src_mask: paddle.Tensor,
                        text_pad: paddle.Tensor,
                        text_mask: paddle.Tensor,
                        align_start: paddle.Tensor,
                        align_end: paddle.Tensor,
                        align_start_lens: paddle.Tensor,
                        mlm_prob: float=0.8,
                        mean_phn_span: int=8,
                        span_bdy: paddle.Tensor=None):
    '''
    Args:
        xs_pad (paddle.Tensor): 
            input speech (B, Tmax, D).
        src_mask (paddle.Tensor): 
            mask of speech (B, 1, Tmax).
        text_pad (paddle.Tensor): 
            input text (B, Tmax2).
        text_mask (paddle.Tensor):
            mask of text (B, 1, Tmax2).
        align_start (paddle.Tensor): 
            frame level phone alignment start (B, Tmax2).
        align_end (paddle.Tensor): 
            frame level phone alignment end (B, Tmax2).
        align_start_lens (paddle.Tensor): 
            length of align_start (B, ).
        mlm_prob (float):
        mean_phn_span (int):
        span_bdy (paddle.Tensor): 
            masked mel boundary of input speech (B, 2).
    Returns:
        paddle.Tensor[bool]: 
            masked position of input speech (B, Tmax).
        paddle.Tensor[bool]: 
            masked position of input text (B, Tmax2).
    '''
    bz, sent_len, _ = paddle.shape(xs_pad)
    masked_pos = paddle.zeros((bz, sent_len))
    _, text_len = paddle.shape(text_pad)
    text_mask_num_lower = math.ceil(text_len * (1 - mlm_prob) * 0.5)
    text_masked_pos = paddle.zeros((bz, text_len))

    if mlm_prob == 1.0:
        masked_pos += 1
    elif mean_phn_span == 0:
        # only speech 
        length = sent_len
        mean_phn_span = min(length * mlm_prob // 3, 50)
        masked_phn_idxs = random_spans_noise_mask(
            length=length, mlm_prob=mlm_prob,
            mean_phn_span=mean_phn_span).nonzero()
        masked_pos[:, masked_phn_idxs] = 1
    else:
        for idx in range(bz):
            # for inference
            if span_bdy is not None:
                for s, e in zip(span_bdy[idx][::2], span_bdy[idx][1::2]):
                    masked_pos[idx, s:e] = 1
            # for training
            else:
                length = align_start_lens[idx]
                if length < 2:
                    continue
                masked_phn_idxs = random_spans_noise_mask(
                    length=length,
                    mlm_prob=mlm_prob,
                    mean_phn_span=mean_phn_span).nonzero()
                unmasked_phn_idxs = list(
                    set(range(length)) - set(masked_phn_idxs[0].tolist()))
                np.random.shuffle(unmasked_phn_idxs)
                masked_text_idxs = unmasked_phn_idxs[:text_mask_num_lower]
                text_masked_pos[idx, masked_text_idxs] = 1
                masked_start = align_start[idx][masked_phn_idxs].tolist()
                masked_end = align_end[idx][masked_phn_idxs].tolist()
                for s, e in zip(masked_start, masked_end):
                    masked_pos[idx, s:e] = 1
    non_eos_mask = paddle.reshape(src_mask, shape=paddle.shape(xs_pad)[:2])
    masked_pos = masked_pos * non_eos_mask.astype(masked_pos.dtype)
    non_eos_text_mask = paddle.reshape(
        text_mask, shape=paddle.shape(text_pad)[:2])
    text_masked_pos = text_masked_pos * non_eos_text_mask.astype(
        text_masked_pos.dtype)
    masked_pos = paddle.cast(masked_pos, 'bool')
    text_masked_pos = paddle.cast(text_masked_pos, 'bool')

    return masked_pos, text_masked_pos


def get_seg_pos(speech_pad: paddle.Tensor,
                text_pad: paddle.Tensor,
                align_start: paddle.Tensor,
                align_end: paddle.Tensor,
                align_start_lens: paddle.Tensor,
                seg_emb: bool=False):
    '''
    Args:
        speech_pad (paddle.Tensor): 
            input speech (B, Tmax, D).
        text_pad (paddle.Tensor): 
            input text (B, Tmax2).
        align_start (paddle.Tensor): 
            frame level phone alignment start (B, Tmax2).
        align_end (paddle.Tensor): 
            frame level phone alignment end (B, Tmax2).
        align_start_lens (paddle.Tensor): 
            length of align_start (B, ).
        seg_emb (bool): 
            whether to use segment embedding.
    Returns:
        paddle.Tensor[int]: n-th phone of each mel, 0<=n<=Tmax2 (B, Tmax).
            eg: 
            Tensor(shape=[1, 328], dtype=int64, place=Place(gpu:0), stop_gradient=True,
            [[0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ,
            1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ,
            1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ,
            1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ,
            1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 3 , 3 , 3 , 4 , 4 , 4 ,
            5 , 5 , 5 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 7 , 7 , 7 , 7 , 7 , 7 , 7 ,
            7 , 8 , 8 , 8 , 8 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 10, 10, 10, 10, 10,
            10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13,
            13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15,
            15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17,
            17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20,
            20, 20, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 23, 23,
            23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25,
            25, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 29,
            29, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 32,
            32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35,
            35, 35, 35, 35, 35, 35, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
            37, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
            38, 38, 0 , 0 ]])
        paddle.Tensor[int]: n-th phone of each phone, 0<=n<=Tmax2 (B, Tmax2).
            eg: 
            Tensor(shape=[1, 38], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                [[1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
                36, 37, 38]])
    '''

    bz, speech_len, _ = paddle.shape(speech_pad)
    _, text_len = paddle.shape(text_pad)

    text_seg_pos = paddle.zeros((bz, text_len), dtype='int64')
    speech_seg_pos = paddle.zeros((bz, speech_len), dtype='int64')

    if not seg_emb:
        return speech_seg_pos, text_seg_pos
    for idx in range(bz):
        align_length = align_start_lens[idx]
        for j in range(align_length):
            s, e = align_start[idx][j], align_end[idx][j]
            speech_seg_pos[idx, s:e] = j + 1
            text_seg_pos[idx, j] = j + 1

    return speech_seg_pos, text_seg_pos


# randomly select the range of speech and text to mask during training
def random_spans_noise_mask(length: int,
                            mlm_prob: float=0.8,
                            mean_phn_span: float=8):
    """This function is copy of `random_spans_helper 
    <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        np.ndarray: a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * mlm_prob))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_phn_span))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_seg(num_items, num_segs):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: 
                an integer scalar > 0
            num_segs: 
                an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segs] containing positive integers that add
            up to num_items
        """
        mask_idxs = np.arange(num_items - 1) < (num_segs - 1)
        np.random.shuffle(mask_idxs)
        first_in_seg = np.pad(mask_idxs, [[1, 0]])
        segment_id = np.cumsum(first_in_seg)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lens = _random_seg(num_noise_tokens, num_noise_spans)
    nonnoise_span_lens = _random_seg(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lens = np.reshape(
        np.stack([nonnoise_span_lens, noise_span_lens], axis=1),
        [num_noise_spans * 2])
    span_starts = np.cumsum(interleaved_span_lens)[:-1]
    span_start_indicator = np.zeros((length, ), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]
