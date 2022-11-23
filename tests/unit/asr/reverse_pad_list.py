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
import unittest

import paddle

import paddlespeech.s2t  # noqa: F401
from paddlespeech.audio.utils.tensor_utils import add_sos_eos
from paddlespeech.audio.utils.tensor_utils import pad_sequence

# from paddlespeech.audio.utils.tensor_utils import reverse_pad_list


def reverse_pad_list(ys_pad: paddle.Tensor,
                     ys_lens: paddle.Tensor,
                     pad_value: float=-1.0) -> paddle.Tensor:
    """Reverse padding for the list of tensors.
    Args:
        ys_pad (tensor): The padded tensor (B, Tokenmax).
        ys_lens (tensor): The lens of token seqs (B)
        pad_value (int): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tokenmax).
    Examples:
        >>> x
        tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> pad_list(x, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])
    """
    r_ys_pad = pad_sequence([(paddle.flip(y[:i], [0]))
                             for y, i in zip(ys_pad, ys_lens)], True, pad_value)
    return r_ys_pad


def naive_reverse_pad_list_with_sos_eos(r_hyps,
                                        r_hyps_lens,
                                        sos=5000,
                                        eos=5000,
                                        ignore_id=-1):
    r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(ignore_id))
    r_hyps, _ = add_sos_eos(r_hyps, sos, eos, ignore_id)
    return r_hyps


def reverse_pad_list_with_sos_eos(r_hyps,
                                  r_hyps_lens,
                                  sos=5000,
                                  eos=5000,
                                  ignore_id=-1):
    #   >>> r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
    #   >>> r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
    max_len = paddle.max(r_hyps_lens)
    index_range = paddle.arange(0, max_len, 1)
    seq_len_expand = r_hyps_lens.unsqueeze(1)
    seq_mask = seq_len_expand > index_range  # (beam, max_len)

    index = (seq_len_expand - 1) - index_range  # (beam, max_len)
    #   >>> index
    #   >>> tensor([[ 2,  1,  0],
    #   >>>         [ 2,  1,  0],
    #   >>>         [ 0, -1, -2]])
    index = index * seq_mask

    #   >>> index
    #   >>> tensor([[2, 1, 0],
    #   >>>         [2, 1, 0],
    #   >>>         [0, 0, 0]])
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

    r_hyps = paddle_gather(r_hyps, 1, index)
    #   >>> r_hyps
    #   >>> tensor([[3, 2, 1],
    #   >>>         [4, 8, 9],
    #   >>>         [2, 2, 2]])
    r_hyps = paddle.where(seq_mask, r_hyps, eos)
    #   >>> r_hyps
    #   >>> tensor([[3, 2, 1],
    #   >>>         [4, 8, 9],
    #   >>>         [2, eos, eos]])
    B = r_hyps.shape[0]
    _sos = paddle.ones([B, 1], dtype=r_hyps.dtype) * sos
    # r_hyps = paddle.concat([hyps[:, 0:1], r_hyps], axis=1)
    r_hyps = paddle.concat([_sos, r_hyps], axis=1)
    #   >>> r_hyps
    #   >>> tensor([[sos, 3, 2, 1],
    #   >>>         [sos, 4, 8, 9],
    #   >>>         [sos, 2, eos, eos]])
    return r_hyps


class TestU2Model(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')

        self.sos = 5000
        self.eos = 5000
        self.ignore_id = -1
        self.reverse_hyps = paddle.to_tensor([[4, 3, 2, 1, -1],
                                              [5, 4, 3, 2, 1]])
        self.reverse_hyps_sos_eos = paddle.to_tensor(
            [[self.sos, 4, 3, 2, 1, self.eos], [self.sos, 5, 4, 3, 2, 1]])

        self.hyps = paddle.to_tensor([[1, 2, 3, 4, -1], [1, 2, 3, 4, 5]])

        self.hyps_lens = paddle.to_tensor([4, 5], paddle.int32)

    def test_reverse_pad_list(self):
        r_hyps = reverse_pad_list(self.hyps, self.hyps_lens)
        self.assertSequenceEqual(r_hyps.tolist(), self.reverse_hyps.tolist())

    def test_naive_reverse_pad_list_with_sos_eos(self):
        r_hyps_sos_eos = naive_reverse_pad_list_with_sos_eos(self.hyps,
                                                             self.hyps_lens)
        self.assertSequenceEqual(r_hyps_sos_eos.tolist(),
                                 self.reverse_hyps_sos_eos.tolist())

    def test_static_reverse_pad_list_with_sos_eos(self):
        r_hyps_sos_eos_static = reverse_pad_list_with_sos_eos(self.hyps,
                                                              self.hyps_lens)
        self.assertSequenceEqual(r_hyps_sos_eos_static.tolist(),
                                 self.reverse_hyps_sos_eos.tolist())


if __name__ == '__main__':
    unittest.main()
