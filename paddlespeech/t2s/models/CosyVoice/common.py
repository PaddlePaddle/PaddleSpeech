# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import queue
import random
from typing import List

import numpy as np

def device2str(type=None, index=None, *, device=None):
    type = device if device else type
    if isinstance(type, int):
        type = f'gpu:{type}'
    elif isinstance(type, str):
        if 'cuda' in type:
            type = type.replace('cuda', 'gpu')
        if 'cpu' in type:
            type = 'cpu'
        elif index is not None:
            type = f'{type}:{index}'
    elif isinstance(type, paddle.CPUPlace) or (type is None):
        type = 'cpu'
    elif isinstance(type, paddle.CUDAPlace):
        type = f'gpu:{type.get_device_id()}'

    return type


IGNORE_ID = -1


def pad_list(xs: List[paddle.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = paddle.zeros(batchs, max_len, dtype=xs[0].dtype, device=xs[0].place)
    elif ndim == 2:
        pad_res = paddle.zeros(
            batchs, max_len, xs[0].shape[1], dtype=xs[0].dtype, device=xs[0].place
        )
    elif ndim == 3:
        pad_res = paddle.zeros(
            batchs,
            max_len,
            xs[0].shape[1],
            xs[0].shape[2],
            dtype=xs[0].dtype,
            device=xs[0].place,
        )
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, : len(xs[i])] = xs[i]
    return pad_res


def th_accuracy(
    pad_outputs: paddle.Tensor, pad_targets: paddle.Tensor, ignore_label: int
) -> paddle.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)
    ).argmax(2)
    mask = pad_targets != ignore_label
    numerator = paddle.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = paddle.sum(mask)
    return (numerator / denominator).detach()


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def ras_sampling(
    weighted_scores,
    decoded_tokens,
    sampling,
    top_p=0.8,
    top_k=25,
    win_size=10,
    tau_r=0.1,
):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (
        (paddle.to_tensor(decoded_tokens[-win_size:],dtype = paddle.long).to(weighted_scores.place) == top_ids)
        .sum()
        .item()
    )
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)[0]
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = paddle.sort(
        descending=True, stable=True, x=weighted_scores.softmax(axis=0)
    ), paddle.argsort(descending=True, stable=True, x=weighted_scores.softmax(axis=0))
    
    for i in range(len(sorted_idx)):
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = paddle.to_tensor(prob).cuda()
    indices = paddle.to_tensor(indices, dtype=paddle.long).to(weighted_scores.place)
    # top_ids = indices[prob.multinomial(num_samples=1, replacement=True)]
    top_ids = indices[0]
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(axis=0).multinomial(
        num_samples=1, replacement=True
    )
    return top_ids


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.place
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    if fade_in_mel.place == device2str("cpu"):
        fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel.to(device)


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    paddle.seed(seed)


def mask_to_bias(mask: paddle.Tensor, dtype: paddle.dtype) -> paddle.Tensor:
    assert mask.dtype == paddle.bool
    assert dtype in [paddle.float32, paddle.bfloat16, paddle.float16]
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -10000000000.0
    return mask


class TrtContextWrapper:
    def __init__(self, trt_engine, trt_concurrent=1, device="cuda:0"):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        for _ in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            trt_stream = paddle.device.stream_guard(
                paddle.device.Stream(device=device2str(device))
            )
            assert (
                trt_context is not None
            ), "failed to create trt context, maybe not enough CUDA memory, try reduce current trt concurrent {}".format(
                trt_concurrent
            )
            self.trt_context_pool.put([trt_context, trt_stream])
        assert self.trt_context_pool.empty() is False, "no avaialbe estimator context"

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self.trt_context_pool.put([context, stream])