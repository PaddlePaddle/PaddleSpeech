# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
""" Paddle Wav2Vec2 model."""
import math
import uuid
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import EnumMeta
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

from paddlespeech.s2t.modules.align import Conv1D
from paddlespeech.s2t.modules.align import Conv2D
from paddlespeech.s2t.modules.align import Embedding
from paddlespeech.s2t.modules.align import LayerNorm
from paddlespeech.s2t.modules.align import Linear
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


class GLU(nn.Layer):
    r"""Applies the gated linear unit function
    :math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half
    of the input matrices and :math:`b` is the second half.

    Args:
        axis (int): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = nn.GLU()
        >>> input = paddle.randn([4, 2])
        >>> output = m(input)
    """

    def __init__(self, axis: int=-1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, input: Tensor) -> Tensor:
        return F.glu(input, self.axis)


class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
            self,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
            key: str, ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Layer."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
            self,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
            key: str,
            value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Layer."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState, ) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState)
    return cls


class FairseqDropout(paddle.nn.Layer):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True)
        else:
            return x

    def make_generation_fast_(
            self,
            name: str,
            retain_dropout: bool=False,
            retain_dropout_modules: Optional[List[str]]=None,
            **kwargs, ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    "Cannot enable dropout during inference for module {} "
                    "because module_name was not set".format(name))
            elif (retain_dropout_modules is
                  None  # if None, apply to all modules
                  or self.module_name in retain_dropout_modules):
                logger.info("Enabling dropout during inference for module: {}".
                            format(name))
                self.apply_during_inference = True
            else:
                logger.info("Disabling dropout for module: {}".format(name))


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Layer
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Layer weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (Linear, Embedding, Conv2D))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = len(module.weight.shape) == 4

    # 2D matrix
    if not is_conv:
        if isinstance(module, Linear):
            features_weight = module.weight.shape[0]
        else:
            features_weight = module.weight.shape[1]
        assert (
            features_weight %
            block_size == 0), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.weight.shape[2:] == (1, 1):
            assert (module.weight.shape[1] % block_size == 0
                    ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.weight.shape[2] * module.weight.shape[3]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                if isinstance(module, Linear):
                    in_features = weight.shape[0]
                    out_features = weight.shape[1]
                else:
                    in_features = weight.shape[1]
                    out_features = weight.shape[0]

                # split weight matrix into blocks and randomly drop selected blocks
                mask = paddle.zeros(
                    [in_features // block_size * out_features],
                    dtype=paddle.bool)
                # the implementation of bernoulli_, p=0.5
                mask = paddle.ones_like(mask) * 0.5
                mask = paddle.bernoulli(mask)
                mask = mask.unsqueeze(1).tile([1, block_size]).reshape(
                    [-1, in_features])

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.weight.shape[1]
                out_channels = mod.weight.shape[0]

                # split weight matrix into blocks and randomly drop selected blocks
                if module.weight.shape[2:] == (1, 1):
                    mask = paddle.zeros(
                        [in_channels // block_size * out_channels],
                        dtype=paddle.bool)

                    # the implementation of bernoulli_, p=0.5
                    mask = paddle.ones_like(mask) * 0.5
                    mask = paddle.bernoulli(mask)
                    mask = mask.unsqueeze(1).tile([1, block_size]).reshape(
                        [-1, in_channels])
                else:
                    mask = paddle.zeros(weight.shape)

                    # the implementation of bernoulli_, p=0.5
                    mask = paddle.ones_like(mask) * 0.5
                    mask = paddle.bernoulli(mask)
                    mask = mask.unsqueeze(1).tile([1, in_channels, 1, 1])

            # scale weights and apply mask
            s = 1 / (1 - p)
            mod.weight.set_value(s * weight.masked_fill(mask, 0))

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


@with_incremental_state
class MultiheadAttention(nn.Layer):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=0.0,
            qn_block_size=8,
            # TODO: pass in config rather than string.
            # config defined in xformers.components.attention.AttentionConfig
            xformers_att_config: Optional[str]=None,
            xformers_blocksparse_layout: Optional[
                paddle.Tensor]=None,  # This should be part of the config
            xformers_blocksparse_blocksize: Optional[
                int]=16,  # This should be part of the config
    ):
        super().__init__()

        def eval_str_dict(x, type=dict):
            if x is None:
                return None
            if isinstance(x, str):
                x = eval(x)
            return x

        xformers_att_config = eval_str_dict(xformers_att_config)
        self.use_xformers = xformers_att_config is not None
        assert not self.use_xformers, "Do not use xformers in PaddleSpeech"

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__)

        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim
                ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and "
            "value to be of the same size")

        # Todo scaled initialization
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        weight_attr = nn.initializer.XavierUniform()
        kv_proj_bias_attr = nn.initializer.XavierUniform()
        out_proj_bias_attr = nn.initializer.Constant(0)

        self.k_proj = quant_noise(
            nn.Linear(
                self.kdim,
                embed_dim,
                weight_attr=weight_attr,
                bias_attr=bias
                if not bias else kv_proj_bias_attr), q_noise, qn_block_size)
        self.v_proj = quant_noise(
            nn.Linear(
                self.vdim,
                embed_dim,
                weight_attr=weight_attr,
                bias_attr=bias
                if not bias else kv_proj_bias_attr), q_noise, qn_block_size)
        self.q_proj = quant_noise(
            nn.Linear(
                embed_dim, embed_dim, weight_attr=weight_attr, bias_attr=bias),
            q_noise, qn_block_size)

        self.out_proj = quant_noise(
            nn.Linear(
                embed_dim,
                embed_dim,
                weight_attr=weight_attr,
                bias_attr=bias
                if not bias else out_proj_bias_attr), q_noise, qn_block_size)

        #         nn.initializer.XavierUniform(self.k_proj.weight, gain=1 / math.sqrt(2))
        #         nn.initializer.XavierUniform(self.v_proj.weight, gain=1 / math.sqrt(2))
        #         nn.initializer.XavierUniform(self.q_proj.weight, gain=1 / math.sqrt(2))
        #     else:
        #         self.k_proj.weight = paddle.ParamAttr()
        #     nn.initializer.XavierUniform(self.k_proj.weight)
        #     nn.initializer.XavierUniform(self.v_proj.weight)
        #     nn.initializer.XavierUniform(self.q_proj.weight)

        #     nn.initializer.XavierUniform(self.out_proj.weight)
        # if self.out_proj.bias is not None:
        #     nn.initializer.Constant(self.out_proj.bias)
        # if self.bias_k is not None:
        #     nn.initializer.XavierNormal(self.bias_k)
        # if self.bias_v is not None:
        #     nn.initializer.XavierNormal(self.bias_v)

        # self.k_proj = Linear(self.kdim, embed_dim)

        # self.v_proj = Linear(self.vdim, embed_dim)

        # self.q_proj = Linear(embed_dim, embed_dim)

        # self.out_proj = Linear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = paddle.create_parameter(
                shape=[1, 1, embed_dim],
                dtype='float32',
                initializer=nn.initializer.XavierUniform)
            self.bias_v = paddle.create_parameter(
                shape=[1, 1, embed_dim],
                dtype='float32',
                initializer=nn.initializer.XavierUniform)
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.beam_size = 1
        # self.reset_parameters()

        self.onnx_trace = False
        self.skip_embed_dim_check = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.initializer.XavierUniform(
                self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.initializer.XavierUniform(
                self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.initializer.XavierUniform(
                self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            self.k_proj.weight = paddle.ParamAttr()
            nn.initializer.XavierUniform(self.k_proj.weight)
            nn.initializer.XavierUniform(self.v_proj.weight)
            nn.initializer.XavierUniform(self.q_proj.weight)

            nn.initializer.XavierUniform(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.initializer.Constant(self.out_proj.bias)
        if self.bias_k is not None:
            nn.initializer.XavierNormal(self.bias_k)
        if self.bias_v is not None:
            nn.initializer.XavierNormal(self.bias_v)

    def _get_reserve_head_index(self, num_heads_to_keep: int):
        k_proj_heads_norm = []
        q_proj_heads_norm = []
        v_proj_heads_norm = []

        for i in range(self.num_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            k_proj_heads_norm.append(
                paddle.sum(
                    paddle.abs(self.k_proj.weight[:, start_idx:end_idx]))
                .tolist() + paddle.sum(
                    paddle.abs(self.k_proj.bias[start_idx:end_idx])).tolist())
            q_proj_heads_norm.append(
                paddle.sum(
                    paddle.abs(self.q_proj.weight[:, start_idx:end_idx]))
                .tolist() + paddle.sum(
                    paddle.abs(self.q_proj.bias[start_idx:end_idx])).tolist())
            v_proj_heads_norm.append(
                paddle.sum(
                    paddle.abs(self.v_proj.weight[:, start_idx:end_idx]))
                .tolist() + paddle.sum(
                    paddle.abs(self.v_proj.bias[start_idx:end_idx])).tolist())

        heads_norm = []
        for i in range(self.num_heads):
            heads_norm.append(k_proj_heads_norm[i] + q_proj_heads_norm[i] +
                              v_proj_heads_norm[i])

        sorted_head_index = sorted(
            range(self.num_heads), key=lambda k: heads_norm[k], reverse=True)
        reserve_head_index = []
        for i in range(num_heads_to_keep):
            start = sorted_head_index[i] * self.head_dim
            end = (sorted_head_index[i] + 1) * self.head_dim
            reserve_head_index.append((start, end))

        return reserve_head_index

    def _adaptive_prune_heads(self, reserve_head_index: List[Tuple[int, int]]):
        new_q_weight = []
        new_q_bias = []
        new_k_weight = []
        new_k_bias = []
        new_v_weight = []
        new_v_bias = []
        new_out_proj_weight = []

        for ele in reserve_head_index:
            start_idx, end_idx = ele
            new_q_weight.append(self.q_proj.weight[:, start_idx:end_idx])
            new_q_bias.append(self.q_proj.bias[start_idx:end_idx])

            new_k_weight.append(self.k_proj.weight[:, start_idx:end_idx])

            new_k_bias.append(self.k_proj.bias[start_idx:end_idx])

            new_v_weight.append(self.v_proj.weight[:, start_idx:end_idx])
            new_v_bias.append(self.v_proj.bias[start_idx:end_idx])

            new_out_proj_weight.append(
                self.out_proj.weight[start_idx:end_idx, ])

        new_q_weight = paddle.concat(new_q_weight, axis=-1).detach()
        new_k_weight = paddle.concat(new_k_weight, axis=-1).detach()
        new_v_weight = paddle.concat(new_v_weight, axis=-1).detach()
        new_out_proj_weight = paddle.concat(new_out_proj_weight).detach()
        new_q_weight.stop_gradient = False
        new_k_weight.stop_gradient = False
        new_v_weight.stop_gradient = False
        new_out_proj_weight.stop_gradient = False

        new_q_bias = paddle.concat(new_q_bias).detach()
        new_q_bias.stop_gradient = False

        new_k_bias = paddle.concat(new_k_bias).detach()
        new_k_bias.stop_gradient = False

        new_v_bias = paddle.concat(new_v_bias).detach()
        new_v_bias.stop_gradient = False

        self.q_proj.weight = paddle.create_parameter(
            shape=new_q_weight.shape,
            dtype=new_q_weight.dtype,
            default_initializer=paddle.nn.initializer.Assign(new_q_weight))
        self.q_proj.bias = paddle.create_parameter(
            shape=new_q_bias.shape,
            dtype=new_q_bias.dtype,
            default_initializer=paddle.nn.initializer.Assign(new_q_bias))

        self.k_proj.weight = paddle.create_parameter(
            shape=new_k_weight.shape,
            dtype=new_k_weight.dtype,
            default_initializer=paddle.nn.initializer.Assign(new_k_weight))
        self.k_proj.bias = paddle.create_parameter(
            shape=new_k_bias.shape,
            dtype=new_k_bias.dtype,
            default_initializer=paddle.nn.initializer.Assign(new_k_bias))

        self.v_proj.weight = paddle.create_parameter(
            shape=new_v_weight.shape,
            dtype=new_v_weight.dtype,
            default_initializer=paddle.nn.initializer.Assign(new_v_weight))
        self.v_proj.bias = paddle.create_parameter(
            shape=new_v_bias.shape,
            dtype=new_v_bias.dtype,
            default_initializer=paddle.nn.initializer.Assign(new_v_bias))

        self.out_proj.weight = paddle.create_parameter(
            shape=new_out_proj_weight.shape,
            dtype=new_out_proj_weight.dtype,
            default_initializer=paddle.nn.initializer.Assign(
                new_out_proj_weight))

        self.num_heads = len(reserve_head_index)
        self.embed_dim = self.head_dim * self.num_heads
        self.q_proj.out_features = self.embed_dim
        self.k_proj.out_features = self.embed_dim
        self.v_proj.out_features = self.embed_dim

    def _set_skip_embed_dim_check(self):
        self.skip_embed_dim_check = True

    def _pad_masks(
            self,
            key_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if attn_mask is not None:
            shape = attn_mask.shape[:-1] + [
                1,
            ]
            attn_mask = paddle.concat(
                [attn_mask, paddle.zeros(shape, dtype=attn_mask.dtype)],
                axis=-1)
        if key_padding_mask is not None:
            shape = key_padding_mask.shape[:-1] + [
                1,
            ]
            key_padding_mask = paddle.concat(
                [
                    key_padding_mask, paddle.zeros(
                        shape, dtype=key_padding_mask.dtype)
                ],
                axis=-1)
        return key_padding_mask, attn_mask

    def _add_bias(
            self,
            k: Tensor,
            v: Tensor,
            key_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor],
            bsz: int,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        assert self.bias_k is not None
        assert self.bias_v is not None
        k = paddle.concat([k, self.bias_k.tile([1, bsz, 1])], axis=-1)
        v = paddle.concat([v, self.bias_v.tile([1, bsz, 1])], axis=-1)
        key_padding_mask, attn_mask = self._pad_masks(
            key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return k, v, key_padding_mask, attn_mask

    def _append_zero_attn(
            self,
            k: Tensor,
            v: Tensor,
            key_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        zero_attn_shape = k.shape[:-2] + [1] + k.shape[-1:]
        k = paddle.concat(
            [k, paddle.zeros(zero_attn_shape, dtype=k.dtype)], axis=-2)
        v = paddle.concat(
            [v, paddle.zeros(zero_attn_shape, dtype=v.dtype)], axis=-2)
        key_padding_mask, attn_mask = self._pad_masks(
            key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return k, v, key_padding_mask, attn_mask

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor]=None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[
                Tensor]]]]=None,
            need_weights: bool=True,
            static_kv: bool=False,
            attn_mask: Optional[Tensor]=None,
            before_softmax: bool=False,
            need_head_weights: bool=False, ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.place == "xla"

        tgt_len, bsz, embed_dim = query.shape
        src_len = tgt_len
        if not self.skip_embed_dim_check:
            assert (embed_dim == self.embed_dim
                    ), f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.shape) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.shape
            # if not torch.jit.is_scripting():
            #     assert value is not None
            #     assert src_len, key_bsz == value.shape[:2]

        # if (
        #     not self.onnx_trace
        #     and not is_tpu  # don't use PyTorch version on TPUs
        #     and incremental_state is None
        #     and not static_kv
        #     # A workaround for quantization to work. Otherwise JIT compilation
        #     # treats bias in linear module as method.
        #     and not torch.jit.is_scripting()
        #     # The Multihead attention implemented in pytorch forces strong dimension check
        #     # for input embedding dimention and K,Q,V projection dimension.
        #     # Since pruning will break the dimension check and it is not easy to modify the pytorch API,
        #     # it is preferred to bypass the pytorch MHA when we need to skip embed_dim_check
        #     and not self.skip_embed_dim_check
        # ):
        #     assert key is not None and value is not None

        # if self.use_xformers:
        #     return self._xformers_attn_forward(
        #         query, key, value, key_padding_mask, need_weights, attn_mask
        #     )

        # else:
        #     return F.multi_head_attention_forward(
        #         query,
        #         key,
        #         value,
        #         self.embed_dim,
        #         self.num_heads,
        #         torch.empty([0]),
        #         torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
        #         self.bias_k,
        #         self.bias_v,
        #         self.add_zero_attn,
        #         self.dropout_module.p,
        #         self.out_proj.weight,
        #         self.out_proj.bias,
        #         self.training or self.dropout_module.apply_during_inference,
        #         key_padding_mask,
        #         need_weights,
        #         attn_mask,
        #         use_separate_proj_weight=True,
        #         q_proj_weight=self.q_proj.weight,
        #         k_proj_weight=self.k_proj.weight,
        #         v_proj_weight=self.v_proj.weight,
        #     )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.beam_size > 1 and bsz == key.size(1):
                    # key is [T, bsz*beam_size, C], reduce to [T, bsz, C]
                    key = key.reshape(
                        [key.size(0), -1, self.beam_size,
                         key.size(2)])[:, :, 0, :]
                    if key_padding_mask is not None:
                        key_padding_mask = key_padding_mask.reshape(
                            [-1, self.beam_size,
                             key_padding_mask.size(1)])[:, 0, :]
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(
                k, v, attn_mask, key_padding_mask, bsz)

        q = paddle.reshape(
            q, [tgt_len, bsz * self.num_heads, self.head_dim]).transpose(
                [1, 0, 2])
        kv_bsz = bsz  # need default value for scripting
        if k is not None:
            kv_bsz = k.shape[1]
            k = paddle.reshape(
                k, [-1, kv_bsz * self.num_heads, self.head_dim]).transpose(
                    [1, 0, 2])
        if v is not None:
            v = paddle.reshape(
                v, [-1, kv_bsz * self.num_heads, self.head_dim]).transpose(
                    [1, 0, 2])

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                kv_bsz = _prev_key.shape[0]
                prev_key = _prev_key.reshape(
                    [kv_bsz * self.num_heads, -1, self.head_dim])
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = paddle.concat([prev_key, k], axis=1)
                src_len = k.shape[1]
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                assert kv_bsz == _prev_value.size(0)
                prev_value = _prev_value.reshape(
                    [kv_bsz * self.num_heads, -1, self.head_dim])
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = paddle.concat([prev_value, v], axis=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=kv_bsz,
                src_len=k.shape[1],
                static_kv=static_kv, )

            saved_state["prev_key"] = k.reshape(
                [kv_bsz, self.num_heads, -1, self.head_dim])
            saved_state["prev_value"] = v.reshape(
                [kv_bsz, self.num_heads, -1, self.head_dim])
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state,
                                                       saved_state)
        assert k is not None
        assert k.shape[1] == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == kv_bsz
            assert key_padding_mask.shape[1] == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(
                k=k,
                v=v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask)

        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn_weights = paddle.einsum(
                "bxhtd,bhsd->bxhts",
                q.reshape([kv_bsz, -1, self.num_heads] + q.shape[1:]),
                k.reshape([kv_bsz, self.num_heads] + k.shape[1:]), )
            attn_weights = attn_weights.reshape([
                -1,
            ] + attn_weights.shape[-2:])
        else:
            attn_weights = paddle.bmm(q, k.transpose([0, 2, 1]))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len,
                                              bsz)

        assert list(
            attn_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.tile([attn_weights.shape[0], 1, 1])
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.reshape(
                [bsz, self.num_heads, tgt_len, src_len])
            if not is_tpu:
                attn_weights = attn_weights.reshape(
                    [kv_bsz, -1, self.num_heads, tgt_len, src_len])
                attn_weights = paddle.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    .astype('bool'),
                    float('-inf') * paddle.ones_like(attn_weights),
                    attn_weights)
            else:
                attn_weights = attn_weights.transpose([2, 1, 0])
                attn_weights = paddle.where(key_padding_mask,
                                            float('-inf') *
                                            paddle.ones_like(attn_weights),
                                            attn_weights)
                attn_weights = attn_weights.transpose([2, 1, 0])
            attn_weights = attn_weights.reshape(
                [bsz * self.num_heads, tgt_len, src_len])

        if before_softmax:
            return attn_weights, v

        def softmax_supporting_onnx_trace(x, dim: int, onnx_trace: bool=False):
            if onnx_trace:
                return F.softmax(x, axis=dim)
            else:
                return F.softmax(x, axis=dim, dtype='float32')

        attn_weights_float = softmax_supporting_onnx_trace(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = paddle.cast(attn_weights_float, attn_weights.dtype)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn = paddle.einsum(
                "bxhts,bhsd->bxhtd",
                attn_probs.reshape([kv_bsz, -1, self.num_heads] +
                                   attn_probs.shape[1:]),
                v.reshape([kv_bsz, self.num_heads] + v.shape[1:]), )
            attn = attn.reshape([
                -1,
            ] + attn.shape[-2:])
        else:
            attn = paddle.bmm(attn_probs, v)
        assert list(
            attn.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.shape[1] == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.reshape([tgt_len, bsz, self.embed_dim])
        else:
            attn = attn.transpose([1, 0, 2]).reshape(
                [tgt_len, bsz, self.embed_dim])
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.reshape(
                [bsz, self.num_heads, tgt_len, src_len]).transpose([1, 0, 2, 3])
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(axis=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool, ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = paddle.concat(
                [
                    paddle.cast(prev_key_padding_mask, 'float32'),
                    paddle.cast(key_padding_mask, 'float32')
                ],
                axis=1)
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.shape[1]:
                filler = paddle.zeros(
                    [batch_size, src_len - prev_key_padding_mask.shape[1]], )
                new_key_padding_mask = paddle.concat(
                    [
                        paddle.cast(prev_key_padding_mask, 'float32'),
                        paddle.cast(filler, 'float32')
                    ],
                    axis=1)
            else:
                new_key_padding_mask = prev_key_padding_mask
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.shape[1]:
                filler = paddle.zeros(
                    [batch_size, src_len - key_padding_mask.shape[1]], )
                new_key_padding_mask = paddle.concat(
                    [
                        paddle.cast(filler, 'float32'),
                        paddle.cast(key_padding_mask, 'float32')
                    ],
                    axis=1)
            else:
                new_key_padding_mask = paddle.cast(key_padding_mask, 'float32')
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @paddle.jit.to_static
    def reorder_incremental_state(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            new_order: Tensor, ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention:
                        if input_buffer_k.shape[
                                0] * self.beam_size == new_order.shape[0]:
                            return incremental_state
                        elif self.beam_size > 1:
                            input_buffer[k] = paddle.index_select(
                                input_buffer_k,
                                index=new_order.reshape(
                                    [-1, self.beam_size])[:, 0] //
                                self.beam_size,
                                axis=0, )
                        else:
                            input_buffer[k] = paddle.index_select(
                                input_buffer_k, index=new_order, axis=0)
                    else:
                        input_buffer[k] = paddle.index_select(
                            input_buffer_k, index=new_order, axis=0)
            incremental_state = self._set_input_buffer(incremental_state,
                                                       input_buffer)
        return incremental_state

    def set_beam_size(self, beam_size):
        """Used for effiecient beamable enc-dec attention"""
        self.beam_size = beam_size

    def _get_input_buffer(
            self,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]], ):
        return self.set_incremental_state(incremental_state, "attn_state",
                                          buffer)

    def apply_sparse_mask(self,
                          attn_weights,
                          tgt_len: int,
                          src_len: int,
                          bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix +
                             "k_proj.weight"] = state_dict[k][dim:2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix +
                                 "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim:2 * dim]
                    items_to_add[prefix +
                                 "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


class GumbelVectorQuantizer(nn.Layer):
    def __init__(
            self,
            dim,
            num_vars,
            temp,
            groups,
            combine_groups,
            vq_dim,
            time_first,
            activation=nn.GELU(),
            weight_proj_depth=1,
            weight_proj_factor=1, ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = self.create_parameter(
            (1, num_groups * num_vars, var_dim),
            default_initializer=nn.initializer.Uniform())

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                Linear(inner_dim, groups * num_vars), )
        else:
            self.weight_proj = Linear(
                self.input_dim,
                groups * num_vars,
                weight_attr=nn.initializer.Normal(mean=0, std=1),
                bias_attr=nn.initializer.Zero())

        if isinstance(temp, str):
            import ast

            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(self.max_temp * self.temp_decay**num_updates,
                             self.min_temp)

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = paddle.to_tensor(
                inds, dtype='int64', place=self.vars.place).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.reshape(
                    self.num_vars**self.groups, -1)
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (self.vars.squeeze(0).index_select(0, indices)
                .reshape(self.num_vars**self.groups, -1))

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.reshape(-1, self.groups)
        cb_size = indices.shape[0]
        assert (n < cb_size
                ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = paddle.randint(low=0, high=cb_size, shape=(b * n, ))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).reshape(
            b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = paddle.full(indices.shape[:-1], 0, dtype=indices.dtype)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_vars**exponent)
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):
        result = {"num_vars": self.num_vars * self.groups}

        if not self.time_first:
            x = x.transpose([0, 2, 1])

        bsz, tsz, fsz = x.shape
        x = x.reshape([-1, fsz])
        x = self.weight_proj(x)
        x = x.reshape([bsz * tsz * self.groups, -1])

        _, k = x.max(-1)
        hard_x = paddle.zeros_like(x)
        hard_x.scatter_(-1, k.reshape([-1, 1]), 1.0)
        hard_x = hard_x.reshape([bsz * tsz, self.groups, -1])
        hard_probs = paddle.mean(hard_x.astype('float32'), axis=0)
        result["code_perplexity"] = paddle.exp(-paddle.sum(
            hard_probs * paddle.log(hard_probs + 1e-7), axis=-1)).sum()

        avg_probs = F.softmax(
            x.reshape([bsz * tsz, self.groups, -1]).astype('float32'),
            axis=-1).mean(axis=0)
        result["prob_perplexity"] = paddle.exp(-paddle.sum(
            avg_probs * paddle.log(avg_probs + 1e-7), axis=-1)).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(
                x.astype('float32'), temperature=self.curr_temp,
                hard=True).astype(x.dtype)
        else:
            x = hard_x

        x = x.reshape([bsz * tsz, -1])

        vars = self.vars
        if self.combine_groups:
            vars = vars.tile([1, self.groups, 1])

        if produce_targets:
            result["targets"] = (x.reshape([bsz * tsz * self.groups, -1])
                                 .argmax(axis=-1)
                                 .reshape([bsz, tsz, self.groups]).detach())

        x = x.unsqueeze(-1) * vars
        x = x.reshape([bsz * tsz, self.groups, self.num_vars, -1])
        x = x.sum(axis=-2)
        x = x.reshape([bsz, tsz, -1])

        if not self.time_first:
            x = x.transpose([0, 2, 1])

        result["x"] = x

        return result


class GradMultiply(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.numpy().copy()
        return paddle.to_tensor(res, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SamePad(nn.Layer):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x


class TransposeLast(nn.Layer):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        trans_dim = np.arange(x.dim())
        trans_dim[-1], trans_dim[-2] = trans_dim[-2], trans_dim[-1]
        return x.transpose(trans_dim)


class Fp32LayerNorm(LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.astype('float32'),
            self._normalized_shape,
            self.weight.astype('float32') if self.weight is not None else None,
            self.bias.astype('float32') if self.bias is not None else None,
            self._epsilon, )
        return output.astype(input.dtype)


# Todo: change this when paddle supports F.group_norm
class Fp32GroupNorm(nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.group_norm = paddle.nn.GroupNorm(*args, **kwargs)
        fp32_weight = paddle.create_parameter(
            shape=self.group_norm.weight.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(
                self.group_norm.weight))
        fp32_bias = paddle.create_parameter(
            shape=self.group_norm.bias.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(
                self.group_norm.bias))
        self.group_norm.weight = fp32_weight
        self.group_norm.bias = fp32_bias

    def forward(self, input):
        output = self.group_norm(input.astype('float32'))
        return output.astype(input.dtype)


class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


def relu_squared(x: paddle.Tensor):
    return F.relu(x).pow(2)


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""

    def gelu_accurate(x):
        if not hasattr(gelu_accurate, "_a"):
            gelu_accurate._a = math.sqrt(2 / math.pi)
        return (0.5 * x * (1 + paddle.tanh(gelu_accurate._a *
                                           (x + 0.044715 * paddle.pow(x, 3)))))

    def gelu(x: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.gelu(x.astype('float32')).astype(x.dtype)

    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        return relu_squared
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return paddle.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return paddle.nn.Swish
    else:
        raise RuntimeError(
            "--activation-fn {} not supported".format(activation))


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "gelu_fast",  # deprecated
        "gelu_accurate",
        "tanh",
        "linear",
    ]


def compute_mask_indices(
        shape: Tuple[int, int],
        padding_mask: Optional[paddle.Tensor],
        mask_prob: float,
        mask_length: int,
        mask_type: str="static",
        mask_other: float=0.0,
        min_masks: int=0,
        no_overlap: bool=False,
        min_space: int=0,
        require_same_masks: bool=True,
        mask_dropout: float=0.0, ) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length) + np.random.rand())

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(
                mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0
                     for s, e in parts),
                    np.int_, )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray([
                mask_idc[j] + offset
                for j in range(len(mask_idc)) for offset in range(lengths[j])
            ])

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(
                mask_idc, len(mask_idc) - num_holes, replace=False)

        mask[i, mask_idc] = True

    return mask


def index_put(tensor, indices, value):
    tensor[indices] = value
    return tensor


# ToDo if faster?
def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = paddle.empty([max], dtype='int64')
    if max > buffered_arange.buf.numel():
        buffered_arange.buf = paddle.arange(max)
    return buffered_arange.buf[:max]


def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.shape[dim]
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0, ) * (-1 - dim) * 2
    return F.pad(
        x,
        pad=[*pad_offset, 0, remainder, *pad_offset],
        value=value,
        data_format='NLC'), remainder


EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"])
LAYER_TYPE_CHOICES = ChoiceEnum(["transformer"])  # ToDo: conformer 


@dataclass
class Wav2Vec2Config:
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help":
            "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        }, )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"})
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"})
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"})
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"})
    activation_fn: ChoiceEnum(get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"})
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"})
    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"})
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"})
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"})
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"})
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"}, )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help":
            "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        }, )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"})
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help":
            "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        }, )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"})
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"})
    quantize_targets: bool = field(
        default=False, metadata={"help": "use quantized targets"})
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"})
    same_quantizer: bool = field(
        default=False,
        metadata={"help": "use same quantizer for inputs and targets"})
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"})
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"})
    quantizer_depth: int = field(
        default=1,
        metadata={"help": "number of quantizer layers"}, )
    quantizer_factor: int = field(
        default=3,
        metadata={
            "help":
            "dimensionality increase for inner quantizer layers (if depth > 1)"
        }, )
    latent_vars: int = field(
        default=320,
        metadata={
            "help": "number of latent variables V in each group of the codebook"
        }, )
    latent_groups: int = field(
        default=2,
        metadata={
            "help": "number of groups G of latent variables in the codebook"
        }, )
    latent_dim: int = field(
        default=0,
        metadata={
            "help":
            "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        }, )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"})
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"})
    mask_other: float = field(
        default=0,
        metadata={
            "help":
            "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        }, )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"})
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help":
            "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        }, )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"}, )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"})
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"})
    mask_channel_before: bool = False
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"}, )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help":
            "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        }, )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"})
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"}, )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={
            "help": "sample negatives from everywhere, not just masked states"
        }, )
    cross_sample_negatives: int = field(
        default=0,
        metadata={"help": "number of negative examples from the any sample"})
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"})

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={
            "help": "number of filters for convolutional positional embeddings"
        }, )
    conv_pos_groups: int = field(
        default=16,
        metadata={
            "help": "number of groups for convolutional positional embedding"
        }, )
    pos_conv_depth: int = field(
        default=1,
        metadata={"help": "depth of positional encoder network"}, )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help":
            "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        }, )
    max_positions: int = field(
        default=100000, metadata={"help": "Max positions"})
    checkpoint_activations: bool = field(
        default=False,
        metadata={
            "help": "recompute activations and save memory for extra compute"
        }, )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help":
            "pad the input to encoder such that the sequence length is divisible by multiple"
        }, )
    crop_seq_to_multiple: int = field(
        default=1,
        metadata={
            "help":
            "crop convolutional feature extractor output such that the sequence length is divisible by multiple"
        }, )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help":
            "depthwise-conv-kernel-size for convolution in conformer layer"
        }, )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"}, )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"}, )
    fp16: bool = field(
        default=False, metadata={"help": "If fp16 is being used"})


class Wav2Vec2Model(nn.Layer):
    def __init__(self, cfg: Wav2Vec2Config):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias, )

        self.post_extract_proj = (Linear(self.embed, cfg.encoder_embed_dim)
                                  if self.embed != cfg.encoder_embed_dim and
                                  not cfg.quantize_input else None)

        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor, )
            self.project_q = Linear(vq_dim, final_dim)
        else:
            self.project_q = Linear(self.embed, final_dim)

        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.latent_vars,
                    temp=cfg.latent_temp,
                    groups=cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                    weight_proj_depth=cfg.quantizer_depth,
                    weight_proj_factor=cfg.quantizer_factor, )
            self.project_inp = Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = self.create_parameter(
            shape=[cfg.encoder_embed_dim],
            default_initializer=paddle.nn.initializer.Uniform(),
            dtype='float32', )

        encoder_cls = TransformerEncoder

        self.encoder = encoder_cls(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                Linear(final_dim, final_dim * 2), GLU())

        self.final_proj = Linear(cfg.encoder_embed_dim, final_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Config, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def apply_mask(
            self,
            x,
            padding_mask,
            mask_indices=None,
            mask_channel_indices=None, ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space, )
            mask_channel_indices = (
                paddle.to_tensor(mask_channel_indices, plcae=x.plcae)
                .unsqueeze(1).expand([-1, T, -1]))
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout, )
                mask_indices = paddle.to_tensor(mask_indices, place=x.place)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space, )
                mask_channel_indices = (
                    paddle.to_tensor(mask_channel_indices, place=x.place)
                    .unsqueeze(1).expand([-1, T, -1]))
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def sample_negatives(self, y, num, padding_count=None):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return paddle.empty([0], dtype=y.dtype)

        bsz, tsz, fsz = y.shape
        y = y.reshape([-1, fsz])  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with paddle.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (buffered_arange(num).unsqueeze(-1)
                        .expand([-1, self.n_negatives]).flatten())

                neg_idxs = paddle.randint(
                    low=0, high=high - 1, shape=[bsz, self.n_negatives * num])
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (buffered_arange(num).unsqueeze(-1)
                        .expand([-1, self.cross_sample_negatives]).flatten())

                cross_neg_idxs = paddle.randint(
                    low=0,
                    high=cross_high - 1,
                    shape=[bsz, self.cross_sample_negatives * num], )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            neg_idxs = neg_idxs + (paddle.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = paddle.concat([neg_idxs, cross_neg_idxs], axis=1)

        negs = y[neg_idxs.reshape([-1])]
        negs = negs.reshape(
            [bsz, num, self.n_negatives + self.cross_sample_negatives,
             fsz]).transpose([2, 0, 1, 3])  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = paddle.concat([y, negatives], axis=0)

        logits = paddle.nn.functional.cosine_similarity(x, targets, axis=-1)
        logits = logits / self.logit_temp
        logits = logits.astype(x.dtype)

        return logits

    def _get_feat_extract_output_lengths(self, input_lengths: paddle.Tensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return paddle.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(input_lengths, conv_cfg_list[i][1],
                                             conv_cfg_list[i][2])

        return paddle.cast(input_lengths, 'int64')

    def forward(
            self,
            source,
            padding_mask=None,
            mask=True,
            features_only=False,
            layer=None,
            mask_indices=None,
            mask_channel_indices=None,
            padding_count=None, ):

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with paddle.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.pow(2).mean()

        features = features.transpose([0, 2, 1])
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - paddle.cast(padding_mask, 'int64')).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(
                input_lengths)

            padding_mask = paddle.zeros(
                features.shape[:2], dtype=features.dtype)

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[(paddle.arange(padding_mask.shape[0]),
                          output_lengths - 1, )] = 1
            padding_mask = paddle.cast(
                (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])), 'bool')
        else:
            padding_mask = None

        time_steps_to_drop = features.shape[1] % self.crop_seq_to_multiple
        if time_steps_to_drop != 0:
            features = features[:, :-time_steps_to_drop]
            unmasked_features = unmasked_features[:, :-time_steps_to_drop]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-time_steps_to_drop]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices, )
            if mask_indices is not None:
                y = unmasked_features[mask_indices].reshape([
                    unmasked_features.shape[0], -1, unmasked_features.shape[-1]
                ])
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x, layer_results = self.encoder(
            x, padding_mask=padding_mask, layer=layer)

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results,
            }

        if self.quantizer:
            if self.negatives_from_everywhere:
                q = self.quantizer(unmasked_features, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                y = self.project_q(y)

                negs, _ = self.sample_negatives(
                    y,
                    mask_indices[0].sum(),
                    padding_count=padding_count, )
                y = y[mask_indices].reshape([y.shape[0], -1, y.shape[-1]])

            else:
                q = self.quantizer(y, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]

                y = self.project_q(y)

                negs, _ = self.sample_negatives(
                    y,
                    y.shape[1],
                    padding_count=padding_count, )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.shape[0] * y.shape[1], self.codebook_negatives)
                cb_negs = cb_negs.reshape(
                    [self.codebook_negatives, y.shape[0], y.shape[1],
                     -1])  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = paddle.concat([negs, cb_negs], axis=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.shape[1],
                    padding_count=padding_count, )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.shape[1],
                    padding_count=padding_count, )

        x = x[mask_indices].reshape([x.shape[0], -1, x.shape[-1]])

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose([0, 2, 1])
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False, layer=None):
        res = self.forward(
            source, padding_mask, mask=mask, features_only=True, layer=layer)
        return res

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose([2, 1, 0])
        logits = logits.reshape([-1, logits.shape[-1]])
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return paddle.zeros(x.shape[1] * x.shape[2], dtype='int64')

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append((net_output["num_vars"] - net_output["prob_perplexity"])
                       / net_output["num_vars"])

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self, last_layer=None):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

        if last_layer is not None:
            self.encoder.layers = nn.LayerList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer)


class ConvFeatureExtractionModel(nn.Layer):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float=0.0,
            mode: str="default",
            conv_bias: bool=False, ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False, ):
            def make_conv():
                conv = Conv1D(
                    n_in,
                    n_out,
                    k,
                    stride=stride,
                    bias_attr=conv_bias
                    if not conv_bias else paddle.ParamAttr())
                # nn.initializer.KaimingNormal()(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm
                    ) is False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim),
                        TransposeLast(), ),
                    nn.GELU(), )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim),
                    nn.GELU(), )
            else:
                return nn.Sequential(
                    make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.LayerList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias, ))
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)

        return x


def make_conv_pos(e, k, g):
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    pos_conv = Conv1D(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
        weight_attr=nn.initializer.Normal(mean=0, std=std),
        bias_attr=nn.initializer.Constant(0))
    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())

    return pos_conv


class TransformerEncoder(nn.Layer):
    def build_encoder_layer(self, args: Wav2Vec2Config):
        layer = TransformerSentenceEncoderLayer(
            embedding_dim=self.embedding_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=self.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            activation_fn=args.activation_fn,
            layer_norm_first=args.layer_norm_first, )
        return layer

    def __init__(self, args: Wav2Vec2Config):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_conv_block(e, k, g, l):
                return nn.Sequential(*[
                    nn.Sequential(
                        Conv1D(
                            e,
                            e,
                            kernel_size=k,
                            padding=k // 2,
                            groups=g, ),
                        SamePad(k),
                        TransposeLast(),
                        LayerNorm(e, elementwise_affine=False),
                        TransposeLast(),
                        nn.GELU(), ) for _ in range(l)
                ])

            self.pos_conv = make_conv_block(self.embedding_dim, k,
                                            args.conv_pos_groups, num_layers)

        else:
            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args.conv_pos,
                args.conv_pos_groups, )

        self.layers = nn.LayerList([
            self.build_encoder_layer(args) for _ in range(args.encoder_layers)
        ])
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
            self,
            x,
            padding_mask=None,
            tgt_layer=None,
            min_layer=0, ):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose([0, 2, 1]))
        x_conv = x_conv.transpose([0, 2, 1])
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0)
        if pad_length > 0 and padding_mask is None:
            padding_mask = paddle.zeros([x.shape[0], x.shape[1]], dtype='bool')
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask,
                self.required_seq_len_multiple,
                dim=-1,
                value=True)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose([1, 0, 2])

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False)
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose([1, 0, 2])

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(a, b, c):
                return (a[:-pad_length], b[:-pad_length]
                        if b is not None else b, c[:-pad_length], )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerSentenceEncoderLayer(nn.Layer):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float=768,
            ffn_embedding_dim: float=3072,
            num_attention_heads: int=8,
            dropout: float=0.1,
            attention_dropout: float=0.1,
            activation_dropout: float=0.1,
            activation_fn: str="relu",
            layer_norm_first: bool=False, ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True, )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
            self,
            x: paddle.Tensor,
            self_attn_mask: paddle.Tensor=None,
            self_attn_padding_mask: paddle.Tensor=None,
            need_weights: bool=False,
            att_args=None, ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False, )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False, )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)


@dataclass
class AudioPretrainingConfig:
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help":
            "target sample rate. audio files will be up/down sampled to this rate"
        }, )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        }, )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"})
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"})
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to skip small examples"})
