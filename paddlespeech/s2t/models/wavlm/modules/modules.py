# --------------------------------------------------------
# paddle: Large-Scale Self-Supervised  Pre-training  for Full Stack Speech Processing (https://arxiv.org/abs/2110.13900.pdf)
# Github source: https://github.com/microsoft/unilm/tree/master/paddle
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------
import math
import warnings
from typing import Dict
from typing import Optional
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

from .functional import multi_head_attention_forward_paddle


class TransposeLast(nn.Layer):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return paddle.transpose(x, perm=[0, 2, 1])


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps, )
        return output.type_as(input)


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps, )
        return output.type_as(input)


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


class Swish(nn.Layer):
    """Swish function
    """

    def __init__(self):
        """Construct an MultiHeadedAttention object."""
        super(Swish, self).__init__()
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)


class GLU_Linear(nn.Layer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 glu_type="sigmoid",
                 bias_in_glu=True):
        super(GLU_Linear, self).__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = nn.GELU()

        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        # to be consistent with GLU_Linear, we assume the input always has the #channel (#dim) in the last dimension of the tensor, so need to switch the dimension first for 1D-Conv case
        x = self.linear(x)

        if self.glu_type == "bilinear":
            x = (x[:, :, 0:self.output_dim] *
                 x[:, :, self.output_dim:self.output_dim * 2])
        else:
            x = (x[:, :, 0:self.output_dim] *
                 self.glu_act(x[:, :, self.output_dim:self.output_dim * 2]))

        return x


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (0.5 * x * (1 + paddle.tanh(gelu_accurate._a *
                                       (x + 0.044715 * paddle.pow(x, 3)))))


def gelu(x: Tensor) -> Tensor:
    return nn.functional.gelu(x.astype("float32")).astype(x.dtype)


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        warnings.warn(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate")
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return paddle.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "glu":
        return lambda x: x
    else:
        raise RuntimeError(
            "--activation-fn {} not supported".format(activation))


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
        - Module weights must have the right sizes wrt the block size
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
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) %
            block_size == 0), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (module.in_channels % block_size == 0
                    ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = paddle.zeros(
                    in_features // block_size * out_features,
                    device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).reshape(
                    [-1, in_features])

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = paddle.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device, )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).reshape(
                        [-1, in_channels])
                else:
                    mask = paddle.zeros(
                        weight.size(0), weight.size(1), device=weight.device)

                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2).unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1]))

            # scale weights and apply mask
            mask = mask.to(paddle.bool)
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


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
            has_relative_attention_bias=True,
            num_buckets=32,
            max_distance=128,
            gru_rel_pos=True,
            rescale_init=False, ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert (self.head_dim * num_heads == self.embed_dim
                ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and "
            "value to be of the same size")

        k_bias = True
        if rescale_init:
            k_bias = False

        k_embed_dim = embed_dim
        q_embed_dim = embed_dim

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, k_embed_dim, bias_attr=k_bias), q_noise,
            qn_block_size)
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias_attr=bias), q_noise,
            qn_block_size)
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, q_embed_dim, bias_attr=bias), q_noise,
            qn_block_size)

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias_attr=bias), q_noise,
            qn_block_size)

        if add_bias_kv:
            self.bias_k = self.create_parameter(
                shape=[1, 1, embed_dim], dtype="float32")
            self.bias_v = self.create_parameter(
                shape=[1, 1, embed_dim], dtype="float32")

        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = self.create_parameter(
                shape=[1, num_heads, 1, 1], dtype="float32")

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def _relative_positions_bucket(self, relative_positions,
                                   bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (
                relative_positions > 0).astype("int64") * num_buckets
            relative_positions = paddle.abs(relative_positions)
        else:
            relative_positions = -paddle.minimum(
                relative_positions, paddle.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            paddle.log(relative_positions.astype("float32") /
                       max_exact) / math.log(max_distance / max_exact) *
            (num_buckets - max_exact)).astype("int64")
        relative_postion_if_large = paddle.minimum(
            relative_postion_if_large,
            paddle.full_like(relative_postion_if_large, num_buckets - 1))

        relative_buckets += paddle.where(is_small, relative_positions,
                                         relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = paddle.arange(query_length, dtype="int64")[:, None]
        memory_position = paddle.arange(key_length, dtype="int64")[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position, bidirectional=True)
        # relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.transpose([2, 0, 1])
        return values

    def forward(self,
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
                need_head_weights: bool=False,
                position_bias: Optional[Tensor]=None
                ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
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

        tgt_len, bsz, embed_dim = query.shape
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.shape

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias_ = position_bias.unsqueeze(0)
            position_bias = paddle.concat(
                [position_bias_ for _ in range(bsz)], axis=0)
            position_bias = position_bias.reshape(
                [bsz * self.num_heads, tgt_len, src_len])
        if (incremental_state is None and not static_kv and
                self.q_head_dim == self.head_dim):
            assert key is not None and value is not None
            assert attn_mask is None

            attn_mask_rel_pos = None
            if position_bias is not None:
                attn_mask_rel_pos = position_bias
                if self.gru_rel_pos:
                    query_layer = query.transpose([1, 0, 2])
                    new_x_shape = query_layer.shape[:-1] + [self.num_heads, -1]
                    query_layer = query_layer.reshape(new_x_shape)
                    query_layer = query_layer.transpose([0, 2, 1, 3])
                    _B, _H, _L, __ = query_layer.shape

                    gate_a, gate_b = paddle.nn.functional.sigmoid(
                        self.grep_linear(query_layer).reshape(
                            [_B, _H, _L, 2, 4]).sum(-1, keepdim=False)).chunk(
                                2, axis=-1)

                    gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                    attn_mask_rel_pos = gate_a_1.reshape(
                        [bsz * self.num_heads, -1, 1]) * position_bias

                attn_mask_rel_pos = attn_mask_rel_pos.reshape(
                    (-1, tgt_len, tgt_len))
            k_proj_bias = self.k_proj.bias
            if k_proj_bias is None:
                k_proj_bias = paddle.zeros_like(self.q_proj.bias)

            x, attn = multi_head_attention_forward_paddle(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                paddle.empty([0]),
                paddle.concat(
                    (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias),
                    axis=0),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask_rel_pos,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight, )

            return x, attn, position_bias

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
            k = paddle.concat([k, self.bias_k.repeat(1, bsz, 1)], axis=0)
            v = paddle.concat([v, self.bias_v.repeat(1, bsz, 1)], axis=0)
            if attn_mask is not None:
                attn_mask = paddle.concat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)],
                    axis=1)

            if key_padding_mask is not None:
                key_padding_mask = paddle.concat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    axis=1, )

        q = (q.contiguous()
             .reshape([tgt_len, bsz * self.num_heads, self.q_head_dim])
             .transpose([1, 0, 2]))
        if k is not None:
            k = (k.contiguous()
                 .reshape([-1, bsz * self.num_heads, self.k_head_dim])
                 .transpose([1, 0, 2]))
        if v is not None:
            v = (v.contiguous()
                 .reshape([-1, bsz * self.num_heads, self.head_dim])
                 .transpose([1, 0, 2]))

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.reshape(
                    [bsz * self.num_heads, -1, self.head_dim])
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = paddle.concat([prev_key, k], axis=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.reshape(
                    [bsz * self.num_heads, -1, self.head_dim])
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
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv, )

            saved_state["prev_key"] = k.reshape(
                [bsz, self.num_heads, -1, self.head_dim])
            saved_state["prev_value"] = v.reshape(
                [bsz, self.num_heads, -1, self.head_dim])
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state,
                                                       saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = paddle.concat(
                [k, k.new_zeros((k.size(0), 1) + k.shape[2:])], axis=1)
            v = paddle.concat(
                [v, v.new_zeros((v.size(0), 1) + v.shape[2:])], axis=1)
            if attn_mask is not None:
                attn_mask = paddle.concat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)],
                    axis=1)

            if key_padding_mask is not None:
                key_padding_mask = paddle.concat(
                    [
                        key_padding_mask,
                        paddle.zeros(key_padding_mask.size(0),
                                     1).type_as(key_padding_mask),
                    ],
                    axis=1, )

        attn_weights = paddle.matmul(q, k.transpose([0, 2, 1]))

        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len,
                                              bsz)

        assert list(
            attn_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.reshape(
                [bsz, self.num_heads, tgt_len, src_len])
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(paddle.bool),
                float("-inf"), )
            attn_weights = attn_weights.reshape(
                [bsz * self.num_heads, tgt_len, src_len])

        if before_softmax:
            return attn_weights, v, position_bias

        if position_bias is not None:
            if self.gru_rel_pos == 1:
                query_layer = q.reshape(
                    [bsz, self.num_heads, tgt_len, self.q_head_dim])
                _B, _H, _L, __ = query_layer.shape
                gate_a, gate_b = paddle.sigmoid(
                    self.grep_linear(query_layer).reshape([_B, _H, _L, 2, 4])
                    .sum(-1, keepdim=False)).chunk(
                        2, axis=-1)

                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                position_bias = gate_a_1.reshape(
                    [bsz * self.num_heads, -1, 1]) * position_bias

            position_bias = position_bias.reshape(attn_weights.shape)

            attn_weights = attn_weights + position_bias

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = paddle.bmm(attn_probs, v)
        assert list(
            attn.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose([1, 0, 2]).reshape([tgt_len, bsz, embed_dim])
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.reshape(
                [bsz, self.num_heads, tgt_len, src_len]).transpose([1, 0, 2, 3])
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, position_bias

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
                [prev_key_padding_mask.float(), key_padding_mask.float()],
                axis=1)
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = paddle.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device, )
                new_key_padding_mask = paddle.concat(
                    [prev_key_padding_mask.float(), filler.float()], axis=1)

            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = paddle.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device, )
                new_key_padding_mask = paddle.concat(
                    [filler.float(), key_padding_mask.float()], axis=1)

            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

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
