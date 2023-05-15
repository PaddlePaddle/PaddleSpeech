import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Optional, List, Tuple
import math

def _mha_shape_check(query: paddle.Tensor, key: paddle.Tensor, value: paddle.Tensor,
                     key_padding_mask: Optional[paddle.Tensor], attn_mask: Optional[paddle.Tensor], num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value)
    

def scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal):
    """
    Scaled Dot-Product Attention
    """
    
    d_key = k.shape[-1]
    scaled_q = paddle.scale(x=q, scale=d_key ** -0.5)
    product = paddle.matmul(x=scaled_q, y=k, transpose_y=True)
    weights = paddle.nn.functional.softmax(x=product + attn_mask)
    if dropout_p:
        weights = paddle.fluid.layers.nn.dropout(
            weights,
            dropout_prob=dropout_p,
            dropout_implementation="upscale_in_train",
            is_test=False)
    out = paddle.matmul(x=weights, y=v)
    return out

    
def addr(input, vec1, vec2, beta=1, alpha=1, out=None):
    row = vec1.shape[0]
    column = vec2.shape[0]
    vec1 = paddle.unsqueeze(vec1, 0)
    vec1 = paddle.transpose(vec1, [1, 0])
    vec1 = paddle.expand(vec1, [row, column])
    new_vec2 = paddle.zeros([column, column], dtype=vec2.dtype)
    new_vec2[0, :] = vec2
    out = alpha * paddle.matmul(vec1, new_vec2)
    out = beta * input + out
    return out

def multi_head_attention_forward(
    x: paddle.Tensor,
    num_heads: int,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    c_proj: nn.Linear,
    attn_mask: Optional[paddle.Tensor] = None,
):
    max_len, batch_size, emb_dim = x.shape
    head_dim = emb_dim // num_heads
    scaling = float(head_dim) ** -0.5
    q = q_proj(x)  # L, N, E
    k = k_proj(x)  # L, N, E
    v = v_proj(x)  # L, N, E

    v = v.reshape((-1, batch_size * num_heads, head_dim)).transpose((1, 0, 2))
    k = k.reshape((-1, batch_size * num_heads, head_dim)).transpose((1, 0, 2))
    q = q.reshape((-1, batch_size * num_heads, head_dim)).transpose((1, 0, 2))

    q = q * scaling
    qk = paddle.matmul(q, k, transpose_y=True)
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask.unsqueeze_(0)
        assert attn_mask.shape[0] == 1 and attn_mask.shape[1] == max_len and attn_mask.shape[2] == max_len
        qk += attn_mask

    qk = F.softmax(qk, axis=-1)
    atten = paddle.bmm(qk, v)
    atten = atten.transpose((1, 0, 2))
    atten = atten.reshape((max_len, batch_size, emb_dim))
    atten = c_proj(atten)
    return atten

def linear(input, weight, bias=None):
    # compute y = x A^T + b
    # Input: (N, in_feature) paddle tensor
    # weight: (out_feature, in_feature) paddle tensor
    # bias: (out_feature) paddle tensor
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        return paddle.addmm(bias, input, weight)
    output = paddle.matmul(input, weight)
    if bias is not None:
        output += bias
    return output


def _in_projection_packed(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    w: paddle.Tensor,
    b: Optional[paddle.Tensor] = None,
) -> List[paddle.Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    # E = q.size(-1)
    E = q.shape[-1]
    if k is v:
        if q is k:
            # self-attention
            proj = linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose([2, 1, 0]).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose([2, 1, 0]).squeeze(-2).contiguous()
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
    
def _in_projection(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    w_q: paddle.Tensor,
    w_k: paddle.Tensor,
    w_v: paddle.Tensor,
    b_q: Optional[paddle.Tensor] = None,
    b_k: Optional[paddle.Tensor] = None,
    b_v: Optional[paddle.Tensor] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    A, B, C = linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
    
    return A, B, C
    # return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
    
def multi_head_attention_forward_paddle(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[paddle.Tensor],
    in_proj_bias: Optional[paddle.Tensor],
    bias_k: Optional[paddle.Tensor],
    bias_v: Optional[paddle.Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: paddle.Tensor,
    out_proj_bias: Optional[paddle.Tensor],
    training: bool = True,
    key_padding_mask: Optional[paddle.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[paddle.Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[paddle.Tensor] = None,
    k_proj_weight: Optional[paddle.Tensor] = None,
    v_proj_weight: Optional[paddle.Tensor] = None,
    static_k: Optional[paddle.Tensor] = None,
    static_v: Optional[paddle.Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True
    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    
    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    # if not is_batched:
    #     # unsqueeze if the input is unbatched
    #     query = query.unsqueeze(1)
    #     key = key.unsqueeze(1)
    #     value = value.unsqueeze(1)
    #     if key_padding_mask is not None:
    #         key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    # import pdb; pdb.set_trace()
    tgt_len, bsz, embed_dim = query.shape
    # tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    if is_causal:
        attn_mask = None

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, paddle.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
    
    # prep attention mask

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if tuple(attn_mask.shape) != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        # k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        k = paddle.concat([k, bias_k.repeat(1, bsz, 1)], axis=1)
        # v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        v = paddle.concat([v, bias_v.repeat(1, bsz, 1)], axis=1)
        if attn_mask is not None:
            # attn_mask = pad(attn_mask, (0, 1))
            # pad last dim with 0 on one side and 1 on the other
            attn_mask = paddle.concat([attn_mask, paddle.zeros_like(attn_mask[:, :, -1:])], axis=2)
        if key_padding_mask is not None:
            # key_padding_mask = pad(key_padding_mask, (0, 1))
            # pad last dim with 0 on one side and 1 on the other
            key_padding_mask = paddle.concat([key_padding_mask, paddle.zeros_like(key_padding_mask[:, -1:])], axis=1)
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    # q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    q = q.reshape([tgt_len, bsz * num_heads, head_dim]).transpose([1, 0, 2])

    
    if static_k is None:
        # k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        k = k.reshape([k.shape[0], bsz * num_heads, head_dim]).transpose([1, 0, 2])
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        # v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.reshape([v.shape[0], bsz * num_heads, head_dim]).transpose([1, 0, 2])
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        # k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        k = paddle.concat([k, paddle.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], axis=1)
        # v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        v = paddle.concat([v, paddle.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], axis=1)
        if attn_mask is not None:
            # attn_mask = pad(attn_mask, (0, 1))
            attn_mask = paddle.concat([attn_mask, paddle.zeros_like(attn_mask[:, :, -1:])], axis=2)
        if key_padding_mask is not None:
            # key_padding_mask = pad(key_padding_mask, (0, 1))
            key_padding_mask = paddle.concat([key_padding_mask, paddle.zeros_like(key_padding_mask[:, -1:])], axis=1)

    # update source sequence length after adjustments
    src_len = k.shape[1]

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        # key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        key_padding_mask = key_padding_mask.reshape([bsz, 1, 1, src_len]).expand([-1, num_heads, -1, -1]).reshape([bsz * num_heads, 1, src_len])
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        if attn_mask is not None:
            # attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            attn_output_weights = addr(q_scaled, k.transpose(-2, -1))
        else:
            # attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = paddle.bmm(q_scaled, k.transpose(0, 2, 1))
        # attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = paddle.nn.functional.softmax(attn_output_weights, axis=-1)
        if dropout_p > 0.0:
            # attn_output_weights = dropout(attn_output_weights, p=dropout_p)
            attn_output_weights = paddle.nn.functional.dropout(attn_output_weights, p=dropout_p)

        # attn_output = torch.bmm(attn_output_weights, v)
        attn_output = paddle.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose([1, 0, 2]).reshape([tgt_len * bsz, embed_dim])
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.reshape([tgt_len, bsz, attn_output.shape[1]])

        # optionally average attention weights over heads
        # attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.reshape([bsz, num_heads, tgt_len, src_len])
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.shape[0] == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                # attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)
                attn_mask = attn_mask.reshape([bsz, num_heads, -1, src_len])

        q = q.reshape([bsz, num_heads, tgt_len, head_dim])
        k = k.reshape([bsz, num_heads, src_len, head_dim])
        v = v.reshape([bsz, num_heads, src_len, head_dim])
        attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.transpose(perm=[2, 0, 1, 3]).reshape([bsz * tgt_len, embed_dim])
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.reshape([tgt_len, bsz, attn_output.shape[1]])
        # if not is_batched:
        #     # squeeze the output if input was unbatched
        #     attn_output = attn_output.squeeze(1)
        return attn_output, None