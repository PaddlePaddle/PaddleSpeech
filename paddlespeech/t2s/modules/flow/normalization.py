import numbers
from typing import Dict, Optional, Tuple

import paddle
from paddle import nn
from .activations import get_activation
from .embeddings import (CombinedTimestepLabelEmbeddings,
                         PixArtAlphaCombinedTimestepSizeEmbeddings)
def get_activation(act_fn):
    if act_fn == "silu":
        return nn.Silu()
    elif act_fn == "mish":
        return nn.Mish()
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
    
class AdaLayerNorm(paddle.nn.Layer):
    """
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = paddle.nn.Embedding(num_embeddings, embedding_dim)
        self.silu = paddle.nn.SiLU()
        self.linear = paddle.nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim * 2
        )
        self.norm = paddle.nn.LayerNorm(
            normalized_shape=embedding_dim, weight_attr=False, bias_attr=False
        )

    def forward(self, x: paddle.Tensor, timestep: paddle.Tensor) -> paddle.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = paddle.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(paddle.nn.Layer):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None
        self.silu = paddle.nn.SiLU()
        self.linear = paddle.nn.Linear(
            in_features=embedding_dim, out_features=6 * embedding_dim, bias_attr=True
        )
        self.norm = paddle.nn.LayerNorm(
            normalized_shape=embedding_dim,
            weight_attr=False,
            bias_attr=False,
            epsilon=1e-06,
        )

    def forward(
        self,
        x: paddle.Tensor,
        timestep: Optional[paddle.Tensor] = None,
        class_labels: Optional[paddle.LongTensor] = None,
        hidden_dtype: Optional[paddle.dtype] = None,
        emb: Optional[paddle.Tensor] = None,
    ) -> Tuple[
        paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor
    ]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormSingle(paddle.nn.Layer):
    """
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
        )
        self.silu = paddle.nn.SiLU()
        self.linear = paddle.nn.Linear(
            in_features=embedding_dim, out_features=6 * embedding_dim, bias_attr=True
        )

    def forward(
        self,
        timestep: paddle.Tensor,
        added_cond_kwargs: Optional[Dict[str, paddle.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[paddle.dtype] = None,
    ) -> Tuple[
        paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor
    ]:
        embedded_timestep = self.emb(
            timestep,
            **added_cond_kwargs,
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class AdaGroupNorm(paddle.nn.Layer):
    """
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int,
        num_groups: int,
        act_fn: Optional[str] = None,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)
        self.linear = paddle.nn.Linear(
            in_features=embedding_dim, out_features=out_dim * 2
        )

    def forward(self, x: paddle.Tensor, emb: paddle.Tensor) -> paddle.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)
        x = paddle.nn.functional.group_norm(
            x=x, num_groups=self.num_groups, epsilon=self.eps
        )
        x = x * (1 + scale) + shift
        return x


class AdaLayerNormContinuous(paddle.nn.Layer):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-05,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = paddle.nn.SiLU()
        self.linear = paddle.nn.Linear(
            in_features=conditioning_embedding_dim,
            out_features=embedding_dim * 2,
            bias_attr=bias,
        )
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(
        self, x: paddle.Tensor, conditioning_embedding: paddle.Tensor
    ) -> paddle.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = paddle.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x

LayerNorm = paddle.nn.LayerNorm


class LayerNorm(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.eps = eps
        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        self.dim = paddle.Size(dim)
        if elementwise_affine:
            self.weight = paddle.nn.parameter.Parameter(paddle.ones(dim))
            self.bias = (
                paddle.nn.parameter.Parameter(paddle.zeros(dim)) if bias else None
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, input):
        return paddle.nn.functional.layer_norm(
            input, self.dim, self.weight, self.bias, self.eps
        )


class RMSNorm(paddle.nn.Layer):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        self.dim = paddle.Size(dim)
        if elementwise_affine:
            self.weight = paddle.nn.parameter.Parameter(paddle.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(paddle.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.eps)
        if self.weight is not None:
            if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states


class GlobalResponseNorm(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.gamma = paddle.nn.parameter.Parameter(paddle.zeros(1, 1, 1, dim))
        self.beta = paddle.nn.parameter.Parameter(paddle.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = paddle.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-06)
        return self.gamma * (x * nx) + self.beta + x
