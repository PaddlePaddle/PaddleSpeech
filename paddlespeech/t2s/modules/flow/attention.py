from typing import Any, Dict, Optional
import paddle
from .attention_processor import Attention

def _chunked_feed_forward(
    ff: paddle.nn.Layer, hidden_states: paddle.Tensor, chunk_dim: int, chunk_size: int
):
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )
    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = paddle.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output

class SinusoidalPositionalEmbedding(paddle.nn.Layer):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = paddle.arange(max_seq_length).unsqueeze(1)
        div_term = paddle.exp(
            x=paddle.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = paddle.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = paddle.sin(position * div_term)
        pe[0, :, 1::2] = paddle.cos(position * div_term)
        self.register_buffer(name="pe", tensor=pe)

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x

class GatedSelfAttentionDense(paddle.nn.Layer):
    """
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features=context_dim, out_features=query_dim)
        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=query_dim)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=query_dim)
        self.add_parameter(
            name="alpha_attn",
            parameter=paddle.nn.parameter.Parameter(paddle.tensor(0.0)),
        )
        self.add_parameter(
            name="alpha_dense",
            parameter=paddle.nn.parameter.Parameter(paddle.tensor(0.0)),
        )
        self.enabled = True

    def forward(self, x: paddle.Tensor, objs: paddle.Tensor) -> paddle.Tensor:
        if not self.enabled:
            return x
        n_visual = x.shape[1]
        objs = self.linear(objs)
        x = (
            x
            + self.alpha_attn.tanh()
            * self.attn(self.norm1(paddle.cat([x, objs], dim=1)))[:, :n_visual, :]
        )
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))
        return x


class BasicTransformerBlock(paddle.nn.Layer):
    """
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-05,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm_zero = (
            num_embeds_ada_norm is not None and norm_type == "ada_norm_zero"
        )
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None and norm_type == "ada_norm"
        )
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"
        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )
        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm
        if positional_embeddings and num_positional_embeddings is None:
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )
        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None
        

        self.norm1 = paddle.nn.LayerNorm(
            normalized_shape=dim,
            weight_attr=norm_elementwise_affine,
            bias_attr=norm_elementwise_affine,
            epsilon=norm_eps,
        )
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )
        if cross_attention_dim is not None or double_self_attention:
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2 = paddle.nn.LayerNorm(
                    normalized_shape=dim,
                    epsilon=norm_eps,
                    weight_attr=norm_elementwise_affine,
                    bias_attr=norm_elementwise_affine,
                )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim
                if not double_self_attention
                else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )
        else:
            self.norm2 = None
            self.attn2 = None
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )
        elif norm_type in [
            "ada_norm_zero",
            "ada_norm",
            "layer_norm",
            "ada_norm_continuous",
        ]:
            self.norm3 = paddle.nn.LayerNorm(
                normalized_shape=dim,
                epsilon=norm_eps,
                weight_attr=norm_elementwise_affine,
                bias_attr=norm_elementwise_affine,
            )
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(
                dim, cross_attention_dim, num_attention_heads, attention_head_dim
            )
        if norm_type == "ada_norm_single":
            self.scale_shift_table = paddle.nn.parameter.Parameter(
                paddle.randn(6, dim) / dim**0.5
            )
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        timestep: Optional[paddle.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[paddle.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, paddle.Tensor]] = None,
    ) -> paddle.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored."
                )
        batch_size = hidden_states.shape[0]
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention
            else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError("Incorrect norm")
            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)
        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )
        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)
        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


