from typing import Any, Dict, Optional
from .attention_processor import Attention
import paddle
from paddle import nn
from paddlespeech.t2s.modules.flow.lora import LoRACompatibleLinear
class SnakeBeta(paddle.nn.Layer):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self,
        in_features,
        out_features,
        alpha=1.0,
        alpha_trainable=True,
        alpha_logscale=True,
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.in_features = (
            out_features if isinstance(out_features, list) else [out_features]
        )
        self.proj = LoRACompatibleLinear(
            in_features, out_features
        )
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = paddle.nn.parameter.Parameter(
                paddle.zeros(self.in_features) * alpha
            )
            self.beta = paddle.nn.parameter.Parameter(
                paddle.zeros(self.in_features) * alpha
            )
        else:
            self.alpha = paddle.nn.parameter.Parameter(
                paddle.ones(self.in_features) * alpha
            )
            self.beta = paddle.nn.parameter.Parameter(
                paddle.ones(self.in_features) * alpha
            )
        self.alpha.stop_gradient = not alpha_trainable
        self.beta.stop_gradient = not alpha_trainable
        self.no_div_by_zero = 1e-09

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = paddle.exp(x=self.alpha)
            beta = paddle.exp(x=self.beta)
        else:
            alpha = self.alpha
            beta = self.beta
        x = x + 1.0 / (beta + self.no_div_by_zero) * paddle.pow(
            paddle.sin(x * alpha), 2
        )
        return x
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class GELU(nn.Layer):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias_attr=bias)
        self.approximate = approximate

    def gelu(self, gate: paddle.Tensor) -> paddle.Tensor:
        if self.approximate == "tanh":
            approximate_bool = True
        else:
            approximate_bool = False

        if gate.dtype == paddle.float16:
            return F.gelu(gate.astype(paddle.float32), approximate=approximate_bool).astype(paddle.float16)
        else:
            return F.gelu(gate, approximate=approximate_bool)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states) 
        return hidden_states

class FeedForward(paddle.nn.Layer):
    """
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        act_fn = GELU(dim, inner_dim)
        self.net = paddle.nn.LayerList(sublayers=[])
        self.net.append(act_fn)
        self.net.append(paddle.nn.Dropout(p=dropout))
        self.net.append(LoRACompatibleLinear(inner_dim, dim_out))
        if final_dropout:
            self.net.append(paddle.nn.Dropout(p=dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states



class BasicTransformerBlock(paddle.nn.Layer):
    """
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
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
        final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm_zero = (
            num_embeds_ada_norm is not None and norm_type == "ada_norm_zero"
        )
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None and norm_type == "ada_norm"
        )
        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )
        
        self.norm1 = paddle.nn.LayerNorm(
            normalized_shape=dim,
            weight_attr=norm_elementwise_affine,
            bias_attr=norm_elementwise_affine,
        )
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        self.norm2 = None
        self.attn2 = None
        self.norm3 = paddle.nn.LayerNorm(
            normalized_shape=dim,
            weight_attr=norm_elementwise_affine,
            bias_attr=norm_elementwise_affine,
        )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
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
    ):
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention
            else None,
            attention_mask=encoder_attention_mask
            if self.only_cross_attention
            else attention_mask,
            **cross_attention_kwargs,
        )
        
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states
        
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        norm_hidden_states = self.norm3(hidden_states)
        
        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self._chunk_size is not None:
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )
            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = paddle.cat(
                [
                    self.ff(hid_slice)
                    for hid_slice in norm_hidden_states.chunk(
                        num_chunks, dim=self._chunk_dim
                    )
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)
        
        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = ff_output + hidden_states
        return hidden_states
