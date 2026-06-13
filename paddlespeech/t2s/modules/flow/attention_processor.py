import inspect
import math
from typing import Callable, List, Optional, Union
import paddle

class Attention(paddle.nn.Layer):
    """
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-05,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        context_pre_only=None,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self._from_deprecated_attn_block = _from_deprecated_attn_block
        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention
        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )
        if norm_num_groups is not None:
            self.group_norm = paddle.nn.GroupNorm(
                num_channels=query_dim,
                num_groups=norm_num_groups,
                epsilon=eps,
                weight_attr=True,
                bias_attr=True,
            )
        else:
            self.group_norm = None
        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(
                f_channels=query_dim, zq_channels=spatial_norm_dim
            )
        else:
            self.spatial_norm = None
        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = paddle.nn.LayerNorm(normalized_shape=dim_head, epsilon=eps)
            self.norm_k = paddle.nn.LayerNorm(normalized_shape=dim_head, epsilon=eps)
        else:
            raise ValueError(
                f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'"
            )
        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = paddle.nn.LayerNorm(
                normalized_shape=self.cross_attention_dim
            )
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim
            self.norm_cross = paddle.nn.GroupNorm(
                num_channels=norm_cross_num_channels,
                num_groups=cross_attention_norm_num_groups,
                epsilon=1e-05,
                weight_attr=True,
                bias_attr=True,
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )
        self.to_q = paddle.nn.Linear(
            in_features=query_dim, out_features=self.inner_dim, bias_attr=bias
        )
        if not self.only_cross_attention:
            self.to_k = paddle.nn.Linear(
                in_features=self.cross_attention_dim,
                out_features=self.inner_dim,
                bias_attr=bias,
            )
            self.to_v = paddle.nn.Linear(
                in_features=self.cross_attention_dim,
                out_features=self.inner_dim,
                bias_attr=bias,
            )
        else:
            self.to_k = None
            self.to_v = None
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = paddle.nn.Linear(
                in_features=added_kv_proj_dim, out_features=self.inner_dim
            )
            self.add_v_proj = paddle.nn.Linear(
                in_features=added_kv_proj_dim, out_features=self.inner_dim
            )
            if self.context_pre_only is not None:
                self.add_q_proj = paddle.nn.Linear(
                    in_features=added_kv_proj_dim, out_features=self.inner_dim
                )
        self.to_out = paddle.nn.LayerList(sublayers=[])
        self.to_out.append(
            paddle.nn.Linear(
                in_features=self.inner_dim,
                out_features=self.out_dim,
                bias_attr=out_bias,
            )
        )
        self.to_out.append(paddle.nn.Dropout(p=dropout))
        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = paddle.nn.Linear(
                in_features=self.inner_dim,
                out_features=self.out_dim,
                bias_attr=out_bias,
            )
            processor = AttnProcessor()
        processor = AttnProcessor2_0()
        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        """
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, paddle.nn.Layer)
            and not isinstance(processor, paddle.nn.Layer)
        ):
            self._modules.pop("processor")
        self.processor = processor

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        **cross_attention_kwargs,
    ) -> paddle.Tensor:
        """
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k
            for k, _ in cross_attention_kwargs.items()
            if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {
            k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters
        }
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: paddle.Tensor) -> paddle.Tensor:
        """
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_len, dim * head_size
        )
        return tensor

    def head_to_batch_dim(
        self, tensor: paddle.Tensor, out_dim: int = 3
    ) -> paddle.Tensor:
        """
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(
            batch_size, seq_len * extra_dim, head_size, dim // head_size
        )
        tensor = tensor.permute(0, 2, 1, 3)
        if out_dim == 3:
            tensor = tensor.reshape(
                batch_size * head_size, seq_len * extra_dim, dim // head_size
            )
        return tensor

    def get_attention_scores(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        attention_mask: paddle.Tensor = None,
    ) -> paddle.Tensor:
        """
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()
        if attention_mask is None:
            baddbmm_input = paddle.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.place,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1
        attention_scores = paddle.baddbmm(
            input=baddbmm_input,
            x=query,
            y=key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input
        if self.upcast_softmax:
            attention_scores = attention_scores.float()
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores
        attention_probs = attention_probs.to(dtype)
        return attention_probs

    def prepare_attention_mask(
        self,
        attention_mask: paddle.Tensor,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ) -> paddle.Tensor:
        """
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask
        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                padding_shape = (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    target_length,
                )
                padding = paddle.zeros(
                    padding_shape,
                    dtype=attention_mask.dtype,
                    device=attention_mask.place,
                )
                attention_mask = paddle.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = paddle.compat.pad(
                    attention_mask, (0, target_length), value=0.0
                )
        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, axis=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)
        return attention_mask

    def norm_encoder_hidden_states(
        self, encoder_hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert (
            self.norm_cross is not None
        ), "self.norm_cross must be defined to call self.norm_encoder_hidden_states"
        if isinstance(self.norm_cross, paddle.nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, paddle.nn.GroupNorm):
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False
        return encoder_hidden_states

    @paddle.no_grad()
    def fuse_projections(self, fuse=True):
        device = self.to_q.weight.data.place
        dtype = self.to_q.weight.data.dtype
        if not self.is_cross_attention:
            concatenated_weights = paddle.cat(
                [self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data]
            )
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]
            self.to_qkv = paddle.nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias_attr=self.use_bias,
            )
            self.to_qkv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = paddle.cat(
                    [self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data]
                )
                self.to_qkv.bias.copy_(concatenated_bias)
        else:
            concatenated_weights = paddle.cat(
                [self.to_k.weight.data, self.to_v.weight.data]
            )
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]
            self.to_kv = paddle.nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias_attr=self.use_bias,
            )
            self.to_kv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = paddle.cat(
                    [self.to_k.bias.data, self.to_v.bias.data]
                )
                self.to_kv.bias.copy_(concatenated_bias)
        self.fused_projections = fuse


class AttnProcessor:
    """
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
        *args,
        **kwargs,
    ) -> paddle.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

class AttnProcessor2_0:
    """
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        pass
    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
        *args,
        **kwargs,
    ) -> paddle.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                [batch_size, attn.heads, -1, attention_mask.shape[-1]]
            )
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = paddle.transpose(query.view([batch_size, -1, attn.heads, head_dim]),perm = [0,2,1])
        key = paddle.transpose(key.view([batch_size, -1, attn.heads, head_dim]),perm = [0,2,1])
        value =paddle.transpose(value.view([batch_size, -1, attn.heads, head_dim]),perm = [0,2,1])
        hidden_states = paddle.nn.functional.scaled_dot_product_attention(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose([0, 2, 1, 3])
        hidden_states = paddle.transpose(hidden_states,perm = [0,2,1]).reshape(
            [batch_size, -1, attn.heads * head_dim]
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states =paddle.transpose(hidden_states,perm = [0,1,3,2]).reshape(
                [batch_size, channel, height, width]
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
