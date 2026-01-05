import inspect
import math
from importlib import import_module
from typing import Callable, List, Optional, Union

import paddle

from ..image_processor import IPAdapterMaskProcessor
from ..utils import deprecate, logging
from ..utils.import_utils import is_torch_npu_available, is_xformers_available
from ..utils.torch_utils import maybe_allow_in_graph
from .lora import LoRALinearLayer

logger = logging.get_logger(__name__)
if is_torch_npu_available():
    import torch_npu
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


@maybe_allow_in_graph
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
        if processor is None:
            processor = (
                AttnProcessor2_0()
>>>>>>                if hasattr(torch.nn.functional, "scaled_dot_product_attention")
                and self.scale_qk
                else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_npu_flash_attention(self, use_npu_flash_attention: bool) -> None:
        """
        Set whether to use npu flash attention from `torch_npu` or not.

        """
        if use_npu_flash_attention:
            processor = AttnProcessorNPU()
        else:
            processor = (
                AttnProcessor2_0()
>>>>>>                if hasattr(torch.nn.functional, "scaled_dot_product_attention")
                and self.scale_qk
                else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: bool,
        attention_op: Optional[Callable] = None,
    ) -> None:
        """
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                The attention operation to use. Defaults to `None` which uses the default attention operation from
                `xformers`.
        """
        is_lora = hasattr(self, "processor") and isinstance(
            self.processor, LORA_ATTENTION_PROCESSORS
        )
        is_custom_diffusion = hasattr(self, "processor") and isinstance(
            self.processor,
            (
                CustomDiffusionAttnProcessor,
                CustomDiffusionXFormersAttnProcessor,
                CustomDiffusionAttnProcessor2_0,
            ),
        )
        is_added_kv_processor = hasattr(self, "processor") and isinstance(
            self.processor,
            (
                AttnAddedKVProcessor,
                AttnAddedKVProcessor2_0,
                SlicedAttnAddedKVProcessor,
                XFormersAttnAddedKVProcessor,
                LoRAAttnAddedKVProcessor,
            ),
        )
        if use_memory_efficient_attention_xformers:
            if is_added_kv_processor and (is_lora or is_custom_diffusion):
                raise NotImplementedError(
                    f"Memory efficient attention is currently not supported for LoRA or custom diffusion for attention processor type {self.processor}"
                )
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    "Refer to https://github.com/facebookresearch/xformers for more information on how to install xformers",
                    name="xformers",
                )
            elif not paddle.device.cuda.device_count() >= 1:
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only available for GPU "
                )
            else:
                try:
                    _ = xformers.ops.memory_efficient_attention(
                        paddle.randn((1, 2, 40), device="cuda"),
                        paddle.randn((1, 2, 40), device="cuda"),
                        paddle.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e
            if is_lora:
                processor = LoRAXFormersAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                    attention_op=attention_op,
                )
                processor.set_state_dict(state_dict=self.processor.state_dict())
                processor.to(self.processor.to_q_lora.up.weight.place)
            elif is_custom_diffusion:
                processor = CustomDiffusionXFormersAttnProcessor(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    attention_op=attention_op,
                )
                processor.set_state_dict(state_dict=self.processor.state_dict())
                if hasattr(self.processor, "to_k_custom_diffusion"):
                    processor.to(self.processor.to_k_custom_diffusion.weight.place)
            elif is_added_kv_processor:
                logger.info(
                    "Memory efficient attention with `xformers` might currently not work correctly if an attention mask is required for the attention operation."
                )
                processor = XFormersAttnAddedKVProcessor(attention_op=attention_op)
            else:
                processor = XFormersAttnProcessor(attention_op=attention_op)
        elif is_lora:
            attn_processor_class = (
                LoRAAttnProcessor2_0
>>>>>>                if hasattr(torch.nn.functional, "scaled_dot_product_attention")
                else LoRAAttnProcessor
            )
            processor = attn_processor_class(
                hidden_size=self.processor.hidden_size,
                cross_attention_dim=self.processor.cross_attention_dim,
                rank=self.processor.rank,
            )
            processor.set_state_dict(state_dict=self.processor.state_dict())
            processor.to(self.processor.to_q_lora.up.weight.place)
        elif is_custom_diffusion:
            attn_processor_class = (
                CustomDiffusionAttnProcessor2_0
>>>>>>                if hasattr(torch.nn.functional, "scaled_dot_product_attention")
                else CustomDiffusionAttnProcessor
            )
            processor = attn_processor_class(
                train_kv=self.processor.train_kv,
                train_q_out=self.processor.train_q_out,
                hidden_size=self.processor.hidden_size,
                cross_attention_dim=self.processor.cross_attention_dim,
            )
            processor.set_state_dict(state_dict=self.processor.state_dict())
            if hasattr(self.processor, "to_k_custom_diffusion"):
                processor.to(self.processor.to_k_custom_diffusion.weight.place)
        else:
            processor = (
                AttnProcessor2_0()
>>>>>>                if hasattr(torch.nn.functional, "scaled_dot_product_attention")
                and self.scale_qk
                else AttnProcessor()
            )
        self.set_processor(processor)

    def set_attention_slice(self, slice_size: int) -> None:
        """
        Set the slice size for attention computation.

        Args:
            slice_size (`int`):
                The slice size for attention computation.
        """
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(
                f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}."
            )
        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = AttnAddedKVProcessor()
        else:
            processor = (
                AttnProcessor2_0()
>>>>>>                if hasattr(torch.nn.functional, "scaled_dot_product_attention")
                and self.scale_qk
                else AttnProcessor()
            )
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
            logger.info(
                f"You are removing possibly trained weights of {self.processor} with {processor}"
            )
            self._modules.pop("processor")
        self.processor = processor

    def get_processor(
        self, return_deprecated_lora: bool = False
    ) -> "AttentionProcessor":
        """
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor
        is_lora_activated = {
            name: (module.lora_layer is not None)
            for name, module in self.named_sublayers(include_self=True)
            if hasattr(module, "lora_layer")
        }
        if not any(is_lora_activated.values()):
            return self.processor
        is_lora_activated.pop("add_k_proj", None)
        is_lora_activated.pop("add_v_proj", None)
        if not all(is_lora_activated.values()):
            raise ValueError(
                f"Make sure that either all layers or no layers have LoRA activated, but have {is_lora_activated}"
            )
        non_lora_processor_cls_name = self.processor.__class__.__name__
        lora_processor_cls = getattr(
            import_module(__name__), "LoRA" + non_lora_processor_cls_name
        )
        hidden_size = self.inner_dim
        if lora_processor_cls in [
            LoRAAttnProcessor,
            LoRAAttnProcessor2_0,
            LoRAXFormersAttnProcessor,
        ]:
            kwargs = {
                "cross_attention_dim": self.cross_attention_dim,
                "rank": self.to_q.lora_layer.rank,
                "network_alpha": self.to_q.lora_layer.network_alpha,
                "q_rank": self.to_q.lora_layer.rank,
                "q_hidden_size": self.to_q.lora_layer.out_features,
                "k_rank": self.to_k.lora_layer.rank,
                "k_hidden_size": self.to_k.lora_layer.out_features,
                "v_rank": self.to_v.lora_layer.rank,
                "v_hidden_size": self.to_v.lora_layer.out_features,
                "out_rank": self.to_out[0].lora_layer.rank,
                "out_hidden_size": self.to_out[0].lora_layer.out_features,
            }
            if hasattr(self.processor, "attention_op"):
                kwargs["attention_op"] = self.processor.attention_op
            lora_processor = lora_processor_cls(hidden_size, **kwargs)
            lora_processor.to_q_lora.load_state_dict(self.to_q.lora_layer.state_dict())
            lora_processor.to_k_lora.load_state_dict(self.to_k.lora_layer.state_dict())
            lora_processor.to_v_lora.load_state_dict(self.to_v.lora_layer.state_dict())
            lora_processor.to_out_lora.load_state_dict(
                self.to_out[0].lora_layer.state_dict()
            )
        elif lora_processor_cls == LoRAAttnAddedKVProcessor:
            lora_processor = lora_processor_cls(
                hidden_size,
                cross_attention_dim=self.add_k_proj.weight.shape[0],
                rank=self.to_q.lora_layer.rank,
                network_alpha=self.to_q.lora_layer.network_alpha,
            )
            lora_processor.to_q_lora.load_state_dict(self.to_q.lora_layer.state_dict())
            lora_processor.to_k_lora.load_state_dict(self.to_k.lora_layer.state_dict())
            lora_processor.to_v_lora.load_state_dict(self.to_v.lora_layer.state_dict())
            lora_processor.to_out_lora.load_state_dict(
                self.to_out[0].lora_layer.state_dict()
            )
            if self.add_k_proj.lora_layer is not None:
                lora_processor.add_k_proj_lora.load_state_dict(
                    self.add_k_proj.lora_layer.state_dict()
                )
                lora_processor.add_v_proj_lora.load_state_dict(
                    self.add_v_proj.lora_layer.state_dict()
                )
            else:
                lora_processor.add_k_proj_lora = None
                lora_processor.add_v_proj_lora = None
        else:
            raise ValueError(f"{lora_processor_cls} does not exist.")
        return lora_processor

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
        print("processor:", self.processor, "2" * 200)
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
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
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
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
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


class CustomDiffusionAttnProcessor(paddle.nn.Layer):
    """
    Processor for implementing attention for the Custom Diffusion method.

    Args:
        train_kv (`bool`, defaults to `True`):
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `True`):
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
    """

    def __init__(
        self,
        train_kv: bool = True,
        train_q_out: bool = True,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.train_kv = train_kv
        self.train_q_out = train_q_out
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        if self.train_kv:
            self.to_k_custom_diffusion = paddle.nn.Linear(
                in_features=cross_attention_dim or hidden_size,
                out_features=hidden_size,
                bias_attr=False,
            )
            self.to_v_custom_diffusion = paddle.nn.Linear(
                in_features=cross_attention_dim or hidden_size,
                out_features=hidden_size,
                bias_attr=False,
            )
        if self.train_q_out:
            self.to_q_custom_diffusion = paddle.nn.Linear(
                in_features=hidden_size, out_features=hidden_size, bias_attr=False
            )
            self.to_out_custom_diffusion = paddle.nn.LayerList(sublayers=[])
            self.to_out_custom_diffusion.append(
                paddle.nn.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    bias_attr=out_bias,
                )
            )
            self.to_out_custom_diffusion.append(paddle.nn.Dropout(p=dropout))

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states).to(attn.to_q.weight.dtype)
        else:
            query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))
        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
        if self.train_kv:
            key = self.to_k_custom_diffusion(
                encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype)
            )
            value = self.to_v_custom_diffusion(
                encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype)
            )
            key = key.to(attn.to_q.weight.dtype)
            value = value.to(attn.to_q.weight.dtype)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = paddle.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        if self.train_q_out:
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AttnAddedKVProcessor:
    """
    Processor for performing attention-related computations with extra learnable key and value matrices for the text
    encoder.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        *args,
        **kwargs,
    ) -> paddle.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        residual = hidden_states
        hidden_states = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], -1
        ).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(
            encoder_hidden_states_key_proj
        )
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(
            encoder_hidden_states_value_proj
        )
        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = paddle.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = paddle.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual
        return hidden_states


class AttnAddedKVProcessor2_0:
    """
    Processor for performing scaled dot-product attention (enabled by default if you're using PyTorch 2.0), with extra
    learnable key and value matrices for the text encoder.
    """

    def __init__(self):
>>>>>>        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnAddedKVProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        *args,
        **kwargs,
    ) -> paddle.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        residual = hidden_states
        hidden_states = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], -1
        ).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size, out_dim=4
        )
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query, out_dim=4)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(
            encoder_hidden_states_key_proj, out_dim=4
        )
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(
            encoder_hidden_states_value_proj, out_dim=4
        )
        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key, out_dim=4)
            value = attn.head_to_batch_dim(value, out_dim=4)
            key = paddle.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = paddle.cat([encoder_hidden_states_value_proj, value], dim=2)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj
        hidden_states = paddle.nn.functional.scaled_dot_product_attention(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose([0, 2, 1, 3])
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, residual.shape[1]
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual
        return hidden_states


class JointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
>>>>>>        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.FloatTensor,
        encoder_hidden_states: paddle.FloatTensor = None,
        attention_mask: Optional[paddle.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> paddle.FloatTensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        batch_size = encoder_hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        query = paddle.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = paddle.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = paddle.cat([value, encoder_hidden_states_value_proj], dim=1)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = (
            hidden_states
        ) = paddle.nn.functional.scaled_dot_product_attention(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(
            [0, 2, 1, 3]
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        return hidden_states, encoder_hidden_states


class FusedJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
>>>>>>        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.FloatTensor,
        encoder_hidden_states: paddle.FloatTensor = None,
        attention_mask: Optional[paddle.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> paddle.FloatTensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        batch_size = encoder_hidden_states.shape[0]
        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = paddle.compat.split(qkv, split_size, dim=-1)
        encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
        split_size = encoder_qkv.shape[-1] // 3
        (
            encoder_hidden_states_query_proj,
            encoder_hidden_states_key_proj,
            encoder_hidden_states_value_proj,
        ) = paddle.compat.split(encoder_qkv, split_size, dim=-1)
        query = paddle.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = paddle.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = paddle.cat([value, encoder_hidden_states_value_proj], dim=1)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = (
            hidden_states
        ) = paddle.nn.functional.scaled_dot_product_attention(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(
            [0, 2, 1, 3]
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        return hidden_states, encoder_hidden_states


class XFormersAttnAddedKVProcessor:
    """
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        residual = hidden_states
        hidden_states = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], -1
        ).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(
            encoder_hidden_states_key_proj
        )
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(
            encoder_hidden_states_value_proj
        )
        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = paddle.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = paddle.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj
        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual
        return hidden_states


class XFormersAttnProcessor:
    """
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

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
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        batch_size, key_tokens, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, key_tokens, batch_size
        )
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)
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
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        hidden_states = hidden_states.to(query.dtype)
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


class AttnProcessorNPU:
    """
    Processor for implementing flash attention using torch_npu. Torch_npu supports only fp16 and bf16 data types. If
    fp32 is used, F.scaled_dot_product_attention will be used for computation, but the acceleration effect on NPU is
    not significant.

    """

    def __init__(self):
        if not is_torch_npu_available():
            raise ImportError(
                "AttnProcessorNPU requires torch_npu extensions and is supported only on npu devices."
            )

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
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
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
                batch_size, attn.heads, -1, attention_mask.shape[-1]
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
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if query.dtype in (paddle.float16, paddle.bfloat16):
            hidden_states = torch_npu.npu_fusion_attention(
                query,
                key,
                value,
                attn.heads,
                input_layout="BNSD",
                pse=None,
                atten_mask=attention_mask,
                scale=1.0 / math.sqrt(query.shape[-1]),
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0,
                sync=False,
                inner_precise=0,
            )[0]
        else:
            hidden_states = paddle.nn.functional.scaled_dot_product_attention(
                query.transpose([0, 2, 1, 3]),
                key.transpose([0, 2, 1, 3]),
                value.transpose([0, 2, 1, 3]),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose([0, 2, 1, 3])
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
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
        return

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
                batch_size, attn.heads, -1, attention_mask.shape[-1]
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
        query = paddle.transpose(query.view(batch_size, -1, attn.heads, head_dim),perm = [0,2,1])
        key = paddle.transpose(key.view(batch_size, -1, attn.heads, head_dim),perm = [0,2,1])
        value =paddle.transpose( value.view(batch_size, -1, attn.heads, head_dim),perm = [0,2,1])
        hidden_states = paddle.nn.functional.scaled_dot_product_attention(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose([0, 2, 1, 3])
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
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


class HunyuanAttnProcessor2_0:
    """
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the HunyuanDiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
>>>>>>        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
        image_rotary_emb: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        from .embeddings import apply_rotary_emb

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
                batch_size, attn.heads, -1, attention_mask.shape[-1]
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
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)
        hidden_states = paddle.nn.functional.scaled_dot_product_attention(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose([0, 2, 1, 3])
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
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


class FusedAttnProcessor2_0:
    """
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). It uses
    fused projection layers. For self-attention modules, all projection matrices (i.e., query, key, value) are fused.
    For cross-attention modules, key and value projection matrices are fused.

    <Tip warning={true}>

    This API is currently 🧪 experimental in nature and can change in future.

    </Tip>
    """

    def __init__(self):
>>>>>>        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "FusedAttnProcessor2_0 requires at least PyTorch 2.0, to use it. Please upgrade PyTorch to > 2.0."
            )

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
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
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
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )
        if encoder_hidden_states is None:
            qkv = attn.to_qkv(hidden_states)
            split_size = qkv.shape[-1] // 3
            query, key, value = paddle.compat.split(qkv, split_size, dim=-1)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            query = attn.to_q(hidden_states)
            kv = attn.to_kv(encoder_hidden_states)
            split_size = kv.shape[-1] // 2
            key, value = paddle.compat.split(kv, split_size, dim=-1)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = paddle.nn.functional.scaled_dot_product_attention(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose([0, 2, 1, 3])
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
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


class CustomDiffusionXFormersAttnProcessor(paddle.nn.Layer):
    """
    Processor for implementing memory efficient attention using xFormers for the Custom Diffusion method.

    Args:
    train_kv (`bool`, defaults to `True`):
        Whether to newly train the key and value matrices corresponding to the text features.
    train_q_out (`bool`, defaults to `True`):
        Whether to newly train query matrices corresponding to the latent image features.
    hidden_size (`int`, *optional*, defaults to `None`):
        The hidden size of the attention layer.
    cross_attention_dim (`int`, *optional*, defaults to `None`):
        The number of channels in the `encoder_hidden_states`.
    out_bias (`bool`, defaults to `True`):
        Whether to include the bias parameter in `train_q_out`.
    dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability to use.
    attention_op (`Callable`, *optional*, defaults to `None`):
        The base
        [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to use
        as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best operator.
    """

    def __init__(
        self,
        train_kv: bool = True,
        train_q_out: bool = False,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        dropout: float = 0.0,
        attention_op: Optional[Callable] = None,
    ):
        super().__init__()
        self.train_kv = train_kv
        self.train_q_out = train_q_out
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.attention_op = attention_op
        if self.train_kv:
            self.to_k_custom_diffusion = paddle.nn.Linear(
                in_features=cross_attention_dim or hidden_size,
                out_features=hidden_size,
                bias_attr=False,
            )
            self.to_v_custom_diffusion = paddle.nn.Linear(
                in_features=cross_attention_dim or hidden_size,
                out_features=hidden_size,
                bias_attr=False,
            )
        if self.train_q_out:
            self.to_q_custom_diffusion = paddle.nn.Linear(
                in_features=hidden_size, out_features=hidden_size, bias_attr=False
            )
            self.to_out_custom_diffusion = paddle.nn.LayerList(sublayers=[])
            self.to_out_custom_diffusion.append(
                paddle.nn.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    bias_attr=out_bias,
                )
            )
            self.to_out_custom_diffusion.append(paddle.nn.Dropout(p=dropout))

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states).to(attn.to_q.weight.dtype)
        else:
            query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))
        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
        if self.train_kv:
            key = self.to_k_custom_diffusion(
                encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype)
            )
            value = self.to_v_custom_diffusion(
                encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype)
            )
            key = key.to(attn.to_q.weight.dtype)
            value = value.to(attn.to_q.weight.dtype)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = paddle.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        if self.train_q_out:
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CustomDiffusionAttnProcessor2_0(paddle.nn.Layer):
    """
    Processor for implementing attention for the Custom Diffusion method using PyTorch 2.0’s memory-efficient scaled
    dot-product attention.

    Args:
        train_kv (`bool`, defaults to `True`):
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `True`):
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
    """

    def __init__(
        self,
        train_kv: bool = True,
        train_q_out: bool = True,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.train_kv = train_kv
        self.train_q_out = train_q_out
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        if self.train_kv:
            self.to_k_custom_diffusion = paddle.nn.Linear(
                in_features=cross_attention_dim or hidden_size,
                out_features=hidden_size,
                bias_attr=False,
            )
            self.to_v_custom_diffusion = paddle.nn.Linear(
                in_features=cross_attention_dim or hidden_size,
                out_features=hidden_size,
                bias_attr=False,
            )
        if self.train_q_out:
            self.to_q_custom_diffusion = paddle.nn.Linear(
                in_features=hidden_size, out_features=hidden_size, bias_attr=False
            )
            self.to_out_custom_diffusion = paddle.nn.LayerList(sublayers=[])
            self.to_out_custom_diffusion.append(
                paddle.nn.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    bias_attr=out_bias,
                )
            )
            self.to_out_custom_diffusion.append(paddle.nn.Dropout(p=dropout))

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states)
        else:
            query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
        if self.train_kv:
            key = self.to_k_custom_diffusion(
                encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype)
            )
            value = self.to_v_custom_diffusion(
                encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype)
            )
            key = key.to(attn.to_q.weight.dtype)
            value = value.to(attn.to_q.weight.dtype)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = paddle.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()
        inner_dim = hidden_states.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = paddle.nn.functional.scaled_dot_product_attention(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose([0, 2, 1, 3])
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        if self.train_q_out:
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class SlicedAttnProcessor:
    """
    Processor for implementing sliced attention.

    Args:
        slice_size (`int`, *optional*):
            The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
            `attention_head_dim` must be a multiple of the `slice_size`.
    """

    def __init__(self, slice_size: int):
        self.slice_size = slice_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        residual = hidden_states
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
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = paddle.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads),
            device=query.place,
            dtype=query.dtype,
        )
        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size
            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = (
                attention_mask[start_idx:end_idx]
                if attention_mask is not None
                else None
            )
            attn_slice = attn.get_attention_scores(
                query_slice, key_slice, attn_mask_slice
            )
            attn_slice = paddle.bmm(attn_slice, value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice
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


class SlicedAttnAddedKVProcessor:
    """
    Processor for implementing sliced attention with extra learnable key and value matrices for the text encoder.

    Args:
        slice_size (`int`, *optional*):
            The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
            `attention_head_dim` must be a multiple of the `slice_size`.
    """

    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(
        self,
        attn: "Attention",
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        hidden_states = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], -1
        ).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(
            encoder_hidden_states_key_proj
        )
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(
            encoder_hidden_states_value_proj
        )
        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = paddle.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = paddle.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj
        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = paddle.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads),
            device=query.place,
            dtype=query.dtype,
        )
        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size
            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = (
                attention_mask[start_idx:end_idx]
                if attention_mask is not None
                else None
            )
            attn_slice = attn.get_attention_scores(
                query_slice, key_slice, attn_mask_slice
            )
            attn_slice = paddle.bmm(attn_slice, value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual
        return hidden_states


class SpatialNorm(paddle.nn.Layer):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(self, f_channels: int, zq_channels: int):
        super().__init__()
        self.norm_layer = paddle.nn.GroupNorm(
            num_channels=f_channels,
            num_groups=32,
            epsilon=1e-06,
            weight_attr=True,
            bias_attr=True,
        )
        self.conv_y = paddle.nn.Conv2d(
            zq_channels, f_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv_b = paddle.nn.Conv2d(
            zq_channels, f_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, f: paddle.Tensor, zq: paddle.Tensor) -> paddle.Tensor:
        f_size = f.shape[-2:]
        zq = paddle.nn.functional.interpolate(x=zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class LoRAAttnProcessor(paddle.nn.Layer):
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        rank: int = 4,
        network_alpha: Optional[int] = None,
        **kwargs,
    ):
        deprecation_message = "Using LoRAAttnProcessor is deprecated. Please use the PEFT backend for all things LoRA. You can install PEFT by running `pip install peft`."
        deprecate(
            "LoRAAttnProcessor", "0.30.0", deprecation_message, standard_warn=False
        )
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size
        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size
        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = (
            out_hidden_size if out_hidden_size is not None else hidden_size
        )
        self.to_q_lora = LoRALinearLayer(
            q_hidden_size, q_hidden_size, q_rank, network_alpha
        )
        self.to_k_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
        )
        self.to_v_lora = LoRALinearLayer(
            cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha
        )
        self.to_out_lora = LoRALinearLayer(
            out_hidden_size, out_hidden_size, out_rank, network_alpha
        )

    def __call__(
        self, attn: Attention, hidden_states: paddle.Tensor, **kwargs
    ) -> paddle.Tensor:
        self_cls_name = self.__class__.__name__
        deprecate(
            self_cls_name,
            "0.26.0",
            f"Make sure use {self_cls_name[4:]} instead by settingLoRA layers to `self.{{to_q,to_k,to_v,to_out[0]}}.lora_layer` respectively. This will be done automatically when using `LoraLoaderMixin.load_lora_weights`",
        )
        attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.place)
        attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.place)
        attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.place)
        attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.place)
        attn._modules.pop("processor")
        attn.processor = AttnProcessor()
        return attn.processor(attn, hidden_states, **kwargs)


class LoRAAttnProcessor2_0(paddle.nn.Layer):
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        rank: int = 4,
        network_alpha: Optional[int] = None,
        **kwargs,
    ):
        deprecation_message = "Using LoRAAttnProcessor is deprecated. Please use the PEFT backend for all things LoRA. You can install PEFT by running `pip install peft`."
        deprecate(
            "LoRAAttnProcessor2_0", "0.30.0", deprecation_message, standard_warn=False
        )
        super().__init__()
>>>>>>        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size
        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size
        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = (
            out_hidden_size if out_hidden_size is not None else hidden_size
        )
        self.to_q_lora = LoRALinearLayer(
            q_hidden_size, q_hidden_size, q_rank, network_alpha
        )
        self.to_k_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
        )
        self.to_v_lora = LoRALinearLayer(
            cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha
        )
        self.to_out_lora = LoRALinearLayer(
            out_hidden_size, out_hidden_size, out_rank, network_alpha
        )

    def __call__(
        self, attn: Attention, hidden_states: paddle.Tensor, **kwargs
    ) -> paddle.Tensor:
        self_cls_name = self.__class__.__name__
        deprecate(
            self_cls_name,
            "0.26.0",
            f"Make sure use {self_cls_name[4:]} instead by settingLoRA layers to `self.{{to_q,to_k,to_v,to_out[0]}}.lora_layer` respectively. This will be done automatically when using `LoraLoaderMixin.load_lora_weights`",
        )
        attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.place)
        attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.place)
        attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.place)
        attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.place)
        attn._modules.pop("processor")
        attn.processor = AttnProcessor2_0()
        return attn.processor(attn, hidden_states, **kwargs)


class LoRAXFormersAttnProcessor(paddle.nn.Layer):
    """
    Processor for implementing the LoRA attention mechanism with memory efficient attention using xFormers.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
        kwargs (`dict`):
            Additional keyword arguments to pass to the `LoRALinearLayer` layers.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        rank: int = 4,
        attention_op: Optional[Callable] = None,
        network_alpha: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.attention_op = attention_op
        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size
        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size
        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = (
            out_hidden_size if out_hidden_size is not None else hidden_size
        )
        self.to_q_lora = LoRALinearLayer(
            q_hidden_size, q_hidden_size, q_rank, network_alpha
        )
        self.to_k_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
        )
        self.to_v_lora = LoRALinearLayer(
            cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha
        )
        self.to_out_lora = LoRALinearLayer(
            out_hidden_size, out_hidden_size, out_rank, network_alpha
        )

    def __call__(
        self, attn: Attention, hidden_states: paddle.Tensor, **kwargs
    ) -> paddle.Tensor:
        self_cls_name = self.__class__.__name__
        deprecate(
            self_cls_name,
            "0.26.0",
            f"Make sure use {self_cls_name[4:]} instead by settingLoRA layers to `self.{{to_q,to_k,to_v,add_k_proj,add_v_proj,to_out[0]}}.lora_layer` respectively. This will be done automatically when using `LoraLoaderMixin.load_lora_weights`",
        )
        attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.place)
        attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.place)
        attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.place)
        attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.place)
        attn._modules.pop("processor")
        attn.processor = XFormersAttnProcessor()
        return attn.processor(attn, hidden_states, **kwargs)


class LoRAAttnAddedKVProcessor(paddle.nn.Layer):
    """
    Processor for implementing the LoRA attention mechanism with extra learnable key and value matrices for the text
    encoder.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
        kwargs (`dict`):
            Additional keyword arguments to pass to the `LoRALinearLayer` layers.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        rank: int = 4,
        network_alpha: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.add_k_proj_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
        )
        self.add_v_proj_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
        )
        self.to_k_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(
            hidden_size, hidden_size, rank, network_alpha
        )

    def __call__(
        self, attn: Attention, hidden_states: paddle.Tensor, **kwargs
    ) -> paddle.Tensor:
        self_cls_name = self.__class__.__name__
        deprecate(
            self_cls_name,
            "0.26.0",
            f"Make sure use {self_cls_name[4:]} instead by settingLoRA layers to `self.{{to_q,to_k,to_v,add_k_proj,add_v_proj,to_out[0]}}.lora_layer` respectively. This will be done automatically when using `LoraLoaderMixin.load_lora_weights`",
        )
        attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.place)
        attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.place)
        attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.place)
        attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.place)
        attn._modules.pop("processor")
        attn.processor = AttnAddedKVProcessor()
        return attn.processor(attn, hidden_states, **kwargs)


class IPAdapterAttnProcessor(paddle.nn.Layer):
    """
    Attention processor for Multiple IP-Adapters.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or List[`float`], defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(
        self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens
        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError(
                "`scale` should be a list of integers with the same length as `num_tokens`."
            )
        self.scale = scale
        self.to_k_ip = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Linear(
                    in_features=cross_attention_dim,
                    out_features=hidden_size,
                    bias_attr=False,
                )
                for _ in range(len(num_tokens))
            ]
        )
        self.to_v_ip = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Linear(
                    in_features=cross_attention_dim,
                    out_features=hidden_size,
                    bias_attr=False,
                )
                for _ in range(len(num_tokens))
            ]
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[paddle.Tensor] = None,
    ):
        residual = hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release. Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                deprecate(
                    "encoder_hidden_states not a tuple",
                    "1.0.0",
                    deprecation_message,
                    standard_warn=False,
                )
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )
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
        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match length of self.scale array ({len(self.scale)}) and number of ip_hidden_states ({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(
                    zip(ip_adapter_masks, self.scale, ip_hidden_states)
                ):
                    if not isinstance(mask, paddle.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape [1, num_images_for_ip_adapter, height, width]. Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]
                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])
                        ip_key = attn.head_to_batch_dim(ip_key)
                        ip_value = attn.head_to_batch_dim(ip_value)
                        ip_attention_probs = attn.get_attention_scores(
                            query, ip_key, None
                        )
                        _current_ip_hidden_states = paddle.bmm(
                            ip_attention_probs, ip_value
                        )
                        _current_ip_hidden_states = attn.batch_to_head_dim(
                            _current_ip_hidden_states
                        )
                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )
                        mask_downsample = mask_downsample.to(
                            dtype=query.dtype, device=query.place
                        )
                        hidden_states = hidden_states + scale[i] * (
                            _current_ip_hidden_states * mask_downsample
                        )
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)
                    ip_key = attn.head_to_batch_dim(ip_key)
                    ip_value = attn.head_to_batch_dim(ip_value)
                    ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
                    current_ip_hidden_states = paddle.bmm(ip_attention_probs, ip_value)
                    current_ip_hidden_states = attn.batch_to_head_dim(
                        current_ip_hidden_states
                    )
                    hidden_states = hidden_states + scale * current_ip_hidden_states
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


class IPAdapterAttnProcessor2_0(paddle.nn.Layer):
    """
    Attention processor for IP-Adapter for PyTorch 2.0.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or `List[float]`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(
        self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0
    ):
        super().__init__()
>>>>>>        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens
        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError(
                "`scale` should be a list of integers with the same length as `num_tokens`."
            )
        self.scale = scale
        self.to_k_ip = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Linear(
                    in_features=cross_attention_dim,
                    out_features=hidden_size,
                    bias_attr=False,
                )
                for _ in range(len(num_tokens))
            ]
        )
        self.to_v_ip = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Linear(
                    in_features=cross_attention_dim,
                    out_features=hidden_size,
                    bias_attr=False,
                )
                for _ in range(len(num_tokens))
            ]
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[paddle.Tensor] = None,
    ):
        residual = hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release. Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                deprecate(
                    "encoder_hidden_states not a tuple",
                    "1.0.0",
                    deprecation_message,
                    standard_warn=False,
                )
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )
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
                batch_size, attn.heads, -1, attention_mask.shape[-1]
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
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = paddle.nn.functional.scaled_dot_product_attention(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose([0, 2, 1, 3])
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match length of self.scale array ({len(self.scale)}) and number of ip_hidden_states ({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(
                    zip(ip_adapter_masks, self.scale, ip_hidden_states)
                ):
                    if not isinstance(mask, paddle.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape [1, num_images_for_ip_adapter, height, width]. Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]
                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])
                        ip_key = ip_key.view(
                            batch_size, -1, attn.heads, head_dim
                        ).transpose(1, 2)
                        ip_value = ip_value.view(
                            batch_size, -1, attn.heads, head_dim
                        ).transpose(1, 2)
                        _current_ip_hidden_states = (
                            paddle.nn.functional.scaled_dot_product_attention(
                                query.transpose([0, 2, 1, 3]),
                                ip_key.transpose([0, 2, 1, 3]),
                                ip_value.transpose([0, 2, 1, 3]),
                                attn_mask=None,
                                dropout_p=0.0,
                                is_causal=False,
                            ).transpose([0, 2, 1, 3])
                        )
                        _current_ip_hidden_states = _current_ip_hidden_states.transpose(
                            1, 2
                        ).reshape(batch_size, -1, attn.heads * head_dim)
                        _current_ip_hidden_states = _current_ip_hidden_states.to(
                            query.dtype
                        )
                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )
                        mask_downsample = mask_downsample.to(
                            dtype=query.dtype, device=query.place
                        )
                        hidden_states = hidden_states + scale[i] * (
                            _current_ip_hidden_states * mask_downsample
                        )
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)
                    ip_key = ip_key.view(
                        batch_size, -1, attn.heads, head_dim
                    ).transpose(1, 2)
                    ip_value = ip_value.view(
                        batch_size, -1, attn.heads, head_dim
                    ).transpose(1, 2)
                    current_ip_hidden_states = (
                        paddle.nn.functional.scaled_dot_product_attention(
                            query.transpose([0, 2, 1, 3]),
                            ip_key.transpose([0, 2, 1, 3]),
                            ip_value.transpose([0, 2, 1, 3]),
                            attn_mask=None,
                            dropout_p=0.0,
                            is_causal=False,
                        ).transpose([0, 2, 1, 3])
                    )
                    current_ip_hidden_states = current_ip_hidden_states.transpose(
                        1, 2
                    ).reshape(batch_size, -1, attn.heads * head_dim)
                    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)
                    hidden_states = hidden_states + scale * current_ip_hidden_states
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


LORA_ATTENTION_PROCESSORS = (
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    LoRAAttnAddedKVProcessor,
)
ADDED_KV_ATTENTION_PROCESSORS = (
    AttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    XFormersAttnAddedKVProcessor,
    LoRAAttnAddedKVProcessor,
)
CROSS_ATTENTION_PROCESSORS = (
    AttnProcessor,
    AttnProcessor2_0,
    XFormersAttnProcessor,
    SlicedAttnProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)
AttentionProcessor = Union[
    AttnProcessor,
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    XFormersAttnProcessor,
    SlicedAttnProcessor,
    AttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    XFormersAttnAddedKVProcessor,
    CustomDiffusionAttnProcessor,
    CustomDiffusionXFormersAttnProcessor,
    CustomDiffusionAttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    LoRAAttnAddedKVProcessor,
]
