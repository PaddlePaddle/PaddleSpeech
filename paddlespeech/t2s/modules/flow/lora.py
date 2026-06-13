from typing import Optional, Tuple, Union

import paddle


class LoRALinearLayer(paddle.nn.Layer):
    """
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[paddle.CPUPlace, paddle.CUDAPlace, str]] = None,
        dtype: Optional[paddle.dtype] = None,
    ):
        super().__init__()
        self.down = paddle.nn.Linear(
            in_features=in_features, out_features=rank, bias_attr=False
        )
        self.up = paddle.nn.Linear(
            in_features=rank, out_features=out_features, bias_attr=False
        )
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features
        paddle.nn.init.normal_(self.down.weight, std=1 / rank)
        paddle.nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank
        return up_hidden_states.to(orig_dtype)



class LoRACompatibleLinear(paddle.nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return
        dtype, device = self.weight.data.dtype, self.weight.data.place
        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()
        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank
        fused_weight = (
            w_orig + lora_scale * paddle.bmm(w_up[None, :], w_down[None, :])[0]
        )
        if safe_fusing and paddle.isnan(fused_weight).any().item():
            raise ValueError(
                f"This LoRA weight seems to be broken. Encountered NaN values when trying to fuse LoRA weights for {self}.LoRA weights will not be fused."
            )
        self.weight.data = fused_weight.to(device=device, dtype=dtype)
        self.lora_layer = None
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (
            getattr(self, "w_up", None) is not None
            and getattr(self, "w_down", None) is not None
        ):
            return
        fused_weight = self.weight.data
        dtype, device = fused_weight.dtype, fused_weight.place
        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device).float()
        unfused_weight = (
            fused_weight.float()
            - self._lora_scale * paddle.bmm(w_up[None, :], w_down[None, :])[0]
        )
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)
        self.w_up = None
        self.w_down = None

    def forward(
        self, hidden_states: paddle.Tensor, scale: float = 1.0
    ) -> paddle.Tensor:
        if self.lora_layer is None:
            out = super().forward(hidden_states)
            return out
        else:
            out = super().forward(hidden_states) + scale * self.lora_layer(
                hidden_states
            )
            return out
