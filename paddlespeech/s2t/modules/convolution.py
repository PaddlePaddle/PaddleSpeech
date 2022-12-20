from typing import Tuple

import paddle
from paddle import nn
from paddle.nn import initializer as I
from typeguard import check_argument_types

__all__ = ['ConvolutionModule2']

from paddlespeech.s2t import masked_fill
from paddlespeech.s2t.modules.align import Conv1D, BatchNorm1D, LayerNorm


class ConvolutionModule2(nn.Layer):
    """ConvolutionModule in Conformer model."""

    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Layer = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True,
                 adaptive_scale: bool = False,
                 init_weights: bool = False):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        assert check_argument_types()
        super().__init__()
        self.bias = bias
        self.channels = channels
        self.kernel_size = kernel_size
        self.adaptive_scale = adaptive_scale
        ada_scale = self.create_parameter([1, 1, channels], default_initializer=I.Constant(1.0))
        self.add_parameter('ada_scale', ada_scale)
        ada_bias = self.create_parameter([1, 1, channels], default_initializer=I.Constant(0.0))
        self.add_parameter('ada_bias', ada_bias)

        self.pointwise_conv1 = Conv1D(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=None
            if bias else False,  # None for True, using bias as default config
        )

        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = Conv1D(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias_attr=None
            if bias else False,  # None for True, using bias as default config
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = BatchNorm1D(channels)
        else:
            self.use_layer_norm = True
            self.norm = LayerNorm(channels)

        self.pointwise_conv2 = Conv1D(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=None
            if bias else False,  # None for True, using bias as default config
        )
        self.activation = activation

        if init_weights:
            self.init_weights()

    def init_weights(self):
        pw_max = self.channels ** -0.5
        dw_max = self.kernel_size ** -0.5
        self.pointwise_conv1._param_attr = paddle.nn.initializer.Uniform(low=-pw_max, high=pw_max)
        if self.bias:
            self.pointwise_conv1._bias_attr = paddle.nn.initializer.Uniform(low=-pw_max, high=pw_max)
        self.depthwise_conv._param_attr = paddle.nn.initializer.Uniform(low=-dw_max, high=dw_max)
        if self.bias:
            self.depthwise_conv._bias_attr = paddle.nn.initializer.Uniform(low=-dw_max, high=dw_max)
        self.pointwise_conv2._param_attr = paddle.nn.initializer.Uniform(low=-pw_max, high=pw_max)
        if self.bias:
            self.pointwise_conv2._bias_attr = paddle.nn.initializer.Uniform(low=-pw_max, high=pw_max)

    def forward(
            self,
            x: paddle.Tensor,
            mask_pad: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool),
            cache: paddle.Tensor = paddle.zeros([0, 0, 0]),
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        if self.adaptive_scale:
            x = self.ada_scale * x + self.ada_bias

        # exchange the temporal dimension and the feature dimension
        x = x.transpose([0, 2, 1])  # [B, C, T]

        # mask batch padding
        if mask_pad.shape[2] > 0:  # time > 0
            x = masked_fill(x, mask_pad, 0.0)

        if self.lorder > 0:
            if cache.shape[2] == 0:  # cache_t == 0
                x = nn.functional.pad(x, [self.lorder, 0], 'constant', 0.0, data_format='NCL')
            else:
                assert cache.shape[0] == x.shape[0]  # B
                assert cache.shape[1] == x.shape[1]  # C
                x = paddle.concat((cache, x), axis=2)

            assert (x.shape[2] > self.lorder)
            new_cache = x[:, :, -self.lorder:]  # [B, C, T]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = paddle.zeros([0, 0, 0], dtype=x.dtype)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, axis=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose([0, 2, 1])  # [B, T, C]
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose([0, 2, 1])  # [B, C, T]
        x = self.pointwise_conv2(x)

        # mask batch padding
        if mask_pad.shape[2] > 0:  # time > 0
            x = masked_fill(x, mask_pad, 0.0)

        x = x.transpose([0, 2, 1])  # [B, T, C]
        return x, new_cache
