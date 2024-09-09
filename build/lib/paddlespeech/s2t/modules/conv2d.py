from typing import Optional
from typing import Union

import paddle
import paddle.nn.functional as F
from paddle.nn.layer.conv import _ConvNd

__all__ = ['Conv2DValid']


class Conv2DValid(_ConvNd):
    """
    Conv2d operator for VALID mode padding.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int=1,
                 padding: Union[str, int]=0,
                 dilation: int=1,
                 groups: int=1,
                 padding_mode: str='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW",
                 valid_trigx: bool=False,
                 valid_trigy: bool=False) -> None:
        super(Conv2DValid, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            False,
            2,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)
        self.valid_trigx = valid_trigx
        self.valid_trigy = valid_trigy

    def _conv_forward(self,
                      input: paddle.Tensor,
                      weight: paddle.Tensor,
                      bias: Optional[paddle.Tensor]):
        validx, validy = 0, 0
        if self.valid_trigx:
            validx = (input.shape[-2] *
                      (self._stride[-2] - 1) - 1 + self._kernel_size[-2]) // 2
        if self.valid_trigy:
            validy = (input.shape[-1] *
                      (self._stride[-1] - 1) - 1 + self._kernel_size[-1]) // 2
        return F.conv2d(input, weight, bias, self._stride, (validx, validy),
                        self._dilation, self._groups)

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return self._conv_forward(input, self.weight, self.bias)
