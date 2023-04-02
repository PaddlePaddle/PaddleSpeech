# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
from paddle import nn

__all__ = [
    "Conv1dCell",
    "Conv1dBatchNorm",
]


class Conv1dCell(nn.Conv1D):
    """A subclass of Conv1D layer, which can be used in an autoregressive
    decoder like an RNN cell. 
    
    When used in autoregressive decoding, it performs causal temporal
    convolution incrementally. At each time step, it takes a step input and
    returns a step output.
    
    Notes
    ------
    It is done by caching an internal buffer of length ``receptive_file - 1``.
    when adding a step input, the buffer is shited by one step, the latest
    input is added to be buffer and the oldest step is discarded. And it
    returns a step output. For single step case, convolution is equivalent to a
    linear transformation.
    That it can be used as a cell depends on several restrictions:
    1. stride must be 1;
    2. padding must be a causal padding (recpetive_field - 1, 0).
    Thus, these arguments are removed from the ``__init__`` method of this
    class.

    Args:
        in_channels (int): 
            The feature size of the input.
        out_channels (int): 
            The feature size of the output.
        kernel_size (int or Tuple[int]): 
            The size of the kernel.
        dilation (int or Tuple[int]): 
            The dilation of the convolution, by default 1
        weight_attr (ParamAttr, Initializer, str or bool, optional): 
            The parameter attribute of the convolution kernel, 
            by default None.
        bias_attr (ParamAttr, Initializer, str or bool, optional):
            The parameter attribute of the bias. 
            If ``False``, this layer does not have a bias, by default None.
            
    Examples: 
        >>> cell = Conv1dCell(3, 4, kernel_size=5)
        >>> inputs = [paddle.randn([4, 3]) for _ in range(16)]
        >>> outputs = []
        >>> cell.eval()
        >>> cell.start_sequence()
        >>> for xt in inputs:
        >>>     outputs.append(cell.add_input(xt))
        >>> len(outputs))
        16
        >>> outputs[0].shape
        [4, 4]
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 weight_attr=None,
                 bias_attr=None):
        _dilation = dilation[0] if isinstance(dilation,
                                              (tuple, list)) else dilation
        _kernel_size = kernel_size[0] if isinstance(kernel_size, (
            tuple, list)) else kernel_size
        self._r = 1 + (_kernel_size - 1) * _dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=(self._r - 1, 0),
            dilation=dilation,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format="NCL")

    @property
    def receptive_field(self):
        """The receptive field of the Conv1dCell.
        """
        return self._r

    def start_sequence(self):
        """Prepare the layer for a series of incremental forward.
        
        Warnings:
            This method should be called before a sequence of calls to
            ``add_input``.

        Raises:
            Exception
                If this method is called when the layer is in training mode.
        """
        if self.training:
            raise Exception("only use start_sequence in evaluation")
        self._buffer = None

        # NOTE: call self's weight norm hook expliccitly since self.weight
        # is visited directly in this method without calling self.__call__
        # method. If we do not trigger the weight norm hook, the weight
        # may be outdated. e.g. after loading from a saved checkpoint
        # see also: https://github.com/pytorch/pytorch/issues/47588
        for hook in self._forward_pre_hooks.values():
            hook(self, None)
        self._reshaped_weight = paddle.reshape(self.weight,
                                               (self._out_channels, -1))

    def initialize_buffer(self, x_t):
        """Initialize the buffer for the step input.

        Args:
            x_t (Tensor): 
                The step input. shape=(batch_size, in_channels)
            
        """
        batch_size, _ = x_t.shape
        self._buffer = paddle.zeros(
            (batch_size, self._in_channels, self.receptive_field),
            dtype=x_t.dtype)

    def update_buffer(self, x_t):
        """Shift the buffer by one step.

        Args:
            x_t (Tensor): T
                he step input. shape=(batch_size, in_channels)
            
        """
        self._buffer = paddle.concat(
            [self._buffer[:, :, 1:], paddle.unsqueeze(x_t, -1)], -1)

    def add_input(self, x_t):
        """Add step input and compute step output.

        Args:
            x_t (Tensor): 
                The step input. shape=(batch_size, in_channels)
          
        Returns: 
            y_t (Tensor): 
                The step output. shape=(batch_size, out_channels)

        """
        batch_size = x_t.shape[0]
        if self.receptive_field > 1:
            if self._buffer is None:
                self.initialize_buffer(x_t)

            # update buffer
            self.update_buffer(x_t)
            if self._dilation[0] > 1:
                input = self._buffer[:, :, ::self._dilation[0]]
            else:
                input = self._buffer
            input = paddle.reshape(input, (batch_size, -1))
        else:
            input = x_t
        y_t = paddle.matmul(input, self._reshaped_weight, transpose_y=True)
        y_t = y_t + self.bias
        return y_t


class Conv1dBatchNorm(nn.Layer):
    """A Conv1D Layer followed by a BatchNorm1D.

    Args:
        in_channels (int): 
            The feature size of the input.
        out_channels (int): 
            The feature size of the output.
        kernel_size (int): 
            The size of the convolution kernel.
        stride (int, optional): 
            The stride of the convolution, by default 1.
        padding (int, str or Tuple[int], optional):
            The padding of the convolution.
            If int, a symmetrical padding is applied before convolution;
            If str, it should be "same" or "valid";
            If Tuple[int], its length should be 2, meaning
            ``(pad_before, pad_after)``, by default 0.
        weight_attr (ParamAttr, Initializer, str or bool, optional):
            The parameter attribute of the convolution kernel,
            by default None.
        bias_attr (ParamAttr, Initializer, str or bool, optional):
            The parameter attribute of the bias of the convolution,
            by defaultNone.
        data_format (str ["NCL" or "NLC"], optional): 
            The data layout of the input, by default "NCL"
        momentum (float, optional): 
            The momentum of the BatchNorm1D layer, by default 0.9
        epsilon (float, optional): 
            The epsilon of the BatchNorm1D layer, by default 1e-05
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCL",
                 momentum=0.9,
                 epsilon=1e-05):
        super().__init__()
        self.conv = nn.Conv1D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)
        self.bn = nn.BatchNorm1D(
            out_channels,
            momentum=momentum,
            epsilon=epsilon,
            data_format=data_format)

    def forward(self, x):
        """Forward pass of the Conv1dBatchNorm layer.
        
        Args:
            x (Tensor): 
                The input tensor. Its data layout depends on ``data_format``. 
                shape=(B, C_in, T_in) or (B, T_in, C_in)
    
        Returns:
            Tensor: 
                The output tensor. shape=(B, C_out, T_out) or (B, T_out, C_out)
                
        """
        x = self.conv(x)
        x = self.bn(x)
        return x
