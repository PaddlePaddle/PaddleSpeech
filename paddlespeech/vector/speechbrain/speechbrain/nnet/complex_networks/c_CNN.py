"""Library implementing complex-valued convolutional neural networks.

Authors
 * Titouan Parcollet 2020
"""
import paddle
import paddle.nn as nn
import logging
import paddle.nn.functional as F
from speechbrain.nnet.CNN import get_padding_elem
from speechbrain.nnet.complex_networks.c_ops import (
    unitary_init,
    complex_init,
    affect_conv_init,
    complex_conv_op,
)

logger = logging.getLogger(__name__)


class CConv1d(paddle.nn.Layer):
    """This function implements complex-valued 1d convolution.

    Arguments
    ---------
    out_channels : int
        Number of output channels. Please note
        that these are complex-valued neurons. If 256
        channels are specified, the output dimension
        will be 512.
    kernel_size : int
        Kernel size of the convolutional filters.
    stride : int, optional
        Stride factor of the convolutional filters (default 1).
    dilation : int, optional
        Dilation factor of the convolutional filters (default 1).
    padding : str, optional
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is same as input shape.
        "causal" results in causal (dilated) convolutions. (default "same")
    padding_mode : str, optional
        This flag specifies the type of padding. See torch.nn documentation
        for more information (default "reflect").
    groups : int, optional
        This option specifies the convolutional groups. See torch.nn
        documentation for more information (default 1).
    bias : bool, optional
        If True, the additive bias b is adopted (default True).
    init_criterion : str, optional
        (glorot, he).
        This parameter controls the initialization criterion of the weights.
        It is combined with weights_init to build the initialization method of
        the complex-valued weights. (default "glorot")
    weight_init : str, optional
        (complex, unitary).
        This parameter defines the initialization procedure of the
        complex-valued weights. "complex" will generate random complex-valued
        weights following the init_criterion and the complex polar form.
        "unitary" will normalize the weights to lie on the unit circle. (default "complex")
        More details in: "Deep Complex Networks", Trabelsi C. et al.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 16, 30])
    >>> cnn_1d = CConv1d(
    ...     input_shape=inp_tensor.shape, out_channels=12, kernel_size=5
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 16, 24])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        init_criterion="glorot",
        weight_init="complex",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.init_criterion = init_criterion
        self.weight_init = weight_init

        self.in_channels = self._check_input(input_shape) // 2

        # Managing the weight initialization and bias by directly setting the
        # correct function

        (self.k_shape, self.w_shape) = self._get_kernel_and_weight_shape()

        self.real_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))
        self.imag_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))

        if self.bias:
            self.b = torch.nn.Parameter(paddle.Tensor(2 * self.out_channels))
            self.b.data.fill_(0)
        else:
            self.b = None

        self.winit = {"complex": complex_init, "unitary": unitary_init}[
            self.weight_init
        ]

        affect_conv_init(
            self.real_weight,
            self.imag_weight,
            self.kernel_size,
            self.winit,
            self.init_criterion,
        )

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : paddle.Tensor
            (batch, time, channel).
            Input to convolve. 3d or 4d tensors are expected.

        """
        # (batch, channel, time)
        x = x.transpose(1, -1)
        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got %s."
                % (self.padding)
            )

        wx = complex_conv_op(
            x,
            self.real_weight,
            self.imag_weight,
            self.b,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            conv1d=True,
        )

        wx = wx.transpose(1, -1)
        return wx

    def _manage_padding(self, x, kernel_size, dilation, stride):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : paddle.Tensor
            Input tensor.
        kernel_size : int
            Kernel size.
        dilation : int
            Dilation.
        stride : int
            Stride.
        """

        # Detecting input shape
        L_in = x.shape[-1]

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, tuple(padding), mode=self.padding_mode)

        return x

    def _check_input(self, input_shape):
        """Checks the input and returns the number of input channels.
        """

        if len(input_shape) == 3:
            in_channels = input_shape[2]
        else:
            raise ValueError(
                "ComplexConv1d expects 3d inputs. Got " + input_shape
            )

        # Kernel size must be odd
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        # Check complex format
        if in_channels % 2 != 0:
            raise ValueError(
                "Complex Tensors must have dimensions divisible by 2."
                " input.size()["
                + str(self.channels_axis)
                + "] = "
                + str(self.nb_channels)
            )

        return in_channels

    def _get_kernel_and_weight_shape(self):
        """ Returns the kernel size and weight shape for convolutional layers.
        """

        ks = self.kernel_size
        w_shape = (self.out_channels, self.in_channels) + tuple((ks,))
        return ks, w_shape


class CConv2d(nn.Layer):
    """This function implements complex-valued 1d convolution.

    Arguments
    ---------
    out_channels : int
        Number of output channels. Please note
        that these are complex-valued neurons. If 256
        channels are specified, the output dimension
        will be 512.
    kernel_size : int
        Kernel size of the convolutional filters.
    stride : int, optional
        Stride factor of the convolutional filters (default 1).
    dilation : int, optional
        Dilation factor of the convolutional filters (default 1).
    padding : str, optional
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is same as input shape.
        "causal" results in causal (dilated) convolutions. (default "same")
    padding_mode : str, optional
        This flag specifies the type of padding (default "reflect").
        See torch.nn documentation for more information.
    groups : int, optional
        This option specifies the convolutional groups (default 1). See torch.nn
        documentation for more information.
    bias : bool, optional
        If True, the additive bias b is adopted (default True).
    init_criterion : str , optional
        (glorot, he).
        This parameter controls the initialization criterion of the weights (default "glorot").
        It is combined with weights_init to build the initialization method of
        the complex-valued weights.
    weight_init : str, optional
        (complex, unitary).
        This parameter defines the initialization procedure of the
        complex-valued weights (default complex). "complex" will generate random complex-valued
        weights following the init_criterion and the complex polar form.
        "unitary" will normalize the weights to lie on the unit circle.
        More details in: "Deep Complex Networks", Trabelsi C. et al.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 16, 30, 30])
    >>> cnn_2d = CConv2d(
    ...     input_shape=inp_tensor.shape, out_channels=12, kernel_size=5
    ... )
    >>> out_tensor = cnn_2d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 16, 30, 24])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        init_criterion="glorot",
        weight_init="complex",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.init_criterion = init_criterion
        self.weight_init = weight_init

        # k -> [k,k]
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size, self.kernel_size]

        if isinstance(self.dilation, int):
            self.dilation = [self.dilation, self.dilation]

        if isinstance(self.stride, int):
            self.stride = [self.stride, self.stride]

        self.in_channels = self._check_input(input_shape) // 2

        # Managing the weight initialization and bias by directly setting the
        # correct function

        (self.k_shape, self.w_shape) = self._get_kernel_and_weight_shape()

        self.real_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))
        self.imag_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))

        if self.bias:
            self.b = torch.nn.Parameter(paddle.Tensor(2 * self.out_channels))
            self.b.data.fill_(0)
        else:
            self.b = None

        self.winit = {"complex": complex_init, "unitary": unitary_init}[
            self.weight_init
        ]

        affect_conv_init(
            self.real_weight,
            self.imag_weight,
            self.kernel_size,
            self.winit,
            self.init_criterion,
        )

    def forward(self, x, init_params=False):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : paddle.Tensor
            (batch, time, feature, channels).
            Input to convolve. 3d or 4d tensors are expected.
        """

        if init_params:
            self.init_params(x)

        # (batch, channel, feature, time)
        x = x.transpose(1, -1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got %s."
                % (self.padding)
            )

        wx = complex_conv_op(
            x,
            self.real_weight,
            self.imag_weight,
            self.b,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            conv1d=False,
        )

        wx = wx.transpose(1, -1)

        return wx

    def _get_kernel_and_weight_shape(self):
        """ Returns the kernel size and weight shape for convolutional layers.
        """

        ks = (self.kernel_size[0], self.kernel_size[1])
        w_shape = (self.out_channels, self.in_channels) + (*ks,)
        return ks, w_shape

    def _manage_padding(self, x, kernel_size, dilation, stride):
        """This function performs zero-padding on the time and frequency axes
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : paddle.Tensor
            Input tensor.
        kernel_size : int
            Kernel size.
        dilation : int
            Dilation.
        stride: int
            Stride.
        """
        # Detecting input shape
        L_in = x.shape[-1]

        # Time padding
        padding_time = get_padding_elem(
            L_in, stride[-1], kernel_size[-1], dilation[-1]
        )

        padding_freq = get_padding_elem(
            L_in, stride[-2], kernel_size[-2], dilation[-2]
        )
        padding = padding_time + padding_freq

        # Applying padding
        x = nn.functional.pad(x, tuple(padding), mode=self.padding_mode)

        return x

    def _check_input(self, input_shape):
        """Checks the input and returns the number of input channels.
        """
        if len(input_shape) == 3:
            self.unsqueeze = True
            in_channels = 1

        elif len(input_shape) == 4:
            in_channels = input_shape[3]

        else:
            raise ValueError("Expected 3d or 4d inputs. Got " + input_shape)

        # Kernel size must be odd
        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        # Check complex format
        if in_channels % 2 != 0:
            raise ValueError(
                "Complex Tensors must have dimensions divisible by 2."
                " input.size()["
                + str(self.channels_axis)
                + "] = "
                + str(self.nb_channels)
            )

        return in_channels
