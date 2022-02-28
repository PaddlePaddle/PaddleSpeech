"""Library implementing quaternion-valued convolutional neural networks.

Authors
 * Titouan Parcollet 2020
"""

import paddle
import paddle.nn as nn
import logging
import paddle.nn.functional as F
from speechbrain.nnet.CNN import get_padding_elem
from speechbrain.nnet.quaternion_networks.q_ops import (
    unitary_init,
    quaternion_init,
    affect_conv_init,
    quaternion_conv_op,
    quaternion_conv_rotation_op,
)
from typing import Tuple

logger = logging.getLogger(__name__)


class QConv1d(paddle.nn.Layer):
    """This function implements quaternion-valued 1d convolution.

    Arguments
    ---------
    input_shape : tuple
        The shape of the input.
    out_channels : int
        Number of output channels. Please note
        that these are quaternion-valued neurons. If 256
        channels are specified, the output dimension
        will be 1024.
    kernel_size : int
        Kernel size of the convolutional filters.
    stride : int, optional
        Stride factor of the convolutional filters (default 1).
    dilation : int, optional
        Dilation factor of the convolutional filters (default 1).
    padding : str, optional
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is same as input shape.
        "causal" results in causal (dilated) convolutions (default "same").
    padding_mode : str, optional
        This flag specifies the type of padding. See torch.nn documentation
        for more information (default "reflect").
    groups : int, optional
        Default: 1
        This option specifies the convolutional groups. See torch.nn
        documentation for more information (default 1).
    bias : bool, optional
        If True, the additive bias b is adopted (default True).
    init_criterion : str , optional
        (glorot, he).
        This parameter controls the initialization criterion of the weights.
        It is combined with weights_init to build the initialization method of
        the quaternion-valued weights (default "glorot").
    weight_init : str, optional
        (quaternion, unitary).
        This parameter defines the initialization procedure of the
        quaternion-valued weights. "quaternion" will generate random quaternion
        weights following the init_criterion and the quaternion polar form.
        "unitary" will normalize the weights to lie on the unit circle (default "quaternion").
        More details in: "Quaternion Recurrent Neural Networks",
        Parcollet T. et al.
    spinor : bool, optional
        When True, the layer will be turned into a spinor layer. More precisely
        W*x will be turned into W*x*W-1. The input x will be rotated by W such
        as in a spinor neural network. However, x MUST be a quaternion with
        the real part equal to zero. (0 + xi + yj + zk). Indeed, the rotation
        operation only acts on the vector part. Note that W will always be
        normalized before the rotation to ensure the quaternion algebra (default False).
        More details in: "Quaternion neural networks", Parcollet T.
    vector_scale : bool, optional
        The vector_scale is only used when spinor = True. In the context of a
        spinor neural network, multiple rotations of the input vector x are
        performed and summed. Hence, the norm of the output vector always
        increases with the number of layers, making the neural network instable
        with deep configurations. The vector_scale parameters are learnable
        parameters that acts like gates by multiplying the output vector with
        a small trainable parameter (default False).

    Example
    -------
    >>> inp_tensor = torch.rand([10, 16, 40])
    >>> cnn_1d = QConv1d(
    ...     input_shape=inp_tensor.shape, out_channels=12, kernel_size=3
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 16, 48])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        init_criterion="glorot",
        weight_init="quaternion",
        spinor=False,
        vector_scale=False,
    ):
        super().__init__()
        self.input_shape = input_shape
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
        self.spinor = spinor
        self.vector_scale = vector_scale

        self.in_channels = self._check_input(input_shape) // 4

        # Managing the weight initialization and bias by directly setting the
        # correct function

        (self.k_shape, self.w_shape) = self._get_kernel_and_weight_shape()

        self.r_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))
        self.i_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))
        self.j_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))
        self.k_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))

        # Spinor specific parameters
        if self.spinor:
            self.zero_kernel = torch.nn.Parameter(
                torch.zeros(self.r_weight.shape), requires_grad=False
            )
        else:
            self.zero_kernel = paddle.Tensor(self.r_weight.shape).requires_grad_(
                False
            )

        if self.spinor and self.vector_scale:
            self.scale_param = torch.nn.Parameter(
                paddle.Tensor(self.r_weight.shape)
            )
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        else:
            self.scale_param = paddle.Tensor(self.r_weight.shape).requires_grad_(
                False
            )

        if self.bias:
            self.b = torch.nn.Parameter(paddle.Tensor(4 * self.out_channels))
            self.b.data.fill_(0)
        else:
            self.b = paddle.Tensor(4 * self.out_channels).requires_grad_(False)

        self.winit = {"quaternion": quaternion_init, "unitary": unitary_init}[
            self.weight_init
        ]

        # Initialise the weights
        affect_conv_init(
            self.r_weight,
            self.i_weight,
            self.j_weight,
            self.k_weight,
            self.kernel_size,
            self.winit,
            self.init_criterion,
        )

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : paddle.Tensor (batch, time, channel)
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
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding
            )

        if self.spinor:
            out = quaternion_conv_rotation_op(
                x,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.b,
                scale=self.scale_param,
                zero_kernel=self.zero_kernel,
                stride=self.stride,
                dilation=self.dilation,
                padding=0,  # already managed
                groups=self.groups,
                conv1d=True,
            )
        else:
            out = quaternion_conv_op(
                x,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.b,
                stride=self.stride,
                dilation=self.dilation,
                padding=0,  # already managed
                groups=self.groups,
                conv1d=True,
            )

        out = out.transpose(1, -1)
        return out

    def _get_kernel_and_weight_shape(self):
        """ Returns the kernel size and weight shape for convolutional layers.
        """
        ks = self.kernel_size
        w_shape = (self.out_channels, self.in_channels) + tuple((ks,))
        return ks, w_shape

    def _manage_padding(
        self, x, kernel_size: int, dilation: int, stride: int,
    ):
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
        stride: int
            Stride.
        """

        # Detecting input shape
        L_in = x.shape[-1]

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)

        return x

    def _check_input(self, input_shape):
        """Checks the input and returns the number of input channels.
        """

        if len(input_shape) == 3:
            in_channels = input_shape[2]
        else:
            raise ValueError(
                "QuaternionConv1d expects 3d inputs. Got " + str(input_shape)
            )

        # Kernel size must be odd
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got "
                + str(self.kernel_size)
            )

        # Check quaternion format
        if in_channels % 4 != 0:
            raise ValueError(
                "Quaternion Tensors must have dimensions divisible by 4."
                " input.size()[3] = " + str(in_channels)
            )

        return in_channels


class QConv2d(paddle.nn.Layer):
    """This function implements quaternion-valued 1d convolution.

    Arguments
    ---------
    input_shape : tuple
        The shape of the input.
    out_channels : int
        Number of output channels. Please note
        that these are quaternion-valued neurons. If 256
        channels are specified, the output dimension
        will be 1024.
    kernel_size : int
        Kernel size of the convolutional filters.
    stride : int, optional
        Stride factor of the convolutional filters (default 1).
    dilation : int, optional
        Dilation factor of the convolutional filters (default 1).
    padding : str, optional
        (same, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is same as input shape (default "same").
    padding_mode : str, optional
        This flag specifies the type of padding. See torch.nn documentation
        for more information. (default "reflect")
    groups : int, optional
        This option specifies the convolutional groups. See torch.nn
        documentation for more information. (default 1).
    bias : bool, optional
        If True, the additive bias b is adopted (default True).
    init_criterion : str , optional
        (glorot, he).
        This parameter controls the initialization criterion of the weights.
        It is combined with weights_init to build the initialization method of
        the quaternion-valued weights (default "glorot").
    weight_init : str, optional
        (quaternion, unitary).
        This parameter defines the initialization procedure of the
        quaternion-valued weights. "quaternion" will generate random quaternion
        weights following the init_criterion and the quaternion polar form.
        "unitary" will normalize the weights to lie on the unit circle (default "quaternion").
        More details in: "Quaternion Recurrent Neural Networks",
        Parcollet T. et al.
    spinor : bool, optional
        When True, the layer will be turned into a spinor layer. More precisely
        W*x will be turned into W*x*W-1. The input x will be rotated by W such
        as in a spinor neural network. However, x MUST be a quaternion with
        the real part equal to zero. (0 + xi + yj + zk). Indeed, the rotation
        operation only acts on the vector part. Note that W will always be
        normalized before the rotation to ensure the quaternion algebra (default False).
        More details in: "Quaternion neural networks", Parcollet T.
    vector_scale : bool, optional
        The vector_scale is only used when spinor = True. In the context of a
        spinor neural network, multiple rotations of the input vector x are
        performed and summed. Hence, the norm of the output vector always
        increases with the number of layers, making the neural network instable
        with deep configurations. The vector_scale parameters are learnable
        parameters that acts like gates by multiplying the output vector with
        a small trainable parameter (default False).

    Example
    -------
    >>> inp_tensor = torch.rand([10, 4, 16, 40])
    >>> cnn_1d = QConv2d(
    ...     input_shape=inp_tensor.shape, out_channels=12, kernel_size=3
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 4, 16, 48])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        init_criterion="glorot",
        weight_init="quaternion",
        spinor=False,
        vector_scale=False,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.spinor = spinor
        self.vector_scale = vector_scale

        # handle the case if some parameters are int
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)

        self.in_channels = self._check_input(input_shape) // 4

        # Managing the weight initialization and bias by directly setting the
        # correct function

        (self.k_shape, self.w_shape) = self._get_kernel_and_weight_shape()

        self.r_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))
        self.i_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))
        self.j_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))
        self.k_weight = torch.nn.Parameter(paddle.Tensor(*self.w_shape))

        # Spinor specific parameters
        if self.spinor:
            self.zero_kernel = torch.nn.Parameter(
                torch.zeros(self.r_weight.shape), requires_grad=False
            )
        else:
            self.zero_kernel = paddle.Tensor(self.r_weight.shape).requires_grad_(
                False
            )

        if self.spinor and self.vector_scale:
            self.scale_param = torch.nn.Parameter(
                paddle.Tensor(self.r_weight.shape)
            )
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        else:
            self.scale_param = paddle.Tensor(self.r_weight.shape).requires_grad_(
                False
            )

        if self.bias:
            self.b = torch.nn.Parameter(paddle.Tensor(4 * self.out_channels))
            self.b.data.fill_(0)
        else:
            self.b = paddle.Tensor(4 * self.out_channels).requires_grad_(False)

        self.winit = {"quaternion": quaternion_init, "unitary": unitary_init}[
            self.weight_init
        ]

        # Initialise the weights
        affect_conv_init(
            self.r_weight,
            self.i_weight,
            self.j_weight,
            self.k_weight,
            self.kernel_size,
            self.winit,
            self.init_criterion,
        )

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : paddle.Tensor (batch, time, channel)
            Input to convolve. 3d or 4d tensors are expected.
        """

        # (batch, channel, time)
        x = x.transpose(1, -1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding
            )

        if self.spinor:
            out = quaternion_conv_rotation_op(
                x,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.b,
                scale=self.scale_param,
                zero_kernel=self.zero_kernel,
                stride=self.stride[0],
                dilation=self.dilation[0],
                padding=0,  # already managed
                groups=self.groups,
                conv1d=True,
            )
        else:
            out = quaternion_conv_op(
                x,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.b,
                stride=self.stride[0],
                dilation=self.dilation[0],
                padding=0,  # already managed
                groups=self.groups,
                conv1d=False,
            )

        out = out.transpose(1, -1)
        return out

    def _check_input(self, input_shape):
        """Checks the input and returns the number of input channels.
        """

        if len(input_shape) == 4:
            in_channels = input_shape[-1]
        else:
            raise ValueError(
                "QuaternionConv1d expects 4d inputs. Got " + str(input_shape)
            )

        # Kernel size must be divisible by 4.
        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got "
                + str(self.kernel_size)
            )

        # Check quaternion format
        if in_channels % 4 != 0:
            raise ValueError(
                "Quaternion Tensors must have dimensions divisible by 4."
                " input.size()[" + str(-1) + "] = " + str(in_channels)
            )

        return in_channels

    def _get_kernel_and_weight_shape(self):
        """ Returns the kernel size and weight shape for convolutional layers.
        """

        ks = (self.kernel_size[0], self.kernel_size[1])
        w_shape = (self.out_channels, self.in_channels) + (*ks,)
        return ks, w_shape

    def _manage_padding(
        self,
        x,
        kernel_size: Tuple[int, int],
        dilation: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        """This function performs zero-padding on the time and frequency axises
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
        x = nn.functional.pad(x, padding, mode=self.padding_mode)

        return x
