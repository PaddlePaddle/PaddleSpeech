"""This library implements different operations needed by complex-
 valued architectures.
 This work is inspired by: "Deep Complex Networks" from Trabelsi C.
 et al.

Authors
 * Titouan Parcollet 2020
"""

import paddle
import paddle.nn.functional as F
import numpy as np


def check_complex_input(input_shape):
    """Check the complex-valued shape for a linear layer.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    """
    if len(input_shape) not in {2, 3}:
        raise Exception(
            "Complex linear accepts only input of dimension 2 or 3."
            " input.dim = " + str(input.dim())
        )

    nb_hidden = input_shape[-1]

    if nb_hidden % 1 != 0:
        raise Exception(
            "Complex Tensors must have an even number of hidden dimensions."
            " input.size()[1] = " + str(nb_hidden)
        )


def get_real(input, input_type="linear", channels_axis=1):
    """Returns the real components of the complex-valued input.

    Arguments
    ---------
    input : paddle.Tensor
        Input tensor.
    input_type : str,
        (convolution, linear) (default "linear")
    channels_axis : int.
        Default 1.
    """

    if input_type == "linear":
        nb_hidden = input.size()[-1]
        if input.dim() == 2:
            return input.narrow(
                1, 0, nb_hidden // 2
            )  # input[:, :nb_hidden / 2]
        elif input.dim() == 3:
            return input.narrow(
                2, 0, nb_hidden // 2
            )  # input[:, :, :nb_hidden / 2]
    else:
        nb_featmaps = input.size(channels_axis)
        return input.narrow(channels_axis, 0, nb_featmaps // 2)


def get_imag(input, input_type="linear", channels_axis=1):
    """Returns the imaginary components of the complex-valued input.

    Arguments
    ---------
    input : paddle.Tensor
        Input tensor.
    input_type : str,
        (convolution, linear) (default "linear")
    channels_axis : int.
        Default 1.
    """

    if input_type == "linear":
        nb_hidden = input.size()[-1]
        if input.dim() == 2:
            return input.narrow(
                1, nb_hidden // 2, nb_hidden // 2
            )  # input[:, :nb_hidden / 2]
        elif input.dim() == 3:
            return input.narrow(
                2, nb_hidden // 2, nb_hidden // 2
            )  # input[:, :, :nb_hidden / 2]
    else:
        nb_featmaps = input.size(channels_axis)
        return input.narrow(channels_axis, nb_featmaps // 2, nb_featmaps // 2)


def get_conjugate(input, input_type="linear", channels_axis=1):
    """Returns the conjugate (z = r - xi) of the input complex numbers.

    Arguments
    ---------
    input : paddle.Tensor
        Input tensor
    input_type : str,
        (convolution, linear) (default "linear")
    channels_axis : int.
        Default 1.
    """
    input_imag = get_imag(input, input_type, channels_axis)
    input_real = get_real(input, input_type, channels_axis)
    if input_type == "linear":
        return torch.cat([input_real, -input_imag], dim=-1)
    elif input_type == "convolution":
        return torch.cat([input_real, -input_imag], dim=channels_axis)


def complex_linear_op(input, real_weight, imag_weight, bias):
    """
    Applies a complex linear transformation to the incoming data.

    Arguments
    ---------
    input : paddle.Tensor
        Complex input tensor to be transformed.
    real_weight : torch.Parameter
        Real part of the quaternion weight matrix of this layer.
    imag_weight : torch.Parameter
        First imaginary part of the quaternion weight matrix of this layer.
    bias : torch.Parameter
    """

    cat_real = torch.cat([real_weight, -imag_weight], dim=0)
    cat_imag = torch.cat([imag_weight, real_weight], dim=0)
    cat_complex = torch.cat([cat_real, cat_imag], dim=1)

    # If the input is already [batch*time, N]
    if input.dim() == 2:
        if bias.requires_grad:
            return torch.addmm(bias, input, cat_complex)
        else:
            return torch.mm(input, cat_complex)
    else:
        output = torch.matmul(input, cat_complex)
        if bias.requires_grad:
            return output + bias
        else:
            return output


def complex_conv_op(
    input, real_weight, imag_weight, bias, stride, padding, dilation, conv1d
):
    """Applies a complex convolution to the incoming data.

    Arguments
    ---------
    input : paddle.Tensor
        Complex input tensor to be transformed.
    conv1d : bool
        If true, a 1D convolution operation will be applied. Otherwise, a 2D
        convolution is called.
    real_weight : torch.Parameter
        Real part of the quaternion weight matrix of this layer.
    imag_weight : torch.Parameter
        First imaginary part of the quaternion weight matrix of this layer.
    bias : torch.Parameter
    stride : int
        Stride factor of the convolutional filters.
    padding : int
        Amount of padding. See torch.nn documentation for more information.
    dilation : int
        Dilation factor of the convolutional filters.
    """
    cat_real = torch.cat([real_weight, -imag_weight], dim=1)
    cat_imag = torch.cat([imag_weight, real_weight], dim=1)
    cat_complex = torch.cat([cat_real, cat_imag], dim=0)

    if conv1d:
        convfunc = F.conv1d
    else:
        convfunc = F.conv2d

    return convfunc(input, cat_complex, bias, stride, padding, dilation)


def unitary_init(
    in_features, out_features, kernel_size=None, criterion="glorot"
):
    """ Returns a matrice of unitary complex numbers.

    Arguments
    ---------
    in_features : int
        Number of real values of the input layer (quaternion // 4).
    out_features : int
        Number of real values of the output layer (quaternion // 4).
    kernel_size : int
        Kernel_size for convolutional layers (ex: (3,3)).
    criterion : str
        (glorot, he) (default "glorot").
    """

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)

    # Unitary complex
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i] ** 2 + v_i[i] ** 2) + 0.0001
        v_r[i] /= norm
        v_i[i] /= norm

    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)

    return (v_r, v_i)


def complex_init(
    in_features, out_features, kernel_size=None, criterion="glorot"
):
    """ Returns a matrice of complex numbers initialized as described in:
    "Deep Complex Networks", Trabelsi C. et al.

    Arguments
    ---------
    in_features : int
        Number of real values of the input layer (quaternion // 4).
    out_features : int
        Number of real values of the output layer (quaternion // 4).
    kernel_size : int
        Kernel_size for convolutional layers (ex: (3,3)).
    criterion: str
        (glorot, he) (default "glorot")
    """

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_out = out_features * receptive_field
        fan_in = in_features * receptive_field
    else:
        fan_out = out_features
        fan_in = in_features
    if criterion == "glorot":
        s = 1.0 / (fan_in + fan_out)
    else:
        s = 1.0 / fan_in

    if kernel_size is None:
        size = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            size = (out_features, in_features) + tuple((kernel_size,))
        else:
            size = (out_features, in_features) + (*kernel_size,)

    modulus = np.random.rayleigh(scale=s, size=size)
    phase = np.random.uniform(-np.pi, np.pi, size)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)

    return (weight_real, weight_imag)


def affect_init(real_weight, imag_weight, init_func, criterion):
    """ Applies the weight initialization function given to the parameters.

    Arguments
    ---------
    real_weight: torch.Parameters
    imag_weight: torch.Parameters
    init_func: function
        (unitary_init, complex_init)
    criterion: str
        (glorot, he)
    """
    a, b = init_func(real_weight.size(0), real_weight.size(1), None, criterion)
    a, b = torch.from_numpy(a), torch.from_numpy(b)
    real_weight.data = a.type_as(real_weight.data)
    imag_weight.data = b.type_as(imag_weight.data)


def affect_conv_init(
    real_weight, imag_weight, kernel_size, init_func, criterion
):
    """ Applies the weight initialization function given to the parameters.
    This is specifically written for convolutional layers.

    Arguments
    ---------
    real_weight: torch.Parameters
    imag_weight: torch.Parameters
    kernel_size: int
    init_func: function
        (unitary_init, complex_init)
    criterion: str
        (glorot, he)
    """
    in_channels = real_weight.size(1)
    out_channels = real_weight.size(0)
    a, b = init_func(
        in_channels, out_channels, kernel_size=kernel_size, criterion=criterion,
    )
    a, b = torch.from_numpy(a), torch.from_numpy(b)
    real_weight.data = a.type_as(real_weight.data)
    imag_weight.data = b.type_as(imag_weight.data)


# The following mean function using a list of reduced axes is taken from:
# https://discuss.pytorch.org/t/sum-mul-over-multiple-axes/1882/8
def multi_mean(input, axes, keepdim=False):
    """
    Performs `torch.mean` over multiple dimensions of `input`.
    """
    axes = sorted(axes)
    m = input
    for axis in reversed(axes):
        m = m.mean(axis, keepdim)
    return m
