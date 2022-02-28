"""Library implementing complex-valued linear transformation.

Authors
 * Titouan Parcollet 2020
"""

import paddle
import logging
from speechbrain.nnet.complex_networks.c_ops import (
    affect_init,
    complex_init,
    unitary_init,
    complex_linear_op,
    check_complex_input,
)

logger = logging.getLogger(__name__)


class CLinear(paddle.nn.Layer):
    """This function implements a fully connected complex-valued
    linear layer: y = Wx + b. y, W, x and b are thus complex
    numbers. A complex number is written as: r + xi. A tensor of
    complex numbers x = [batch, 32] can be understood as
    [batch, 0:15] = R and [batch, 16:31] = Xi. Thus the features
    dimension is cut in half (must be divisible by 2).

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output). Please note that these are complex-valued neurons. If 256
        neurons are specified, the output dimension will be 512.
    input_shape : tuple
        Expected size of the input.
    bias : bool
        if True, the additive bias b is adopted.
    init_criterion : str , optional
        (glorot, he).
        This parameter controls the initialization criterion of the weights.
        It is combined with weights_init to build the initialization method of
        the complex-valued weights (default "glorot").
    weight_init : str, optional
        (complex, unitary).
        This parameter defines the initialization procedure of the
        complex-valued weights (default "complex"). "complex" will generate random complex-valued
        weights following the init_criterion and the complex polar form.
        "unitary" will normalize the weights to lie on the unit circle.
        More details in: "Deep Complex Networks", Trabelsi C. et al.

    Example
    -------
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin = CLinear(n_neurons=100, input_shape=inputs.shape)
    >>> output = lin(inputs)
    >>> output.shape
    torch.Size([10, 50, 200])
    """

    def __init__(
        self,
        n_neurons,
        input_shape,
        bias=True,
        init_criterion="glorot",
        weight_init="complex",
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.bias = bias
        self.init_criterion = init_criterion
        self.weight_init = weight_init

        # When initialising with speechbrain the input_shape is an integer !
        # we need to transform it into a list it works with all the question ops
        if isinstance(input_shape, int):
            input_shape = [1, input_shape]

        # Check the complex_valued form of the input
        check_complex_input(input_shape)

        # Computing the complex dimensionality of the input
        self.in_features = input_shape[-1] // 2
        self.out_features = self.n_neurons

        # Two weight matrices are created for the real and imaginary parts of
        # the weights. This will also allow an easier complex product.
        self.real_weight = torch.nn.Parameter(
            paddle.Tensor(self.in_features, self.out_features)
        )
        self.imag_weight = torch.nn.Parameter(
            paddle.Tensor(self.in_features, self.out_features)
        )

        if self.bias:
            self.b = torch.nn.Parameter(paddle.Tensor(2 * self.out_features))
        else:
            self.b = paddle.Tensor(2 * self.out_features).requires_grad_(False)

        # Managing the weight initialization and bias
        self.winit = {"complex": complex_init, "unitary": unitary_init}[
            self.weight_init
        ]

        affect_init(
            self.real_weight, self.imag_weight, self.winit, init_criterion
        )

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : paddle.Tensor
            Input to transform linearly.
        """
        wx = complex_linear_op(x, self.real_weight, self.imag_weight, self.b)

        return wx
