"""Library implementing linear transformation.
Authors
 * Mirco Ravanelli 2020
 * Davide Borra 2021
"""
import logging

import paddle

from paddlespeech.s2t.modules import align

logger = logging.getLogger(__name__)


class Linear(paddle.nn.Layer):
    """Computes a linear transformation y = wx + b.
    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple
        It is the shape of the input tensor.
    input_size: int
        Size of the input tensor.
    bias : bool
        If True, the additive bias b is adopted.
    combine_dims : bool
        If True and the input is 4D, combine 3rd and 4th dimensions of input.
    Example
    -------
    >>> inputs = paddle.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    paddle.shape([10, 50, 100])
    """

    def __init__(
            self,
            n_neurons,
            input_shape=None,
            input_size=None,
            bias_attr=None,
            combine_dims=False, ):
        super().__init__()
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        # Weights are initialized following paddle approach
        self.w = align.Linear(input_size, n_neurons, bias_attr=bias_attr)

    def forward(self, x):
        """Returns the linear transformation of input tensor.
        Arguments
        ---------
        x : paddle.Tensor
            Input to transform linearly.
        """
        if x.rank == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        wx = self.w(x)

        return wx
