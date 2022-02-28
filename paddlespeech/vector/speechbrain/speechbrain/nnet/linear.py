"""Library implementing linear transformation.

Authors
 * Mirco Ravanelli 2020
 * Davide Borra 2021
"""

import paddle
import logging
import paddle.nn as nn

logger = logging.getLogger(__name__)


class Linear(nn.Layer):
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
    >>> inputs = torch.rand(10, 50, 40)
    >>> lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    >>> output = lin_t(inputs)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        bias=True,
        combine_dims=False,
    ):
        super().__init__()
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        # Weights are initialized following pytorch approach
        self.w = nn.Linear(input_size, n_neurons, bias_attr=bias)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : paddle.Tensor
            Input to transform linearly.
        """
        if x.ndim == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        wx = self.w(x)

        return wx


class LinearWithConstraint(Linear):
    """Computes a linear transformation y = wx + b with kernel max-norm constaint.
    This corresponds to set an upper bound for the kernel norm.

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
    max_norm : float
        Kernel max-norm

    Example
    -------
    >>> inputs = torch.rand(100,)
    >>> max_norm = 1.
    >>> lin_t_contrained = LinearWithConstraint(input_size=inputs.shape[0], n_neurons=2, max_norm=max_norm)
    >>> output = lin_t_contrained(inputs)
    >>> torch.any(torch.norm(lin_t_contrained.w.weight.data, p=2, dim=0)>max_norm)
    tensor(False)
    """

    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : paddle.Tensor
            Input to transform linearly.
        """
        self.w.weight.data = paddle.renorm(
            self.w.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)
