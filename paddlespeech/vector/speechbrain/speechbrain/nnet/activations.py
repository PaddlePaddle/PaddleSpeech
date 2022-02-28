"""Library implementing activation functions.

Authors
 * Mirco Ravanelli 2020
 * Jianyuan Zhong 2020
"""

import paddle
import logging
import paddle.nn.functional as F

logger = logging.getLogger(__name__)


class Softmax(paddle.nn.Layer):
    """Computes the softmax of a 2d, 3d, or 4d input tensor.

    Arguments
    ---------
    apply_log : bool
        Whether to apply the log function before softmax.
    dim : int
        If the dimension where softmax is applied.

    Example
    -------
    >>> classifier = Softmax()
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = classifier(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(self, apply_log=False, dim=-1):
        super().__init__()

        if apply_log:
            self.act = paddle.nn.LogSoftmax(axis=dim)
        else:
            self.act = paddle.nn.Softmax(axis=dim)

    def forward(self, x):
        """Returns the softmax of the input tensor.

        Arguments
        ---------
        x : paddle.Tensor
            Input tensor.
        """
        # Reshaping the tensors
        dims = x.shape

        if len(dims) == 3:
            x = x.reshape(dims[0] * dims[1], dims[2])

        if len(dims) == 4:
            x = x.reshape(dims[0] * dims[1], dims[2], dims[3])

        x_act = self.act(x)

        # Retrieving the original shape format
        if len(dims) == 3:
            x_act = x_act.reshape(dims[0], dims[1], dims[2])

        if len(dims) == 4:
            x_act = x_act.reshape(dims[0], dims[1], dims[2], dims[3])

        return x_act


class GumbelSoftmax(paddle.nn.Layer):
    """Samples from the Gumbel-Softmax distribution and optionally discretizes.

    Reference: https://arxiv.org/abs/1611.00712, https://arxiv.org/abs/1611.01144

    Arguments
    ----------
    tau: float
        non-negative scalar temperature
    hard: bool
        if True, the returned samples will be discretized as one-hot vectors, but will be differentiated as if it is the soft sample in autograd
    dim: int
        A dimension along which softmax will be computed (default: -1).

    Example
    -------
    >>> x = torch.randn((8, 40, 120))
    >>> act = GumbelSoftmax(0.8, True)
    >>> x = act(x)
    """

    def __init__(self, tau, hard=False, apply_log=False):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.apply_log = apply_log

    def forward(self, x):
        if self.apply_log:
            return torch.log(F.gumbel_softmax(x, tau=self.tau, hard=self.hard))
        return F.gumbel_softmax(x, tau=self.tau, hard=self.hard)


class Swish(paddle.nn.Layer):
    """ The class implements the Swish activation function from
    https://arxiv.org/pdf/2005.03191.pdf

    given input x. Swish(x) = x / (1 + exp(beta * x))

    Arguments
    ---------
    beta: float
        Beta value.

    Example
    -------
    >>> x = torch.randn((8, 40, 120))
    >>> act = Swish()
    >>> x = act(x)
    """

    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """Returns the Swished input tensor.

        Arguments
        ---------
        x : paddle.Tensor
            Input tensor.
        """
        return x * self.sigmoid(self.beta * x)
