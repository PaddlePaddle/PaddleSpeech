"""Library implementing dropout.

Authors
 * Mirco Ravanelli 2020
"""
import paddle  # noqa: F401
import logging
import paddle.nn as nn

logger = logging.getLogger(__name__)


class Dropout2d(nn.Layer):
    """This function implements dropout 2d. It randomly put zeros on
    entire channels.


    Arguments
    ---------
    dropout_rate : float
        It is the dropout factor (between 0 and 1).
    inplace : bool
        If True, it uses inplace operations.

    Example
    -------
    >>> drop = Dropout2d(drop_rate=0.5)
    >>> inputs = paddle.rand(10, 50, 40)
    >>> output=drop(inputs)
    >>> output.shape
    paddle.Size([10, 50, 40])
    """

    def __init__(
        self, drop_rate, inplace=False,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.inplace = inplace
        self.drop = nn.Dropout2D(p=self.drop_rate)

    def forward(self, x):
        """Applies dropout 2d to the input tensor.

        Arguments
        ---------
        x : paddle.Tensor (batch, time, channel1, channel2)
            input to normalize. 4d tensors are expected.
        """

        # time must be the last
        x = x.transpose(perm=[0, 2, 1])
        x_drop = self.drop(x)
        x_drop = x_drop.transpose(-1, 1).transpose(2, -1)

        return x_drop
