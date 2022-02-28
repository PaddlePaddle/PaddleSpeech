"""Library implementing quaternion-valued normalization.

Authors
 * Titouan Parcollet 2020
"""

import paddle
from torch.nn import Parameter


class QBatchNorm(paddle.nn.Layer):
    """This class implements the simplest form of a quaternion batchnorm as
    described in : "Quaternion Convolutional Neural Network for
    Color Image Classification and Forensics", Qilin Y. et al.

    Arguments
    ---------
    input_size : int
        Expected size of the dimension to be normalized.
    dim : int, optional
        It defines the axis that should be normalized. It usually correspond to
        the channel dimension (default -1).
    gamma_init : float, optional
        First value of gamma to be used (mean) (default 1.0).
    beta_param : bool, optional
        When set to True the beta parameter of the BN is applied (default True).
    momentum : float, optional
        It defines the momentum as for the real-valued batch-normalization (default 0.1).
    eps : float, optional
        Term used to stabilize operation (default 1e-4).
    track_running_stats : bool, optional
        Equivalent to the real-valued batchnormalization parameter.
        When True, stats are tracked. When False, solely statistics computed
        over the batch are used (default True).


    Example
    -------
    >>> inp_tensor = torch.rand([10, 40])
    >>> QBN = QBatchNorm(input_size=40)
    >>> out_tensor = QBN(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40])

    """

    def __init__(
        self,
        input_size,
        dim=-1,
        gamma_init=1.0,
        beta_param=True,
        momentum=0.1,
        eps=1e-4,
        track_running_stats=True,
    ):
        super(QBatchNorm, self).__init__()

        self.num_features = input_size // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.momentum = momentum
        self.dim = dim
        self.eps = eps
        self.track_running_stats = track_running_stats

        self.gamma = Parameter(torch.full([self.num_features], self.gamma_init))
        self.beta = Parameter(
            torch.zeros(self.num_features * 4), requires_grad=self.beta_param
        )

        # instantiate moving statistics
        if track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(self.num_features * 4)
            )
            self.register_buffer("running_var", torch.ones(self.num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

    def forward(self, input):
        """Returns the normalized input tensor.

        Arguments
        ---------
        input : paddle.Tensor (batch, time, [channels])
            Input to normalize. It can be 2d, 3d, 4d.
        """

        exponential_average_factor = 0.0

        # Entering training mode
        if self.training:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1

            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = (
                    1.0 / self.num_batches_tracked.item()
                )
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

            # Get mean along batch axis
            mu = torch.mean(input, dim=0)
            mu_r, mu_i, mu_j, mu_k = torch.chunk(mu, 4, dim=self.dim)

            # Get variance along batch axis
            delta = input - mu
            delta_r, delta_i, delta_j, delta_k = torch.chunk(
                delta, 4, dim=self.dim
            )
            quat_variance = torch.mean(
                (delta_r ** 2 + delta_i ** 2 + delta_j ** 2 + delta_k ** 2),
                dim=0,
            )

            denominator = torch.sqrt(quat_variance + self.eps)

            # x - mu / sqrt(var + e)
            out = input / torch.cat(
                [denominator, denominator, denominator, denominator],
                dim=self.dim,
            )

            # Update the running stats
            if self.track_running_stats:
                self.running_mean = (
                    1 - exponential_average_factor
                ) * self.running_mean + exponential_average_factor * mu.view(
                    self.running_mean.size()
                )

                self.running_var = (
                    1 - exponential_average_factor
                ) * self.running_var + exponential_average_factor * quat_variance.view(
                    self.running_var.size()
                )
        else:
            q_var = torch.cat(
                [
                    self.running_var,
                    self.running_var,
                    self.running_var,
                    self.running_var,
                ],
                dim=self.dim,
            )
            out = (input - self.running_mean) / q_var

        # lambda * (x - mu / sqrt(var + e)) + beta

        q_gamma = torch.cat(
            [self.gamma, self.gamma, self.gamma, self.gamma], dim=self.dim
        )
        out = (q_gamma * out) + self.beta

        return out
