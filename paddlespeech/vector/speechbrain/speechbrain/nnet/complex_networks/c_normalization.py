"""Library implementing complex-valued normalization.

Authors
 * Titouan Parcollet 2020
"""

import paddle
from torch.nn import Parameter
import numpy as np
from speechbrain.nnet.complex_networks.c_ops import multi_mean


class CBatchNorm(paddle.nn.Layer):
    """This class is implements the complex-valued batch-normalization
    as introduced by "Deep Complex Networks", Trabelsi C. et al.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    input_size : int
        Expected size of the input.
    dim : int, optional
        It defines the axis that should be normalized. It usually correspond to
        the channel dimension (default -1).
    eps : float, optional
        Term used to stabilize operation (default 1e-4).
    momentum : float, optional
        It defines the momentum as for the real-valued batch-normalization
        (default 0.1).
    scale : bool, optional,
        It defines if scaling should be used or not. It is
        equivalent to the real-valued batchnormalization scaling (default True).
    center : bool, optional
        It defines if centering should be used or not. It is
        equivalent to the real-valued batchnormalization centering
        (default True).
    track_running_stats : bool, optional
        Equivalent to the real-valued batchnormalization parameter.
        When True, stats are tracked. When False, solely statistics computed
        over the batch are used (default True).

    Example
    -------
    >>> inp_tensor = torch.rand([10, 16, 30])
    >>> CBN = CBatchNorm(input_shape=inp_tensor.shape)
    >>> out_tensor = CBN(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 16, 30])

    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        dim=-1,
        eps=1e-4,
        momentum=0.1,
        scale=True,
        center=True,
        track_running_stats=True,
    ):
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.scale = scale
        self.center = center
        self.track_running_stats = track_running_stats

        if input_size is None:
            self.num_complex_features = self._check_input(input_shape)
        else:
            self.num_complex_features = input_size // 2

        if self.scale:
            self.gamma_rr = Parameter(torch.empty(self.num_complex_features))
            self.gamma_ii = Parameter(torch.empty(self.num_complex_features))
            self.gamma_ri = Parameter(torch.empty(self.num_complex_features))
        else:
            self.register_parameter("gamma_rr", None)
            self.register_parameter("gamma_ii", None)
            self.register_parameter("gamma_ri", None)

        if self.center:
            self.beta = Parameter(torch.empty(self.num_complex_features * 2))
        else:
            self.register_parameter("beta", None)

        if self.track_running_stats:

            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
            if self.scale:
                # We initializing the scaling parameter following the proposal
                # of "Deep Complex Networks". Trabelsi C. et al.

                self.register_buffer(
                    "moving_Vrr",
                    torch.ones(self.num_complex_features) * np.sqrt(1 / 2),
                )
                self.register_buffer(
                    "moving_Vii",
                    torch.ones(self.num_complex_features) * np.sqrt(1 / 2),
                )
                self.register_buffer(
                    "moving_Vri", torch.zeros(self.num_complex_features)
                )
            else:
                self.register_parameter("moving_Vrr", None)
                self.register_parameter("moving_Vii", None)
                self.register_parameter("moving_Vri", None)

            if self.center:
                self.register_buffer(
                    "moving_mean", torch.zeros(self.num_complex_features * 2)
                )
            else:
                self.register_parameter("moving_mean", None)

        else:
            self.register_parameter("moving_Vrr", None)
            self.register_parameter("moving_Vii", None)
            self.register_parameter("moving_Vri", None)
            self.register_parameter("moving_mean", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        # Simply reset the running statistics to the initial values
        # "Deep Complex Networks" Trabelsi C. et al.
        if self.track_running_stats:
            if self.center:
                self.moving_mean.zero_()
            if self.scale:
                self.moving_Vrr.fill_(1 / np.sqrt(2))
                self.moving_Vii.fill_(1 / np.sqrt(2))
                self.moving_Vri.zero_()
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        # Simply reset all the parameters.
        # "Deep Complex Networks" Trabelsi C. et al.
        self.reset_running_stats()
        if self.scale:
            self.gamma_rr.data.fill_(1 / np.sqrt(2))
            self.gamma_ii.data.fill_(1 / np.sqrt(2))
            self.gamma_ri.data.zero_()
        if self.center:
            self.beta.data.zero_()

    def forward(self, input):
        """Returns the normalized input tensor.

        Arguments
        ---------
        input : paddle.Tensor (batch, time, [channels])
            Input to normalize. It can be 2d, 3d, 4d.
        """
        exponential_average_factor = 0.0

        # Initialize moving parameters
        if self.training and self.track_running_stats:
            if self.center:
                self.moving_mean = self.moving_mean.detach()
            if self.scale:
                self.moving_Vrr = self.moving_Vrr.detach()
                self.moving_Vii = self.moving_Vii.detach()
                self.moving_Vri = self.moving_Vri.detach()

            self.num_batches_tracked = self.num_batches_tracked.detach()
            self.num_batches_tracked += 1

        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / self.num_batches_tracked.item()
        else:  # use exponential moving average
            exponential_average_factor = self.momentum

        input_shape = input.size()
        ndim = input.dim()
        reduction_axes = list(range(ndim))
        del reduction_axes[self.dim]
        input_dim = input_shape[self.dim] // 2

        # Get the mean and center the input
        mu = multi_mean(input, reduction_axes, True)
        input_centred = input - mu

        if self.scale:
            centred_squared = input_centred ** 2

        # Retrieve the real and image parts of the input tensor w.r.t the
        # dimension
        if self.scale:
            (
                centred_squared_real,
                centred_squared_imag,
            ) = self._retrieve_real_imag(centred_squared, ndim, input_dim)
        if self.center:
            centred_real, centred_imag = self._retrieve_real_imag(
                input_centred, ndim, input_dim
            )

        # We compute the mean for each component
        if self.scale:
            Vrr = (
                multi_mean(
                    centred_squared_real, axes=reduction_axes, keepdim=True
                )
                + self.eps
            )
            Vii = (
                multi_mean(
                    centred_squared_imag, axes=reduction_axes, keepdim=True
                )
                + self.eps
            )

            # Vri contains the real and imaginary covariance
            # for each feature map.
            Vri = multi_mean(
                centred_real * centred_imag, axes=reduction_axes, keepdim=True
            )
        else:
            Vrr = None
            Vii = None
            Vri = None

        # Pick the normalized form corresponding
        # to the training phase when we use running stats.
        if self.training and self.track_running_stats:
            if self.center:
                self.moving_mean = (
                    1 - exponential_average_factor
                ) * self.moving_mean + exponential_average_factor * mu.view(
                    self.moving_mean.size()
                )
            if self.scale:
                self.moving_Vrr = (
                    1 - exponential_average_factor
                ) * self.moving_Vrr + exponential_average_factor * Vrr.view(
                    self.moving_Vrr.size()
                )
                self.moving_Vii = (
                    1 - exponential_average_factor
                ) * self.moving_Vii + exponential_average_factor * Vii.view(
                    self.moving_Vii.size()
                )
                self.moving_Vri = (
                    1 - exponential_average_factor
                ) * self.moving_Vri + exponential_average_factor * Vri.view(
                    self.moving_Vri.size()
                )

        if self.training or (not self.track_running_stats):
            input_inferred = input_centred if self.center else input
            return c_norm(
                input_inferred,
                Vrr,
                Vii,
                Vri,
                self.beta,
                self.gamma_rr,
                self.gamma_ri,
                self.gamma_ii,
                self.scale,
                self.center,
                layernorm=False,
                dim=self.dim,
            )
        else:  # if we are not training or using running_stats
            if self.center:
                input_inferred = input - self.moving_mean.view(mu.size())
            else:
                input_inferred = input
            return c_norm(
                input_inferred,
                self.moving_Vrr,
                self.moving_Vii,
                self.moving_Vri,
                self.beta,
                self.gamma_rr,
                self.gamma_ri,
                self.gamma_ii,
                self.scale,
                self.center,
                layernorm=False,
                dim=self.dim,
            )

    def _retrieve_real_imag(self, tensor, ndim, input_dim):
        """
        Function used to retrieve the real and imaginary component of a tensor
        according to the dimensions
        """

        if self.dim == 1 or ndim == 2:
            tensor_real = tensor[:, :input_dim]
            tensor_imag = tensor[:, input_dim:]
        elif self.dim == -1 and ndim == 3:
            tensor_real = tensor[:, :, :input_dim]
            tensor_imag = tensor[:, :, input_dim:]
        elif self.dim == -1 and ndim == 4:
            tensor_real = tensor[:, :, :, :input_dim]
            tensor_imag = tensor[:, :, :, input_dim:]
        else:
            msg = "Retrieve_real_imag expects 2d to 4d inputs. Got " + str(
                len(tensor)
            )
            raise ValueError(msg)

        return tensor_real, tensor_imag

    def _check_input(self, input_shape):
        """
        Checks the input and returns the number of complex values.
        """

        if input_shape[self.dim] % 2 == 0:
            return input_shape[self.dim] // 2
        else:
            msg = "ComplexBatchNorm dim must be divisible by 2 ! Got " + str(
                input_shape[self.dim]
            )
            raise ValueError(msg)


class CLayerNorm(paddle.nn.Layer):
    """This class is used to instantiate the complex
    layer-normalization as introduced by "Deep Complex Networks",
    Trabelsi C. et al.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    input_size : int
        Expected size of the input dimension.
    dim : int, optional
        It defines the axis that should be normalized. It usually correspond to
        the channel dimension (default -1).
    eps : float, optional
        Term used to stabilize operation (default 1e-4).
    scale : bool, optional,
        It defines if scaling should be used or not. It is
        equivalent to the real-valued batchnormalization scaling (default True).
    center : bool, optional
        It defines if centering should be used or not. It is
        equivalent to the real-valued batchnormalization centering
        (default True).

    Example
    -------
    >>> inp_tensor = torch.rand([10, 16, 30])
    >>> CBN = CLayerNorm(input_shape=inp_tensor.shape)
    >>> out_tensor = CBN(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 16, 30])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        dim=-1,
        eps=1e-4,
        scale=True,
        center=True,
    ):

        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = scale
        self.center = center

        if input_size is None:
            self.num_complex_features = self._check_input(input_shape)
        else:
            self.num_complex_features = input_size // 2

        if self.scale:
            self.gamma_rr = Parameter(torch.empty(self.num_complex_features))
            self.gamma_ii = Parameter(torch.empty(self.num_complex_features))
            self.gamma_ri = Parameter(torch.empty(self.num_complex_features))
        else:
            self.register_parameter("gamma_rr", None)
            self.register_parameter("gamma_ii", None)
            self.register_parameter("gamma_ri", None)

        if self.center:
            self.beta = Parameter(torch.empty(self.num_complex_features * 2))
        else:
            self.register_parameter("beta", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Simply reset all the parameters.
        # "Deep Complex Networks" Trabelsi C. et al.
        if self.scale:
            self.gamma_rr.data.fill_(1 / np.sqrt(2))
            self.gamma_ii.data.fill_(1 / np.sqrt(2))
            self.gamma_ri.data.zero_()
        if self.center:
            self.beta.data.zero_()

    def forward(self, input):

        input_shape = input.size()
        ndim = input.dim()
        reduction_axes = list(range(ndim))
        del reduction_axes[self.dim]
        del reduction_axes[0]
        input_dim = input_shape[self.dim] // 2

        # Get the mean and center
        mu = multi_mean(input, reduction_axes, True)
        if self.center:
            input_centred = input - mu
        else:
            input_centred = input

        centred_squared = input_centred ** 2

        if self.dim == 1 or ndim == 2:
            centred_squared_real = centred_squared[:, :input_dim]
            centred_squared_imag = centred_squared[:, input_dim:]
            centred_real = input_centred[:, :input_dim]
            centred_imag = input_centred[:, input_dim:]
        elif self.dim == -1 and ndim == 3:
            centred_squared_real = centred_squared[:, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, input_dim:]
            centred_real = input_centred[:, :, :input_dim]
            centred_imag = input_centred[:, :, input_dim:]
        elif self.dim == -1 and ndim == 4:
            centred_squared_real = centred_squared[:, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, input_dim:]
        else:
            centred_squared_real = centred_squared[:, :, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, :, input_dim:]

        if self.scale:
            Vrr = (
                multi_mean(
                    centred_squared_real, axes=reduction_axes, keepdim=True
                )
                + self.eps
            )
            Vii = (
                multi_mean(
                    centred_squared_imag, axes=reduction_axes, keepdim=True
                )
                + self.eps
            )

            Vri = multi_mean(
                centred_real * centred_imag, axes=reduction_axes, keepdim=True
            )
        else:
            Vrr = None
            Vii = None
            Vri = None

        return c_norm(
            input_centred,
            Vrr,
            Vii,
            Vri,
            self.beta,
            self.gamma_rr,
            self.gamma_ri,
            self.gamma_ii,
            self.scale,
            self.center,
            dim=self.dim,
            layernorm=True,
        )

    def _check_input(self, input_shape):
        """Checks the input and returns the number of complex values.
        """

        if input_shape[self.dim] % 2 == 0:
            return input_shape[self.dim] // 2
        else:
            msg = "ComplexBatchNorm dim must be dividble by 2 ! Got " + str(
                input_shape[self.dim]
            )
            raise ValueError(msg)


def c_norm(
    input_centred,
    Vrr,
    Vii,
    Vri,
    beta,
    gamma_rr,
    gamma_ri,
    gamma_ii,
    scale=True,
    center=True,
    layernorm=False,
    dim=-1,
):

    """This function is used to apply the complex normalization
    as introduced by "Deep Complex Networks", Trabelsi C. et al.

    Arguments
    ---------
    input_centred : paddle.Tensor
        It is the tensor to be normalized. The features
        dimension is divided by 2 with the first half
        corresponding to the real-parts and the second half
        to the imaginary parts.
    Vrr : paddle.Tensor
        It is a tensor that contains the covariance between real-parts.
    Vii : paddle.Tensor
        It is a tensor that contains the covariance between imaginary-parts.
    Vri : paddle.Tensor
        It is a tensor that contains the covariance between real-parts and
        imaginary-parts.
    beta : paddle.Tensor
        It is a tensor corresponding to the beta parameter on the real-valued
        batch-normalization, but in the complex-valued space.
    gamma_rr : paddle.Tensor
        It is a tensor that contains the gamma between real-parts.
    gamma_ii : paddle.Tensor
        It is a tensor that contains the gamma between imaginary-parts.
    gamma_ri : paddle.Tensor
        It is a tensor that contains the gamma between real-parts and
        imaginary-parts.
    scale : bool, optional
        It defines if scaling should be used or not. It is
        equivalent to the real-valued batchnormalization
        scaling (default True).
    center : bool, optional,
        It defines if centering should be used or not. It is
        equivalent to the real-valued batchnormalization centering
        (default True).
    layernorm : bool, optional
        It defines is c_standardization is called from a layernorm or a
        batchnorm layer (default False).
    dim : int, optional
        It defines the axis that should be considered as the complex-valued
        axis (divided by 2 to get r and i) (default -1).
    """

    ndim = input_centred.dim()
    input_dim = input_centred.size(dim) // 2
    if scale:
        gamma_broadcast_shape = [1] * ndim
        gamma_broadcast_shape[dim] = input_dim
    if center:
        broadcast_beta_shape = [1] * ndim
        broadcast_beta_shape[dim] = input_dim * 2

    if scale:
        standardized_output = c_standardization(
            input_centred, Vrr, Vii, Vri, layernorm, dim=dim
        )

        # Now we perform the scaling and Shifting of the normalized x using
        # the scaling parameter
        #           [  gamma_rr gamma_ri  ]
        #   Gamma = [  gamma_ri gamma_ii  ]
        # and the shifting parameter
        #    Beta = [beta_real beta_imag].T
        # where:
        # x_real_BN = gamma_rr * x_real_normed +
        #             gamma_ri * x_imag_normed + beta_real
        # x_imag_BN = gamma_ri * x_real_normed +
        #             gamma_ii * x_imag_normed + beta_imag

        broadcast_gamma_rr = gamma_rr.view(gamma_broadcast_shape)
        broadcast_gamma_ri = gamma_ri.view(gamma_broadcast_shape)
        broadcast_gamma_ii = gamma_ii.view(gamma_broadcast_shape)

        cat_gamma_4_real = torch.cat(
            [broadcast_gamma_rr, broadcast_gamma_ii], dim=dim
        )
        cat_gamma_4_imag = torch.cat(
            [broadcast_gamma_ri, broadcast_gamma_ri], dim=dim
        )
        if dim == 0:
            centred_real = standardized_output[:input_dim]
            centred_imag = standardized_output[input_dim:]
        elif dim == 1 or (dim == -1 and ndim == 2):
            centred_real = standardized_output[:, :input_dim]
            centred_imag = standardized_output[:, input_dim:]
        elif dim == -1 and ndim == 3:
            centred_real = standardized_output[:, :, :input_dim]
            centred_imag = standardized_output[:, :, input_dim:]
        elif dim == -1 and ndim == 4:
            centred_real = standardized_output[:, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, input_dim:]
        else:
            centred_real = standardized_output[:, :, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, :, input_dim:]

        rolled_standardized_output = torch.cat(
            [centred_imag, centred_real], dim=dim
        )
        if center:
            broadcast_beta = beta.view(broadcast_beta_shape)
            a = cat_gamma_4_real * standardized_output
            b = cat_gamma_4_imag * rolled_standardized_output
            return a + b + broadcast_beta
        else:
            return (
                cat_gamma_4_real * standardized_output
                + cat_gamma_4_imag * rolled_standardized_output
            )
    else:
        if center:
            broadcast_beta = beta.view(broadcast_beta_shape)
            return input_centred + broadcast_beta
        else:
            return input_centred


def c_standardization(input_centred, Vrr, Vii, Vri, layernorm=False, dim=-1):
    """This function is used to standardize a centred tensor of
    complex numbers (mean of the set must be 0).

    Arguments
    ---------
    input_centred : paddle.Tensor
        It is the tensor to be normalized. The features
        dimension is divided by 2 with the first half
        corresponding to the real-parts and the second half
        to the imaginary parts.
    Vrr : paddle.Tensor
        It is a tensor that contains the covariance between real-parts.
    Vii : paddle.Tensor
        It is a tensor that contains the covariance between imaginary-parts.
    Vri : paddle.Tensor
        It is a tensor that contains the covariance between real-parts and
        imaginary-parts.
    layernorm : bool, optional
        It defines is c_standardization is called from a layernorm or a
        batchnorm layer (default False).
    dim : int, optional
        It defines the axis that should be considered as the complex-valued
        axis (divided by 2 to get r and i) (default -1).
    """
    ndim = input_centred.dim()
    input_dim = input_centred.size(dim) // 2
    variances_broadcast = [1] * ndim
    variances_broadcast[dim] = input_dim

    if layernorm:
        variances_broadcast[0] = input_centred.size(0)

    # We require the covariance matrix's inverse square root. That requires
    # square rooting, followed by inversion (During the computation of square
    # root we compute the determinant we'll need for inversion as well).

    # tau = Vrr + Vii = Trace. Guaranteed >=0 because Positive-definite matrix
    tau = Vrr + Vii

    # delta = (Vrr * Vii) - (Vri ** 2) = Determinant
    delta = (Vrr * Vii) - (Vri ** 2)

    s = delta.sqrt()
    t = (tau + 2 * s).sqrt()

    # The square root matrix could now be explicitly formed as
    #       [ Vrr+s Vri   ]
    # (1/t) [ Vir   Vii+s ]
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    # but we don't need to do this immediately since we can also simultaneously
    # invert. We can do this because we've already computed the determinant of
    # the square root matrix, and can thus invert it using the analytical
    # solution for 2x2 matrices
    #      [ A B ]             [  D  -B ]
    # inv( [ C D ] ) = (1/det) [ -C   A ]
    # http://mathworld.wolfram.com/MatrixInverse.html
    # Thus giving us
    #           [  Vii+s  -Vri   ]
    # (1/s)(1/t)[ -Vir     Vrr+s ]
    # So we proceed as follows:

    inverse_st = 1.0 / (s * t)
    Wrr = (Vii + s) * inverse_st
    Wii = (Vrr + s) * inverse_st
    Wri = -Vri * inverse_st

    # And we have computed the inverse square root matrix W = sqrt(V)!
    # Normalization. We multiply, x_normalized = W.x.

    # The returned result will be a complex standardized input
    # where the real and imaginary parts are obtained as follows:
    # x_real_normed = Wrr * x_real_centred + Wri * x_imag_centred
    # x_imag_normed = Wri * x_real_centred + Wii * x_imag_centred

    broadcast_Wrr = Wrr.view(variances_broadcast)
    broadcast_Wri = Wri.view(variances_broadcast)
    broadcast_Wii = Wii.view(variances_broadcast)

    cat_W_4_real = torch.cat([broadcast_Wrr, broadcast_Wii], dim=dim)
    cat_W_4_imag = torch.cat([broadcast_Wri, broadcast_Wri], dim=dim)

    if dim == 0:
        centred_real = input_centred[:input_dim]
        centred_imag = input_centred[input_dim:]
    elif dim == 1 or (dim == -1 and ndim == 2):
        centred_real = input_centred[:, :input_dim]
        centred_imag = input_centred[:, input_dim:]
    elif dim == -1 and ndim == 3:
        centred_real = input_centred[:, :, :input_dim]
        centred_imag = input_centred[:, :, input_dim:]
    elif dim == -1 and ndim == 4:
        centred_real = input_centred[:, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, input_dim:]
    else:
        centred_real = input_centred[:, :, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, :, input_dim:]

    rolled_input = torch.cat([centred_imag, centred_real], dim=dim)

    output = cat_W_4_real * input_centred + cat_W_4_imag * rolled_input

    #   Wrr * x_real_centered | Wii * x_imag_centered
    # + Wri * x_imag_centered | Wri * x_real_centered
    # -----------------------------------------------
    # = output

    return output
