"""Library implementing quaternion-valued recurrent neural networks.

Authors
 * Titouan Parcollet 2020
"""

import paddle
import logging
from speechbrain.nnet.quaternion_networks.q_linear import QLinear
from speechbrain.nnet.quaternion_networks.q_normalization import QBatchNorm
from paddle import Tensor
from typing import Optional

logger = logging.getLogger(__name__)


class QLSTM(paddle.nn.Layer):
    """ This function implements a quaternion-valued LSTM as first introduced
    in : "Quaternion Recurrent Neural Networks", Parcollet T. et al.

    Input format is (batch, time, fea) or (batch, time, fea, channel).
    In the latter shape, the two last dimensions will be merged:
    (batch, time, fea * channel)

    Arguments
    ---------
    hidden_size : int
        Number of output neurons (i.e, the dimensionality of the output).
        Specified value is in terms of quaternion-valued neurons. Thus, the output
        is 4*hidden_size.
    num_layers : int, optional
        Number of layers to employ in the RNN architecture (default 1).
    bias : bool, optional
        If True, the additive bias b is adopted (default True).
    dropout : float, optional
        It is the dropout factor (must be between 0 and 1) (default 0.0).
    bidirectional : bool, optional
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used (default False).
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
    autograd : bool, optional
        When True, the default PyTorch autograd will be used. When False, a
        custom backpropagation will be used, reducing by a factor 3 to 4 the
        memory consumption. It is also 2x slower (default True).


    Example
    -------
    >>> inp_tensor = torch.rand([10, 16, 40])
    >>> rnn = QLSTM(hidden_size=16, input_shape=inp_tensor.shape)
    >>> out_tensor = rnn(inp_tensor)
    >>>
    torch.Size([10, 16, 64])
    """

    def __init__(
        self,
        hidden_size,
        input_shape,
        num_layers=1,
        bias=True,
        dropout=0.0,
        bidirectional=False,
        init_criterion="glorot",
        weight_init="quaternion",
        autograd=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size * 4
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.reshape = False
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.autograd = autograd

        if len(input_shape) > 3:
            self.reshape = True

        # Computing the feature dimensionality
        self.fea_dim = torch.prod(torch.tensor(input_shape[2:]))
        self.batch_size = input_shape[0]

        self.rnn = self._init_layers()

    def _init_layers(self,):
        """Initializes the layers of the quaternionLSTM.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim
        for i in range(self.num_layers):
            rnn_lay = QLSTM_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                init_criterion=self.init_criterion,
                weight_init=self.weight_init,
                autograd=self.autograd,
            )

            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size

        return rnn

    def forward(self, x, hx: Optional[Tensor] = None):
        """Returns the output of the vanilla QuaternionRNN.

        Arguments
        ---------
        x : paddle.Tensor
            Input tensor.
        """

        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        output, hh = self._forward_rnn(x, hx=hx)

        return output, hh

    def _forward_rnn(self, x, hx: Optional[Tensor]):
        """Returns the output of the vanilla QuaternionRNN.

        Arguments
        ---------
        x : paddle.Tensor
            Input tensor.
        """

        h = []
        if hx is not None:
            if self.bidirectional:
                hx = hx.reshape(
                    self.num_layers, self.batch_size * 2, self.hidden_size
                )

        # Processing the different layers
        for i, rnn_lay in enumerate(self.rnn):
            if hx is not None:
                x = rnn_lay(x, hx=hx[i])
            else:
                x = rnn_lay(x, hx=None)
            h.append(x[:, -1, :])
        h = torch.stack(h, dim=1)

        if self.bidirectional:
            h = h.reshape(h.shape[1] * 2, h.shape[0], self.hidden_size)
        else:
            h = h.transpose(0, 1)

        return x, h


class QLSTM_Layer(paddle.nn.Layer):
    """ This function implements quaternion-valued LSTM layer.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors (in term of real values).
    batch_size : int
        Batch size of the input tensors.
    hidden_size : int
        Number of output values (in term of real values).
    num_layers : int, optional
        Number of layers to employ in the RNN architecture (default 1).
    dropout : float, optional
        It is the dropout factor (must be between 0 and 1) (default 0.0).
    bidirectional : bool, optional
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used (default False).
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
    autograd : bool, optional
        When True, the default PyTorch autograd will be used. When False, a
        custom backpropagation will be used, reducing by a factor 3 to 4 the
        memory consumption. It is also 2x slower (default True).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        bidirectional=False,
        init_criterion="glorot",
        weight_init="quaternion",
        autograd="true",
    ):

        super(QLSTM_Layer, self).__init__()

        self.hidden_size = int(hidden_size) // 4  # Express in term of quat
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.autograd = autograd

        self.w = QLinear(
            input_shape=self.input_size,
            n_neurons=self.hidden_size * 4,  # Forget, Input, Output, Cell
            bias=True,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
            autograd=self.autograd,
        )

        self.u = QLinear(
            input_shape=self.hidden_size * 4,  # The input size is in real
            n_neurons=self.hidden_size * 4,
            bias=True,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
            autograd=self.autograd,
        )

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initial state
        self.register_buffer("h_init", torch.zeros(1, self.hidden_size * 4))

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop(self.batch_size)

        # Initializing dropout
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)

        self.drop_mask_te = torch.tensor([1.0]).float()

    def forward(self, x, hx: Optional[Tensor] = None):
        # type: (Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the output of the QuaternionRNN_layer.

        Arguments
        ---------
        x : paddle.Tensor
            Input tensor.
        """
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        self._change_batch_size(x)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)

        # Processing time steps
        if hx is not None:
            h = self._quaternionlstm_cell(w, hx)
        else:
            h = self._quaternionlstm_cell(w, self.h_init)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    def _quaternionlstm_cell(self, w, ht):
        """Returns the hidden states for each time step.

        Arguments
        ---------
        wx : paddle.Tensor
            Linearly transformed input.
        """

        hiddens = []

        # Initialise the cell state
        ct = self.h_init

        # Sampling dropout mask
        drop_mask = self._sample_drop_mask(w)

        # Loop over time axis
        for k in range(w.shape[1]):
            gates = w[:, k] + self.u(ht)
            (
                itr,
                iti,
                itj,
                itk,
                ftr,
                fti,
                ftj,
                ftk,
                otr,
                oti,
                otj,
                otk,
                ctr,
                cti,
                ctj,
                ctk,
            ) = gates.chunk(16, 1)
            it = torch.sigmoid(torch.cat([itr, iti, itj, itk], dim=-1))
            ft = torch.sigmoid(torch.cat([ftr, fti, ftj, ftk], dim=-1))
            ot = torch.sigmoid(torch.cat([otr, oti, otj, otk], dim=-1))

            ct = (
                it
                * torch.tanh(torch.cat([ctr, cti, ctj, ctk], dim=-1))
                * drop_mask
                + ft * ct
            )
            ht = ot * torch.tanh(ct)
            hiddens.append(ht)

        # Stacking hidden states
        h = torch.stack(hiddens, dim=1)
        return h

    def _init_drop(self, batch_size):
        """Initializes the recurrent dropout operation. To speed it up,
        the dropout masks are sampled in advance.
        """
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)
        self.drop_mask_te = torch.tensor([1.0]).float()

        self.N_drop_masks = 16000
        self.drop_mask_cnt = 0

        self.drop_masks = self.drop(
            torch.ones(self.N_drop_masks, self.hidden_size * 4,)
        ).data

    def _sample_drop_mask(self, w):
        """Selects one of the pre-defined dropout masks.
        """
        if self.training:

            # Sample new masks when needed
            if self.drop_mask_cnt + self.batch_size > self.N_drop_masks:
                self.drop_mask_cnt = 0
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size * 4, device=w.device
                    )
                ).data

            # Sampling the mask
            drop_mask = self.drop_masks[
                self.drop_mask_cnt : self.drop_mask_cnt + self.batch_size
            ]
            self.drop_mask_cnt = self.drop_mask_cnt + self.batch_size

        else:
            drop_mask = self.drop_mask_te

        return drop_mask

    def _change_batch_size(self, x):
        """This function changes the batch size when it is different from
        the one detected in the initialization method. This might happen in
        the case of multi-gpu or when we have different batch sizes in train
        and test. We also update the h_int and drop masks.
        """

        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

            if self.training:
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size * 4, device=x.device
                    )
                ).data


class QRNN(paddle.nn.Layer):
    """ This function implements a vanilla quaternion-valued RNN.

    Input format is (batch, time, fea) or (batch, time, fea, channel).
    In the latter shape, the two last dimensions will be merged:
    (batch, time, fea * channel)

    Arguments
    ---------
    hidden_size : int
        Number of output neurons (i.e, the dimensionality of the output).
        Specified value is in term of quaternion-valued neurons. Thus, the output
        is 4*hidden_size.
    num_layers : int, optional
        Number of layers to employ in the RNN architecture (default 1).
    nonlinearity : str, optional
        Type of nonlinearity (tanh, relu) (default "tanh").
    bias : bool, optional
        If True, the additive bias b is adopted (default True).
    dropout : float, optional
        It is the dropout factor (must be between 0 and 1) (default 0.0).
    bidirectional : bool, optional
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used (default False).
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
    autograd : bool, optional
        When True, the default PyTorch autograd will be used. When False, a
        custom backpropagation will be used, reducing by a factor 3 to 4 the
        memory consumption. It is also 2x slower (default True).


    Example
    -------
    >>> inp_tensor = torch.rand([10, 16, 40])
    >>> rnn = QRNN(hidden_size=16, input_shape=inp_tensor.shape)
    >>> out_tensor = rnn(inp_tensor)
    >>>
    torch.Size([10, 16, 64])
    """

    def __init__(
        self,
        hidden_size,
        input_shape,
        nonlinearity="tanh",
        num_layers=1,
        bias=True,
        dropout=0.0,
        bidirectional=False,
        init_criterion="glorot",
        weight_init="quaternion",
        autograd=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size * 4  # z = x + iy
        self.nonlinearity = nonlinearity
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.reshape = False
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.autograd = autograd

        if len(input_shape) > 3:
            self.reshape = True

        # Computing the feature dimensionality
        self.fea_dim = torch.prod(torch.tensor(input_shape[2:]))
        self.batch_size = input_shape[0]

        self.rnn = self._init_layers()

    def _init_layers(self,):
        """
        Initializes the layers of the quaternionRNN.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """

        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim
        for i in range(self.num_layers):
            rnn_lay = QRNN_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                nonlinearity=self.nonlinearity,
                bidirectional=self.bidirectional,
                init_criterion=self.init_criterion,
                weight_init=self.weight_init,
                autograd=self.autograd,
            )

            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size

        return rnn

    def forward(self, x, hx: Optional[Tensor] = None):
        """Returns the output of the vanilla QuaternionRNN.

        Arguments
        ---------
        x : paddle.Tensor
        """
        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        output, hh = self._forward_rnn(x, hx=hx)

        return output, hh

    def _forward_rnn(self, x, hx: Optional[Tensor]):
        """Returns the output of the vanilla QuaternionRNN.

        Arguments
        ---------
        x : paddle.Tensor
        """
        h = []
        if hx is not None:
            if self.bidirectional:
                hx = hx.reshape(
                    self.num_layers, self.batch_size * 2, self.hidden_size
                )

        # Processing the different layers
        for i, rnn_lay in enumerate(self.rnn):
            if hx is not None:
                x = rnn_lay(x, hx=hx[i])
            else:
                x = rnn_lay(x, hx=None)
            h.append(x[:, -1, :])
        h = torch.stack(h, dim=1)

        if self.bidirectional:
            h = h.reshape(h.shape[1] * 2, h.shape[0], self.hidden_size)
        else:
            h = h.transpose(0, 1)

        return x, h


class QRNN_Layer(paddle.nn.Layer):
    """This function implements quaternion-valued recurrent layer.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors (in term of real values).
    batch_size : int
        Batch size of the input tensors.
    hidden_size : int
        Number of output values (in term of real values).
    num_layers : int, optional
        Number of layers to employ in the RNN architecture (default 1).
    nonlinearity : str, optional
        Type of nonlinearity (tanh, relu) (default "tanh").
    dropout : float, optional
        It is the dropout factor (must be between 0 and 1) (default 0.0).
    bidirectional : bool, optional
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used (default False).
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
    autograd : bool, optional
        When True, the default PyTorch autograd will be used. When False, a
        custom backpropagation will be used, reducing by a factor 3 to 4 the
        memory consumption. It is also 2x slower (default True).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        nonlinearity="tanh",
        bidirectional=False,
        init_criterion="glorot",
        weight_init="quaternion",
        autograd="true",
    ):

        super(QRNN_Layer, self).__init__()

        self.hidden_size = int(hidden_size) // 4  # Express in term of quat
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.autograd = autograd

        self.w = QLinear(
            input_shape=self.input_size,
            n_neurons=self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
            autograd=self.autograd,
        )

        self.u = QLinear(
            input_shape=self.hidden_size * 4,  # The input size is in real
            n_neurons=self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
            autograd=self.autograd,
        )

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initial state
        self.register_buffer("h_init", torch.zeros(1, self.hidden_size * 4))

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop(self.batch_size)

        # Initializing dropout
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)

        self.drop_mask_te = torch.tensor([1.0]).float()

        # Setting the activation function
        if nonlinearity == "tanh":
            self.act = torch.nn.Tanh()
        else:
            self.act = torch.nn.ReLU()

    def forward(self, x, hx: Optional[Tensor] = None):
        # type: (Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the output of the QuaternionRNN_layer.

        Arguments
        ---------
        x : paddle.Tensor
        """
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        self._change_batch_size(x)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)

        # Processing time steps
        if hx is not None:
            h = self._quaternionrnn_cell(w, hx)
        else:
            h = self._quaternionrnn_cell(w, self.h_init)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    def _quaternionrnn_cell(self, w, ht):
        """Returns the hidden states for each time step.

        Arguments
        ---------
        wx : paddle.Tensor
            Linearly transformed input.
        """
        hiddens = []

        # Sampling dropout mask
        drop_mask = self._sample_drop_mask(w)

        # Loop over time axis
        for k in range(w.shape[1]):
            at = w[:, k] + self.u(ht)
            ht = self.act(at) * drop_mask
            hiddens.append(ht)

        # Stacking hidden states
        h = torch.stack(hiddens, dim=1)
        return h

    def _init_drop(self, batch_size):
        """Initializes the recurrent dropout operation. To speed it up,
        the dropout masks are sampled in advance.
        """

        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)
        self.drop_mask_te = torch.tensor([1.0]).float()

        self.N_drop_masks = 16000
        self.drop_mask_cnt = 0

        self.drop_masks = self.drop(
            torch.ones(self.N_drop_masks, self.hidden_size * 4,)
        ).data

    def _sample_drop_mask(self, w):
        """Selects one of the pre-defined dropout masks.
        """

        if self.training:

            # Sample new masks when needed
            if self.drop_mask_cnt + self.batch_size > self.N_drop_masks:
                self.drop_mask_cnt = 0
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size * 4, device=w.device
                    )
                ).data

            # Sampling the mask
            drop_mask = self.drop_masks[
                self.drop_mask_cnt : self.drop_mask_cnt + self.batch_size
            ]
            self.drop_mask_cnt = self.drop_mask_cnt + self.batch_size

        else:
            drop_mask = self.drop_mask_te

        return drop_mask

    def _change_batch_size(self, x):
        """This function changes the batch size when it is different from
        the one detected in the initialization method. This might happen in
        the case of multi-gpu or when we have different batch sizes in train
        and test. We also update the h_int and drop masks.
        """

        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

            if self.training:
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size * 2, device=x.device
                    )
                ).data


class QLiGRU(paddle.nn.Layer):
    """ This function implements a quaternion-valued Light GRU (liGRU).

    Ligru is single-gate GRU model based on batch-norm + relu
    activations + recurrent dropout. For more info see:

    "M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio,
    Light Gated Recurrent Units for Speech Recognition,
    in IEEE Transactions on Emerging Topics in Computational Intelligence,
    2018" (https://arxiv.org/abs/1803.10225)

    To speed it up, it is compiled with the torch just-in-time compiler (jit)
    right before using it.

    It accepts in input tensors formatted as (batch, time, fea).
    In the case of 4d inputs like (batch, time, fea, channel) the tensor is
    flattened as (batch, time, fea*channel).

    Arguments
    ---------
    hidden_size : int
        Number of output neurons (i.e, the dimensionality of the output).
        Specified value is in term of quaternion-valued neurons. Thus, the output
        is 2*hidden_size.
    nonlinearity : str
        Type of nonlinearity (tanh, relu).
    normalization : str
        Type of normalization for the ligru model (batchnorm, layernorm).
        Every string different from batchnorm and layernorm will result
        in no normalization.
    num_layers : int
        Number of layers to employ in the RNN architecture.
    bias : bool
        If True, the additive bias b is adopted.
    dropout: float
        It is the dropout factor (must be between 0 and 1).
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used.
    init_criterion : str, optional
        (glorot, he).
        This parameter controls the initialization criterion of the weights.
        It is combined with weights_init to build the initialization method of
        the quaternion-valued weights (default "glorot").
    weight_init : str, optional
        (quaternion, unitary).
        This parameter defines the initialization procedure of the
        quaternion-valued weights. "quaternion" will generate random quaternion-valued
        weights following the init_criterion and the quaternion polar form.
        "unitary" will normalize the weights to lie on the unit circle (default "quaternion").
        More details in: "Deep quaternion Networks", Trabelsi C. et al.
    autograd : bool, optional
        When True, the default PyTorch autograd will be used. When False, a
        custom backpropagation will be used, reducing by a factor 3 to 4 the
        memory consumption. It is also 2x slower (default True).

    Example
    -------
    >>> inp_tensor = torch.rand([10, 16, 40])
    >>> rnn = QLiGRU(input_shape=inp_tensor.shape, hidden_size=16)
    >>> out_tensor = rnn(inp_tensor)
    >>>
    torch.Size([4, 10, 5])
    """

    def __init__(
        self,
        hidden_size,
        input_shape,
        nonlinearity="leaky_relu",
        num_layers=1,
        bias=True,
        dropout=0.0,
        bidirectional=False,
        init_criterion="glorot",
        weight_init="quaternion",
        autograd=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size * 4  # q = x + iy + jz + kw
        self.nonlinearity = nonlinearity
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.reshape = False
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.autograd = autograd

        if len(input_shape) > 3:
            self.reshape = True

        self.fea_dim = torch.prod(torch.tensor(input_shape[2:]))
        self.batch_size = input_shape[0]
        self.rnn = self._init_layers()

    def _init_layers(self):
        """
        Initializes the layers of the liGRU.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim

        for i in range(self.num_layers):
            rnn_lay = QLiGRU_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                nonlinearity=self.nonlinearity,
                bidirectional=self.bidirectional,
                init_criterion=self.init_criterion,
                weight_init=self.weight_init,
                autograd=self.autograd,
            )
            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size
        return rnn

    def forward(self, x, hx: Optional[Tensor] = None):
        """Returns the output of the QuaternionliGRU.

        Arguments
        ---------
        x : paddle.Tensor
        """
        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # run ligru
        output, hh = self._forward_ligru(x, hx=hx)

        return output, hh

    def _forward_ligru(self, x, hx: Optional[Tensor]):
        """Returns the output of the quaternionliGRU.

        Arguments
        ---------
        x : paddle.Tensor
            Input tensor.
        """
        h = []
        if hx is not None:
            if self.bidirectional:
                hx = hx.reshape(
                    self.num_layers, self.batch_size * 2, self.hidden_size
                )
        # Processing the different layers
        for i, ligru_lay in enumerate(self.rnn):
            if hx is not None:
                x = ligru_lay(x, hx=hx[i])
            else:
                x = ligru_lay(x, hx=None)
            h.append(x[:, -1, :])
        h = torch.stack(h, dim=1)

        if self.bidirectional:
            h = h.reshape(h.shape[1] * 2, h.shape[0], self.hidden_size)
        else:
            h = h.transpose(0, 1)

        return x, h


class QLiGRU_Layer(paddle.nn.Layer):
    """ This function implements quaternion-valued Light-Gated Recurrent Units
    (ligru) layer.

    Arguments
    ---------
    input_size: int
        Feature dimensionality of the input tensors.
    batch_size: int
        Batch size of the input tensors.
    hidden_size: int
        Number of output values.
    num_layers: int
        Number of layers to employ in the RNN architecture.
    nonlinearity: str
        Type of nonlinearity (tanh, relu).
    dropout: float
        It is the dropout factor (must be between 0 and 1).
    bidirectional: bool
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used.
    init_criterion: str , optional
        (glorot, he).
        This parameter controls the initialization criterion of the weights.
        It is combined with weights_init to build the initialization method of
        the quaternion-valued weights (default "glorot").
    weight_init: str, optional
        (quaternion, unitary).
        This parameter defines the initialization procedure of the
        quaternion-valued weights. "quaternion" will generate random quaternion
        weights following the init_criterion and the quaternion polar form.
        "unitary" will normalize the weights to lie on the unit circle (default "quaternion").
        More details in: "Deep quaternion Networks", Trabelsi C. et al.
    autograd: bool, optional
        When True, the default PyTorch autograd will be used. When False, a
        custom backpropagation will be used, reducing by a factor 3 to 4 the
        memory consumption. It is also 2x slower (default True).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        nonlinearity="leaky_relu",
        normalization="batchnorm",
        bidirectional=False,
        init_criterion="glorot",
        weight_init="quaternion",
        autograd=True,
    ):

        super(QLiGRU_Layer, self).__init__()
        self.hidden_size = int(hidden_size) // 4
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.normalization = normalization
        self.nonlinearity = nonlinearity
        self.autograd = autograd

        self.w = QLinear(
            input_shape=self.input_size,
            n_neurons=self.hidden_size * 2,
            bias=False,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
            autograd=self.autograd,
        )

        self.u = QLinear(
            input_shape=self.hidden_size * 4,  # The input size is in real
            n_neurons=self.hidden_size * 2,
            bias=False,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
            autograd=self.autograd,
        )

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initializing batch norm
        self.normalize = False

        if self.normalization == "batchnorm":
            self.norm = QBatchNorm(input_size=hidden_size * 2, dim=-1)
            self.normalize = True
        else:
            # Normalization is disabled here. self.norm is only  formally
            # initialized to avoid jit issues.
            self.norm = QBatchNorm(input_size=hidden_size * 2, dim=-1)
            self.normalize = False

        # Initial state
        self.register_buffer("h_init", torch.zeros(1, self.hidden_size * 4))

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop(self.batch_size)

        # Initializing dropout
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)

        self.drop_mask_te = torch.tensor([1.0]).float()

        # Setting the activation function
        if self.nonlinearity == "tanh":
            self.act = torch.nn.Tanh()
        elif self.nonlinearity == "leaky_relu":
            self.act = torch.nn.LeakyReLU()
        else:
            self.act = torch.nn.ReLU()

    def forward(self, x, hx: Optional[Tensor] = None):
        # type: (Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the output of the quaternion liGRU layer.

        Arguments
        ---------
        x : paddle.Tensor
            Input tensor.
        """

        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        self._change_batch_size(x)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)

        # Apply batch normalization
        if self.normalize:
            w_bn = self.norm(w.reshape(w.shape[0] * w.shape[1], w.shape[2]))
            w = w_bn.reshape(w.shape[0], w.shape[1], w.shape[2])

        # Processing time steps
        if hx is not None:
            h = self._quaternion_ligru_cell(w, hx)
        else:
            h = self._quaternion_ligru_cell(w, self.h_init)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    def _quaternion_ligru_cell(self, w, ht):
        """Returns the hidden states for each time step.

        Arguments
        ---------
        wx : paddle.Tensor
            Linearly transformed input.
        """

        hiddens = []

        # Sampling dropout mask
        drop_mask = self._sample_drop_mask(w)

        # Loop over time axis
        for k in range(w.shape[1]):
            gates = w[:, k] + self.u(ht)
            atr, ati, atj, atk, ztr, zti, ztj, ztk = gates.chunk(8, 1)
            at = torch.cat([atr, ati, atj, atk], dim=-1)
            zt = torch.cat([ztr, zti, ztj, ztk], dim=-1)
            zt = torch.sigmoid(zt)
            hcand = self.act(at) * drop_mask
            ht = zt * ht + (1 - zt) * hcand
            hiddens.append(ht)

        # Stacking hidden states
        h = torch.stack(hiddens, dim=1)
        return h

    def _init_drop(self, batch_size):
        """Initializes the recurrent dropout operation. To speed it up,
        the dropout masks are sampled in advance.
        """

        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)
        self.drop_mask_te = torch.tensor([1.0]).float()

        self.N_drop_masks = 16000
        self.drop_mask_cnt = 0

        self.register_buffer(
            "drop_masks",
            self.drop(torch.ones(self.N_drop_masks, self.hidden_size * 4)).data,
        )

    def _sample_drop_mask(self, w):
        """Selects one of the pre-defined dropout masks
        """

        if self.training:

            # Sample new masks when needed
            if self.drop_mask_cnt + self.batch_size > self.N_drop_masks:
                self.drop_mask_cnt = 0
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size * 4, device=w.device
                    )
                ).data

            # Sampling the mask
            drop_mask = self.drop_masks[
                self.drop_mask_cnt : self.drop_mask_cnt + self.batch_size
            ]
            self.drop_mask_cnt = self.drop_mask_cnt + self.batch_size

        else:
            self.drop_mask_te = self.drop_mask_te.to(w.device)
            drop_mask = self.drop_mask_te

        return drop_mask

    def _change_batch_size(self, x):
        """This function changes the batch size when it is different from
        the one detected in the initialization method. This might happen in
        the case of multi-gpu or when we have different batch sizes in train
        and test. We also update the h_int and drop masks.
        """

        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

            if self.training:
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size * 4, device=x.device
                    )
                ).data
