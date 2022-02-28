"""Library implementing complex-valued recurrent neural networks.

Authors
 * Titouan Parcollet 2020
"""

import paddle
import logging
from speechbrain.nnet.complex_networks.c_linear import CLinear
from speechbrain.nnet.complex_networks.c_normalization import (
    CBatchNorm,
    CLayerNorm,
)

logger = logging.getLogger(__name__)


class CLSTM(paddle.nn.Layer):
    """ This function implements a complex-valued LSTM.

    Input format is (batch, time, fea) or (batch, time, fea, channel).
    In the latter shape, the two last dimensions will be merged:
    (batch, time, fea * channel)

    Arguments
    ---------
    hidden_size : int
        Number of output neurons (i.e, the dimensionality of the output).
        Specified value is in term of complex-valued neurons. Thus, the output
        is 2*hidden_size.
    num_layers : int, optional
        Number of layers to employ in the RNN architecture (default 1).
    bias: bool, optional
        If True, the additive bias b is adopted (default True).
    dropout : float, optional
        It is the dropout factor (must be between 0 and 1) (default 0.0).
    return_hidden : bool, optional
        It True, the function returns the last hidden layer.
    bidirectional : bool, optional
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used (default False).
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
    >>> inp_tensor = torch.rand([10, 16, 40])
    >>> rnn = CLSTM(hidden_size=16, input_shape=inp_tensor.shape)
    >>> out_tensor = rnn(inp_tensor)
    >>>
    torch.Size([10, 16, 32])
    """

    def __init__(
        self,
        hidden_size,
        input_shape,
        num_layers=1,
        bias=True,
        dropout=0.0,
        bidirectional=False,
        return_hidden=False,
        init_criterion="glorot",
        weight_init="complex",
    ):
        super().__init__()
        self.hidden_size = hidden_size * 2
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.reshape = False
        self.return_hidden = return_hidden
        self.init_criterion = init_criterion
        self.weight_init = weight_init

        if len(input_shape) > 3:
            self.reshape = True

        # Computing the feature dimensionality
        self.fea_dim = torch.prod(torch.tensor(input_shape[2:]))
        self.batch_size = input_shape[0]

        self.rnn = self._init_layers()

    def _init_layers(self,):
        """
        Initializes the layers of the ComplexLSTM.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """

        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim
        for i in range(self.num_layers):
            rnn_lay = CLSTM_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                init_criterion=self.init_criterion,
                weight_init=self.weight_init,
            )

            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size

        return rnn

    def forward(self, x, hx=None):
        """Returns the output of the CLSTM.

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

        if self.return_hidden:
            return output, hh
        else:
            return output

    def _forward_rnn(self, x, hx):
        """Returns the output of the CLSTM.

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


class CLSTM_Layer(paddle.nn.Layer):
    """ This function implements complex-valued LSTM layer.

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
    init_criterion : str, optional
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
        weight_init="complex",
    ):

        super(CLSTM_Layer, self).__init__()

        self.hidden_size = int(hidden_size) // 2  # Express in term of quat
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.init_criterion = init_criterion
        self.weight_init = weight_init

        self.w = CLinear(
            input_shape=self.input_size,
            n_neurons=self.hidden_size * 4,  # Forget, Input, Output, Cell
            bias=True,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
        )

        self.u = CLinear(
            input_shape=self.hidden_size * 2,  # The input size is in real
            n_neurons=self.hidden_size * 4,
            bias=True,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
        )

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initial state
        self.register_buffer("h_init", torch.zeros(1, self.hidden_size * 2))

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop(self.batch_size)

        # Initializing dropout
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)

        self.drop_mask_te = torch.tensor([1.0]).float()

    def forward(self, x, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the output of the CRNN_layer.

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
            h = self._complexlstm_cell(w, hx)
        else:
            h = self._complexlstm_cell(w, self.h_init)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    def _complexlstm_cell(self, w, ht):
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
            (itr, iti, ftr, fti, otr, oti, ctr, cti) = gates.chunk(8, 1)
            it = torch.sigmoid(torch.cat([itr, iti], dim=-1))
            ft = torch.sigmoid(torch.cat([ftr, fti], dim=-1))
            ot = torch.sigmoid(torch.cat([otr, oti], dim=-1))

            ct = (
                it * torch.tanh(torch.cat([ctr, cti], dim=-1)) * drop_mask
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

        self.register_buffer(
            "drop_masks",
            self.drop(torch.ones(self.N_drop_masks, self.hidden_size * 2)).data,
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
                        self.N_drop_masks, self.hidden_size * 2, device=w.device
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
                    torch.ones(self.N_drop_masks, self.hidden_size * 2)
                ).data


class CRNN(paddle.nn.Layer):
    """ This function implements a vanilla complex-valued RNN.

    Input format is (batch, time, fea) or (batch, time, fea, channel).
    In the latter shape, the two last dimensions will be merged:
    (batch, time, fea * channel)

    Arguments
    ---------
    hidden_size : int
        Number of output neurons (i.e, the dimensionality of the output).
        Specified value is in term of complex-valued neurons. Thus, the output
        is 2*hidden_size.
    num_layers : int, optional
        Number of layers to employ in the RNN architecture (default 1).
    nonlinearity : str, optional
        Type of nonlinearity (tanh, relu) (default "tanh").
    bias : bool, optional
        If True, the additive bias b is adopted (default True).
    dropout : float, optional
        It is the dropout factor (must be between 0 and 1) (default 0.0).
    return_hidden : bool, optional
        It True, the function returns the last hidden layer (default False).
    bidirectional : bool, optional
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used (default False).
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
    >>> inp_tensor = torch.rand([10, 16, 30])
    >>> rnn = CRNN(hidden_size=16, input_shape=inp_tensor.shape)
    >>> out_tensor = rnn(inp_tensor)
    >>>
    torch.Size([10, 16, 32])
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
        return_hidden=False,
        init_criterion="glorot",
        weight_init="complex",
    ):
        super().__init__()
        self.hidden_size = hidden_size * 2  # z = x + iy
        self.nonlinearity = nonlinearity
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.reshape = False
        self.return_hidden = return_hidden
        self.init_criterion = init_criterion
        self.weight_init = weight_init

        if len(input_shape) > 3:
            self.reshape = True

        # Computing the feature dimensionality
        self.fea_dim = torch.prod(torch.tensor(input_shape[2:]))
        self.batch_size = input_shape[0]

        self.rnn = self._init_layers()

    def _init_layers(self,):
        """
        Initializes the layers of the CRNN.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim

        for i in range(self.num_layers):
            rnn_lay = CRNN_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                nonlinearity=self.nonlinearity,
                bidirectional=self.bidirectional,
                init_criterion=self.init_criterion,
                weight_init=self.weight_init,
            )

            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size

        return rnn

    def forward(self, x, hx=None):
        """Returns the output of the vanilla CRNN.

        Arguments
        ---------
        x : paddle.Tensor
        """

        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        output, hh = self._forward_rnn(x, hx=hx)

        if self.return_hidden:
            return output, hh
        else:
            return output

    def _forward_rnn(self, x, hx):
        """Returns the output of the vanilla CRNN.

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


class CRNN_Layer(paddle.nn.Layer):
    """ This function implements complex-valued recurrent layer.

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
        the complex-valued weights (default "glorot").
    weight_init : str, optional
        (complex, unitary).
        This parameter defines the initialization procedure of the
        complex-valued weights (default "complex"). "complex" will generate random complex-valued
        weights following the init_criterion and the complex polar form.
        "unitary" will normalize the weights to lie on the unit circle.
        More details in: "Deep Complex Networks", Trabelsi C. et al.
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
        weight_init="complex",
    ):

        super(CRNN_Layer, self).__init__()
        self.hidden_size = int(hidden_size) // 2  # Express in term of complex
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.init_criterion = init_criterion
        self.weight_init = weight_init

        self.w = CLinear(
            input_shape=self.input_size,
            n_neurons=self.hidden_size,
            bias=False,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
        )

        self.u = CLinear(
            input_shape=self.hidden_size * 2,  # The input size is in real
            n_neurons=self.hidden_size,
            bias=False,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
        )

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initial state
        self.register_buffer("h_init", torch.zeros(1, self.hidden_size * 2))

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

    def forward(self, x, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the output of the CRNN_layer.

        Arguments
        ---------
        x : paddle.Tensor
            Input tensor.
        """

        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        # self._change_batch_size(x)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)

        # Processing time steps
        if hx is not None:
            h = self._complexrnn_cell(w, hx)
        else:
            h = self._complexrnn_cell(w, self.h_init)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    def _complexrnn_cell(self, w, ht):
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

        self.register_buffer(
            "drop_masks",
            self.drop(torch.ones(self.N_drop_masks, self.hidden_size * 2)).data,
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
                        self.N_drop_masks, self.hidden_size * 2, device=w.device
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
                    torch.ones(self.N_drop_masks, self.hidden_size * 2)
                ).data


class CLiGRU(paddle.nn.Layer):
    """ This function implements a complex-valued Light GRU (liGRU).

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
        Specified value is in term of complex-valued neurons. Thus, the output
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
    dropout : float
        It is the dropout factor (must be between 0 and 1).
    return_hidden : bool
        If True, the function returns the last hidden layer.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used.
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
    >>> inp_tensor = torch.rand([10, 16, 30])
    >>> rnn = CLiGRU(input_shape=inp_tensor.shape, hidden_size=16)
    >>> out_tensor = rnn(inp_tensor)
    >>>
    torch.Size([4, 10, 5])
    """

    def __init__(
        self,
        hidden_size,
        input_shape,
        nonlinearity="relu",
        normalization="batchnorm",
        num_layers=1,
        bias=True,
        dropout=0.0,
        bidirectional=False,
        return_hidden=False,
        init_criterion="glorot",
        weight_init="complex",
    ):
        super().__init__()
        self.hidden_size = hidden_size * 2  # z = x + iy
        self.nonlinearity = nonlinearity
        self.num_layers = num_layers
        self.normalization = normalization
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.reshape = False
        self.return_hidden = return_hidden
        self.init_criterion = init_criterion
        self.weight_init = weight_init

        if len(input_shape) > 3:
            self.reshape = True

        self.fea_dim = torch.prod(torch.tensor(input_shape[2:]))
        self.batch_size = input_shape[0]
        self.rnn = self._init_layers()

    def _init_layers(self):
        """Initializes the layers of the liGRU.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """

        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim

        for i in range(self.num_layers):
            rnn_lay = CLiGRU_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                nonlinearity=self.nonlinearity,
                normalization=self.normalization,
                bidirectional=self.bidirectional,
                init_criterion=self.init_criterion,
                weight_init=self.weight_init,
            )
            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size
        return rnn

    def forward(self, x, hx=None):
        """Returns the output of the CliGRU.

        Arguments
        ---------
        x : paddle.Tensor
            Input tensor.
        """

        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # run ligru
        output, hh = self._forward_ligru(x, hx=hx)

        if self.return_hidden:
            return output, hh
        else:
            return output

    def _forward_ligru(self, x, hx):
        """Returns the output of the CliGRU.

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


class CLiGRU_Layer(paddle.nn.Layer):
    """
    This function implements complex-valued Light-Gated Recurrent Unit layer.

    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    batch_size : int
        Batch size of the input tensors.
    hidden_size : int
        Number of output values.
    num_layers : int
        Number of layers to employ in the RNN architecture.
    nonlinearity : str
        Type of nonlinearity (tanh, relu).
    normalization : str
        Type of normalization (batchnorm, layernorm).
        Every string different from batchnorm and layernorm will result
        in no normalization.
    dropout : float
        It is the dropout factor (must be between 0 and 1).
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used.
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
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        nonlinearity="relu",
        normalization="batchnorm",
        bidirectional=False,
        init_criterion="glorot",
        weight_init="complex",
    ):

        super(CLiGRU_Layer, self).__init__()
        self.hidden_size = int(hidden_size) // 2
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.normalization = normalization
        self.nonlinearity = nonlinearity

        self.w = CLinear(
            input_shape=self.input_size,
            n_neurons=self.hidden_size * 2,
            bias=False,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
        )

        self.u = CLinear(
            input_shape=self.hidden_size * 2,  # The input size is in real
            n_neurons=self.hidden_size * 2,
            bias=False,
            weight_init=self.weight_init,
            init_criterion=self.init_criterion,
        )

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initializing batch norm
        self.normalize = False

        if self.normalization == "batchnorm":
            self.norm = CBatchNorm(
                input_size=hidden_size * 2, dim=-1, momentum=0.05
            )
            self.normalize = True

        elif self.normalization == "layernorm":
            self.norm = CLayerNorm(input_size=hidden_size * 2, dim=-1)
            self.normalize = True
        else:
            # Normalization is disabled here. self.norm is only  formally
            # initialized to avoid jit issues.
            self.norm = CLayerNorm(input_size=hidden_size * 2, dim=-1)
            self.normalize = True

        # Initial state
        self.register_buffer("h_init", torch.zeros(1, self.hidden_size * 2))

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop(self.batch_size)

        # Initializing dropout
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)

        self.drop_mask_te = torch.tensor([1.0]).float()

        # Setting the activation function
        if self.nonlinearity == "tanh":
            self.act = torch.nn.Tanh()
        else:
            self.act = torch.nn.ReLU()

    def forward(self, x, hx=None):
        # type: (Tensor, Optional[Tensor], Optional[Bool]) -> Tensor # noqa F821
        """Returns the output of the Complex liGRU layer.

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
            h = self._complex_ligru_cell(w, hx)
        else:
            h = self._complex_ligru_cell(w, self.h_init)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    def _complex_ligru_cell(self, w, ht):
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
            atr, ati, ztr, zti = gates.chunk(4, 1)
            at = torch.cat([atr, ati], dim=-1)
            zt = torch.cat([ztr, zti], dim=-1)
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
            self.drop(torch.ones(self.N_drop_masks, self.hidden_size * 2)).data,
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
                        self.N_drop_masks, self.hidden_size * 2, device=w.device
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
                    torch.ones(self.N_drop_masks, self.hidden_size)
                ).data
