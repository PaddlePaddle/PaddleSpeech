"""Implementation of a Recurrent Language Model.

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Ju-Chieh Chou 2020
 * Titouan Parcollet 2020
 * Abdel 2020
"""
import paddle
from torch import nn
import speechbrain as sb


class RNNLM(nn.Layer):
    """This model is a combination of embedding layer, RNN, DNN.
    It can be used for RNNLM.

    Arguments
    ---------
    output_neurons : int
        Number of entries in embedding table, also the number of neurons in
        output layer.
    embedding_dim : int
        Size of embedding vectors (default 128).
    activation : torch class
        A class used for constructing the activation layers for DNN.
    dropout : float
        Neuron dropout rate applied to embedding, RNN, and DNN.
    rnn_class : torch class
        The type of RNN to use in RNNLM network (LiGRU, LSTM, GRU, RNN)
    rnn_layers : int
        The number of recurrent layers to include.
    rnn_neurons : int
        Number of neurons in each layer of the RNN.
    rnn_re_init : bool
        Whether to initialize rnn with orthogonal initialization.
    rnn_return_hidden : bool
        Whether to return hidden states (default True).
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int
        The number of neurons in the linear layers.

    Example
    -------
    >>> model = RNNLM(output_neurons=5)
    >>> inputs = paddle.Tensor([[1, 2, 3]])
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([1, 3, 5])
    """

    def __init__(
        self,
        output_neurons,
        embedding_dim=128,
        activation=torch.nn.LeakyReLU,
        dropout=0.15,
        rnn_class=sb.nnet.RNN.LSTM,
        rnn_layers=2,
        rnn_neurons=1024,
        rnn_re_init=False,
        return_hidden=False,
        dnn_blocks=1,
        dnn_neurons=512,
    ):
        super().__init__()
        self.embedding = sb.nnet.embedding.Embedding(
            num_embeddings=output_neurons, embedding_dim=embedding_dim
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=dropout,
            re_init=rnn_re_init,
        )
        self.return_hidden = return_hidden
        self.reshape = False

        self.dnn = sb.nnet.containers.Sequential(
            input_shape=[None, None, rnn_neurons]
        )
        for block_index in range(dnn_blocks):
            self.dnn.append(
                sb.nnet.linear.Linear,
                n_neurons=dnn_neurons,
                bias=True,
                layer_name="linear",
            )
            self.dnn.append(sb.nnet.normalization.LayerNorm, layer_name="norm")
            self.dnn.append(activation(), layer_name="act")
            self.dnn.append(torch.nn.Dropout(p=dropout), layer_name="dropout")

        self.out = sb.nnet.linear.Linear(
            input_size=dnn_neurons, n_neurons=output_neurons
        )

    def forward(self, x, hx=None):

        x = self.embedding(x)
        x = self.dropout(x)

        # If 2d tensor, add a time-axis
        # This is used for inference time
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
            self.reshape = True

        x, hidden = self.rnn(x, hx)
        x = self.dnn(x)
        out = self.out(x)

        if self.reshape:
            out = out.squeeze(dim=1)

        if self.return_hidden:
            return out, hidden
        else:
            return out
