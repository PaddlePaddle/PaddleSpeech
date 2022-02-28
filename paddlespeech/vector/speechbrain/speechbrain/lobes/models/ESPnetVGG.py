"""This lobes replicate the encoder first introduced in ESPNET v1

source: https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/rnn/encoders.py

Authors
 * Titouan Parcollet 2020
"""
import paddle
import speechbrain as sb


class ESPnetVGG(sb.nnet.containers.Sequential):
    """This model is a combination of CNNs and RNNs following
        the ESPnet encoder. (VGG+RNN+MLP+tanh())

    Arguments
    ---------
    input_shape : tuple
        The shape of an example expected input.
    activation : torch class
        A class used for constructing the activation layers. For CNN and DNN.
    dropout : float
        Neuron dropout rate, applied to RNN only.
    cnn_channels : list of ints
        A list of the number of output channels for each CNN block.
    rnn_class : torch class
        The type of RNN to use (LiGRU, LSTM, GRU, RNN)
    rnn_layers : int
        The number of recurrent layers to include.
    rnn_neurons : int
        Number of neurons in each layer of the RNN.
    rnn_bidirectional : bool
        Whether this model will process just forward or both directions.
    projection_neurons : int
        The number of neurons in the last linear layer.

    Example
    -------
    >>> inputs = torch.rand([10, 40, 60])
    >>> model = ESPnetVGG(input_shape=inputs.shape)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 10, 512])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.ReLU,
        dropout=0.15,
        cnn_channels=[64, 128],
        rnn_class=sb.nnet.RNN.LSTM,
        rnn_layers=4,
        rnn_neurons=512,
        rnn_bidirectional=True,
        rnn_re_init=False,
        projection_neurons=512,
    ):
        super().__init__(input_shape=input_shape)

        self.append(sb.nnet.containers.Sequential, layer_name="VGG")

        self.append(
            sb.nnet.CNN.Conv2d,
            out_channels=cnn_channels[0],
            kernel_size=(3, 3),
            layer_name="conv_1_1",
        )
        self.append(activation(), layer_name="act_1_1")
        self.append(
            sb.nnet.CNN.Conv2d,
            out_channels=cnn_channels[0],
            kernel_size=(3, 3),
            layer_name="conv_1_2",
        )
        self.append(activation(), layer_name="act_1_2")
        self.append(
            sb.nnet.pooling.Pooling2d(
                pool_type="max", kernel_size=(2, 2), pool_axis=(1, 2),
            ),
            layer_name="pooling_1",
        )

        self.append(
            sb.nnet.CNN.Conv2d,
            out_channels=cnn_channels[1],
            kernel_size=(3, 3),
            layer_name="conv_2_1",
        )
        self.append(activation(), layer_name="act_2_1")
        self.append(
            sb.nnet.CNN.Conv2d,
            out_channels=cnn_channels[1],
            kernel_size=(3, 3),
            layer_name="conv_2_2",
        )
        self.append(activation(), layer_name="act_2_2")
        self.append(
            sb.nnet.pooling.Pooling2d(
                pool_type="max", kernel_size=(2, 2), pool_axis=(1, 2),
            ),
            layer_name="pooling_2",
        )

        if rnn_layers > 0:
            self.append(
                rnn_class,
                layer_name="RNN",
                hidden_size=rnn_neurons,
                num_layers=rnn_layers,
                dropout=dropout,
                bidirectional=rnn_bidirectional,
                re_init=rnn_re_init,
            )

        self.append(
            sb.nnet.linear.Linear,
            n_neurons=projection_neurons,
            layer_name="proj",
        )
        self.append(torch.nn.Tanh(), layer_name="proj_act")
