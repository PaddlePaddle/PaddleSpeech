"""Vanilla Neural Network for simple tests.
Authors
* Elena Rastorgueva 2020
"""
import paddle

from paddlespeech.s2t.models.wav2vec2.modules import containers
from paddlespeech.s2t.models.wav2vec2.modules import linear
from paddlespeech.s2t.models.wav2vec2.modules.normalization import BatchNorm1d


class VanillaNN(containers.Sequential):
    """A simple vanilla Deep Neural Network.
    Arguments
    ---------
    activation : paddle class
        A class used for constructing the activation layers.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int
        The number of neurons in the linear layers.
    Example
    -------
    >>> inputs = paddle.rand([10, 120, 60])
    >>> model = VanillaNN(input_shape=inputs.shape)
    >>> outputs = model(inputs)
    >>> outputs.shape
    paddle.shape([10, 120, 512])
    """

    def __init__(self,
                 input_shape,
                 dnn_blocks=2,
                 dnn_neurons=512,
                 activation=True,
                 normalization=False,
                 dropout_rate=0.0):
        super().__init__(input_shape=[None, None, input_shape])

        if not isinstance(dropout_rate, list):
            dropout_rate = [dropout_rate] * dnn_blocks
        else:
            assert len(
                dropout_rate
            ) == dnn_blocks, "len(dropout_rate) must equal to dnn_blocks"

        for block_index in range(dnn_blocks):
            self.append(
                linear.Linear,
                n_neurons=dnn_neurons,
                bias_attr=None,
                layer_name="linear", )
            if normalization:
                self.append(
                    BatchNorm1d, input_size=dnn_neurons, layer_name='bn')
            if activation:
                self.append(paddle.nn.LeakyReLU(), layer_name="act")
            self.append(
                paddle.nn.Dropout(),
                p=dropout_rate[block_index],
                layer_name='dropout')
