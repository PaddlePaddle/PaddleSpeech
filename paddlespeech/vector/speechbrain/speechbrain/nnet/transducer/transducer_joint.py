"""Library implementing transducer_joint.

Author
    Abdelwahab HEBA 2020
"""

import paddle
import logging
import paddle.nn as nn

logger = logging.getLogger(__name__)


class Transducer_joint(nn.Layer):
    """Computes joint tensor between Transcription network (TN) & Prediction network (PN)

    Arguments
    ---------
    joint_network : torch.class (neural network modules)
        if joint == "concat", we call this network after the concatenation of TN and PN
        if None, we don't use this network.
    joint : joint the two tensors by ("sum",or "concat") option.
    nonlinearity : torch class
        Activation function used after the joint between TN and PN
         Type of nonlinearity (tanh, relu).

    Example
    -------
    >>> from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
    >>> from speechbrain.nnet.linear import Linear
    >>> input_TN = torch.rand(8, 200, 1, 40)
    >>> input_PN = torch.rand(8, 1, 12, 40)
    >>> joint_network = Linear(input_size=80, n_neurons=80)
    >>> TJoint = Transducer_joint(joint_network, joint="concat")
    >>> output = TJoint(input_TN, input_PN)
    >>> output.shape
    torch.Size([8, 200, 12, 80])
    """

    def __init__(
        self, joint_network=None, joint="sum", nonlinearity=torch.nn.LeakyReLU
    ):
        super().__init__()
        self.joint_network = joint_network
        self.joint = joint
        self.nonlinearity = nonlinearity()

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        self.joint_network(first_input)

    def forward(self, input_TN, input_PN):
        """Returns the fusion of inputs tensors.

        Arguments
        ---------
        input_TN : paddle.Tensor
           Input from Transcription Network.

        input_PN : paddle.Tensor
           Input from Prediction Network.
        """
        if len(input_TN.shape) != len(input_PN.shape):
            raise ValueError("Arg 1 and 2 must be have same size")
        if not (len(input_TN.shape) != 4 or len(input_TN.shape) != 1):
            raise ValueError("Tensors 1 and 2 must have dim=1 or dim=4")

        if self.joint == "sum":
            joint = input_TN + input_PN

        if self.joint == "concat":
            # For training
            if len(input_TN.shape) == 4:
                dim = len(input_TN.shape) - 1
                xs = input_TN
                ymat = input_PN
                sz = [
                    max(i, j) for i, j in zip(xs.size()[:-1], ymat.size()[:-1])
                ]
                xs = xs.expand(torch.Size(sz + [xs.shape[-1]]))
                ymat = ymat.expand(torch.Size(sz + [ymat.shape[-1]]))
                joint = torch.cat((xs, ymat), dim=dim)
            # For evaluation
            elif len(input_TN.shape) == 1:
                joint = torch.cat((input_TN, input_PN), dim=0)

            if self.joint_network is not None:
                joint = self.joint_network(joint)

        return self.nonlinearity(joint)
