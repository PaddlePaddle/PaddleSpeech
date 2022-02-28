"""The Guided Attention Loss implementation

This loss can be used to speed up the training of
models in which the correspondence between inputs and
outputs is roughly linear, and the attention alignments
are expected to be approximately diagonal, such as Grapheme-to-Phoneme
and Text-to-Speech

Authors
* Artem Ploujnikov 2021
"""

import paddle
from torch import nn


class GuidedAttentionLoss(nn.Layer):
    """
    A loss implementation that forces attention matrices to be
    near-diagonal, imposing progressively larger penalties for paying
    attention to regions far away from the diagonal). It is useful
    for sequence-to-sequence models in which the sequence of outputs
    is expected to corrsespond closely to the sequence of inputs,
    such as TTS or G2P

    https://arxiv.org/abs/1710.08969

    The implementation is inspired by the R9Y9 DeepVoice3 model
    https://github.com/r9y9/deepvoice3_pytorch

    It should be roughly equivalent to it; however, it has been
    fully vectorized.

    Arguments
    ---------
    sigma:
        the guided attention weight

    Example
    -------
    NOTE: In a real scenario, the input_lengths and
    target_lengths would come from a data batch,
    whereas alignments would come from a model
    >>> import paddle
    >>> from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
    >>> loss = GuidedAttentionLoss(sigma=0.2)
    >>> input_lengths = torch.tensor([2, 3])
    >>> target_lengths = torch.tensor([3, 4])
    >>> alignments = torch.tensor(
    ...     [
    ...         [
    ...             [0.8, 0.2, 0.0],
    ...             [0.4, 0.6, 0.0],
    ...             [0.2, 0.8, 0.0],
    ...             [0.0, 0.0, 0.0],
    ...         ],
    ...         [
    ...             [0.6, 0.2, 0.2],
    ...             [0.1, 0.7, 0.2],
    ...             [0.3, 0.4, 0.3],
    ...             [0.2, 0.3, 0.5],
    ...         ],
    ...     ]
    ... )
    >>> loss(alignments, input_lengths, target_lengths)
    tensor(0.1142)
    """

    def __init__(self, sigma=0.2):
        super().__init__()
        self.sigma = sigma
        self.weight_factor = 2 * (sigma ** 2)

    def forward(
        self,
        attention,
        input_lengths,
        target_lengths,
        max_input_len=None,
        max_target_len=None,
    ):
        """
        Computes the guided attention loss for a single batch

        Arguments
        ---------
        attention: paddle.Tensor
            A padded attention/alignments matrix
            (batch, targets, inputs)
        input_lengths: torch.tensor
            A (batch, lengths) tensor of input lengths
        target_lengths: torch.tensor
            A (batch, lengths) tensor of target lengths
        max_input_len: int
            The maximum input length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism
        max_target_len: int
            The maximum target length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism


        Returns
        -------
        loss: paddle.Tensor
            A single-element tensor with the loss value
        """
        soft_mask = self.guided_attentions(
            input_lengths, target_lengths, max_input_len, max_target_len
        )
        return (attention * soft_mask.transpose(-1, -2)).mean()

    def guided_attentions(
        self,
        input_lengths,
        target_lengths,
        max_input_len=None,
        max_target_len=None,
    ):
        """
        Computes guided attention matrices

        Arguments
        ---------
        input_lengths: paddle.Tensor
            A tensor of input lengths
        target_lengths: paddle.Tensor
            A tensor of target lengths
        max_input_len: int
            The maximum input length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism
        max_target_len: int
            The maximum target length - optional,
            if not computed will be set to the maximum
            of target_lengths. Setting it explicitly
            might be necessary when using data parallelism

        Returns
        -------
        soft_mask: paddle.Tensor
            The guided attention tensor of shape (batch, max_input_len, max_target_len)
        """
        input_lengths_broad = input_lengths.view(-1, 1, 1)
        target_lengths_broad = target_lengths.view(-1, 1, 1)
        if max_input_len is None:
            max_input_len = input_lengths.max()
        if max_target_len is None:
            max_target_len = target_lengths.max()
        input_mesh, target_mesh = torch.meshgrid(
            torch.arange(max_input_len).to(input_lengths.device),
            torch.arange(max_target_len).to(target_lengths.device),
        )
        input_mesh, target_mesh = (
            input_mesh.unsqueeze(0),
            target_mesh.unsqueeze(0),
        )
        input_lengths_broad = input_lengths.view(-1, 1, 1)
        target_lengths_broad = target_lengths.view(-1, 1, 1)
        soft_mask = 1.0 - torch.exp(
            -(
                (
                    input_mesh / input_lengths_broad
                    - target_mesh / target_lengths_broad
                )
                ** 2
            )
            / self.weight_factor
        )
        outside = (input_mesh >= input_lengths_broad) | (
            target_mesh >= target_lengths_broad
        )
        soft_mask[outside] = 0.0
        return soft_mask
