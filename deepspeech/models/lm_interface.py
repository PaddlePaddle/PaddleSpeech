"""Language model interface."""

import argparse

from deepspeech.decoders.scorers.scorer_interface import ScorerInterface
from deepspeech.utils.dynamic_import import dynamic_import

class LMInterface(ScorerInterface):
    """LM Interface model implementation."""

    @staticmethod
    def add_arguments(parser):
        """Add arguments to command line argument parser."""
        return parser

    @classmethod
    def build(cls, n_vocab: int, **kwargs):
        """Initialize this class with python-level args.

        Args:
            idim (int): The number of vocabulary.

        Returns:
            LMinterface: A new instance of LMInterface.

        """
        args = argparse.Namespace(**kwargs)
        return cls(n_vocab, args)

    def forward(self, x, t):
        """Compute LM loss value from buffer sequences.

        Args:
            x (torch.Tensor): Input ids. (batch, len)
            t (torch.Tensor): Target ids. (batch, len)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of
                loss to backward (scalar),
                negative log-likelihood of t: -log p(t) (scalar) and
                the number of elements in x (scalar)

        Notes:
            The last two return values are used
            in perplexity: p(t)^{-n} = exp(-log p(t) / n)

        """
        raise NotImplementedError("forward method is not implemented")


predefined_lms = {
    "transformer": "deepspeech.models.lm.transformer:TransformerLM",
}

def dynamic_import_lm(module):
    """Import LM class dynamically.

    Args:
        module (str): module_name:class_name or alias in `predefined_lms`

    Returns:
        type: LM class

    """
    model_class = dynamic_import(module, predefined_lms)
    assert issubclass(
        model_class, LMInterface
    ), f"{module} does not implement LMInterface"
    return model_class
