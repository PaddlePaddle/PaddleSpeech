# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from espnet(https://github.com/espnet/espnet)
"""ASR Interface module."""
import argparse

from paddlespeech.s2t.utils.dynamic_import import dynamic_import


class ASRInterface:
    """ASR Interface model implementation."""
    @staticmethod
    def add_arguments(parser):
        """Add arguments to parser."""
        return parser

    @classmethod
    def build(cls, idim: int, odim: int, **kwargs):
        """Initialize this class with python-level args.

        Args:
            idim (int): The number of an input feature dim.
            odim (int): The number of output vocab.

        Returns:
            ASRinterface: A new instance of ASRInterface.

        """
        args = argparse.Namespace(**kwargs)
        return cls(idim, odim, args)

    def forward(self, xs, ilens, ys, olens):
        """Compute loss for training.

        :param xs: batch of padded source sequences paddle.Tensor (B, Tmax, idim)
        :param ilens: batch of lengths of source sequences (B), paddle.Tensor
        :param ys: batch of padded target sequences paddle.Tensor (B, Lmax)
        :param olens: batch of lengths of target sequences (B), paddle.Tensor
        :return: loss value
        :rtype: paddle.Tensor
        """
        raise NotImplementedError("forward method is not implemented")

    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        """Recognize x for evaluation.

        :param ndarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param paddle.nn.Layer rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        raise NotImplementedError("recognize method is not implemented")

    def recognize_batch(self, x, recog_args, char_list=None, rnnlm=None):
        """Beam search implementation for batch.

        :param paddle.Tensor x: encoder hidden state sequences (B, Tmax, Henc)
        :param namespace recog_args: argument namespace containing options
        :param list char_list: list of characters
        :param paddle.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        raise NotImplementedError("Batch decoding is not supported yet.")

    def calculate_all_attentions(self, xs, ilens, ys):
        """Calculate attention.

        :param list xs: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float ndarray
        """
        raise NotImplementedError(
            "calculate_all_attentions method is not implemented")

    def calculate_all_ctc_probs(self, xs, ilens, ys):
        """Calculate CTC probability.

        :param list xs_pad: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: CTC probabilities (B, Tmax, vocab)
        :rtype: float ndarray
        """
        raise NotImplementedError(
            "calculate_all_ctc_probs method is not implemented")

    @property
    def attention_plot_class(self):
        """Get attention plot class."""
        from paddlespeech.s2t.training.extensions.plot import PlotAttentionReport

        return PlotAttentionReport

    @property
    def ctc_plot_class(self):
        """Get CTC plot class."""
        from paddlespeech.s2t.training.extensions.plot import PlotCTCReport

        return PlotCTCReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        raise NotImplementedError(
            "get_total_subsampling_factor method is not implemented")

    def encode(self, feat):
        """Encode feature in `beam_search` (optional).

        Args:
            x (numpy.ndarray): input feature (T, D)
        Returns:
            paddle.Tensor: encoded feature (T, D)
        """
        raise NotImplementedError("encode method is not implemented")

    def scorers(self):
        """Get scorers for `beam_search` (optional).

        Returns:
            dict[str, ScorerInterface]: dict of `ScorerInterface` objects

        """
        raise NotImplementedError("decoders method is not implemented")


predefined_asr = {
    "transformer": "paddlespeech.s2t.models.u2:U2Model",
    "conformer": "paddlespeech.s2t.models.u2:U2Model",
}


def dynamic_import_asr(module):
    """Import ASR models dynamically.

    Args:
        module (str): asr name. e.g., transformer, conformer

    Returns:
        type: ASR class

    """
    model_class = dynamic_import(module, predefined_asr)
    assert issubclass(model_class,
                      ASRInterface), f"{module} does not implement ASRInterface"
    return model_class
