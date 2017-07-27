"""Contains various CTC decoders in SWIG."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from swig_ctc_beam_search_decoder import ctc_beam_search_decoder as beam_search_decoder
from swig_ctc_beam_search_decoder import ctc_best_path_decoder as best_path__decoder


def ctc_best_path_decoder(probs_seq, vocabulary):
    best_path__decoder(probs_seq.to_list(), vocabulary)


def ctc_beam_search_decoder(
        probs_seq,
        beam_size,
        vocabulary,
        blank_id,
        cutoff_prob=1.0,
        ext_scoring_func=None, ):
    beam_search_decoder(probs_seq.to_list(), beam_size, vocabulary, blank_id,
                        cutoff_prob, ext_scoring_func)
