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
"""Wrapper for various CTC decoders in SWIG."""
import paddlespeech_ctcdecoders


class Scorer(paddlespeech_ctcdecoders.Scorer):
    """Wrapper for Scorer.

    :param alpha: Parameter associated with language model. Don't use
                  language model when alpha = 0.
    :type alpha: float
    :param beta: Parameter associated with word count. Don't use word
                 count when beta = 0.
    :type beta: float
    :model_path: Path to load language model.
    :type model_path: str
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    """

    def __init__(self, alpha, beta, model_path, vocabulary):
        paddlespeech_ctcdecoders.Scorer.__init__(self, alpha, beta, model_path,
                                                 vocabulary)


def ctc_greedy_decoding(probs_seq, vocabulary, blank_id):
    """Wrapper for ctc best path decodeing function in swig.

    :param probs_seq: 2-D list of probability distributions over each time
                      step, with each element being a list of normalized
                      probabilities over vocabulary and blank.
    :type probs_seq: 2-D list
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :return: Decoding result string.
    :rtype: str
    """
    result = paddlespeech_ctcdecoders.ctc_greedy_decoding(probs_seq.tolist(),
                                                          vocabulary, blank_id)
    return result


def ctc_beam_search_decoding(probs_seq,
                             vocabulary,
                             beam_size,
                             cutoff_prob=1.0,
                             cutoff_top_n=40,
                             ext_scoring_func=None,
                             blank_id=0):
    """Wrapper for the CTC Beam Search Decoding function.

    :param probs_seq: 2-D list of probability distributions over each time
                      step, with each element being a list of normalized
                      probabilities over vocabulary and blank.
    :type probs_seq: 2-D list
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param cutoff_prob: Cutoff probability in pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
    :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                         characters with highest probs in vocabulary will be
                         used in beam search, default 40.
    :type cutoff_top_n: int
    :param ext_scoring_func: External scoring function for
                             partially decoded sentence, e.g. word count
                             or language model.
    :type external_scoring_func: callable
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    :rtype: list
    """
    beam_results = paddlespeech_ctcdecoders.ctc_beam_search_decoding(
        probs_seq.tolist(), vocabulary, beam_size, cutoff_prob, cutoff_top_n,
        ext_scoring_func, blank_id)
    beam_results = [(res[0], res[1].decode('utf-8')) for res in beam_results]
    return beam_results


def ctc_beam_search_decoding_batch(probs_split,
                                   vocabulary,
                                   beam_size,
                                   num_processes,
                                   cutoff_prob=1.0,
                                   cutoff_top_n=40,
                                   ext_scoring_func=None,
                                   blank_id=0):
    """Wrapper for the batched CTC beam search decodeing batch function.

    :param probs_seq: 3-D list with each element as an instance of 2-D list
                      of probabilities used by ctc_beam_search_decoder().
    :type probs_seq: 3-D list
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param num_processes: Number of parallel processes.
    :type num_processes: int
    :param cutoff_prob: Cutoff probability in vocabulary pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
    :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                         characters with highest probs in vocabulary will be
                         used in beam search, default 40.
    :type cutoff_top_n: int
    :param num_processes: Number of parallel processes.
    :type num_processes: int
    :param ext_scoring_func: External scoring function for
                             partially decoded sentence, e.g. word count
                             or language model.
    :type external_scoring_function: callable
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    :rtype: list
    """
    probs_split = [probs_seq.tolist() for probs_seq in probs_split]

    batch_beam_results = paddlespeech_ctcdecoders.ctc_beam_search_decoding_batch(
        probs_split, vocabulary, beam_size, num_processes, cutoff_prob,
        cutoff_top_n, ext_scoring_func, blank_id)
    batch_beam_results = [[(res[0], res[1]) for res in beam_results]
                          for beam_results in batch_beam_results]
    return batch_beam_results


class CTCBeamSearchDecoder(paddlespeech_ctcdecoders.CtcBeamSearchDecoderBatch):
    """Wrapper for CtcBeamSearchDecoderBatch.
    Args:
        vocab_list (list): Vocabulary list.
        beam_size (int): Width for beam search.
        num_processes (int): Number of parallel processes.
        param cutoff_prob (float): Cutoff probability in vocabulary pruning,
                            default 1.0, no pruning.
        cutoff_top_n (int): Cutoff number in pruning, only top cutoff_top_n
                            characters with highest probs in vocabulary will be
                            used in beam search, default 40.
        param ext_scorer (Scorer): External scorer for partially decoded sentence, e.g. word count
                                or language model.
    """

    def __init__(self, vocab_list, batch_size, beam_size, num_processes,
                 cutoff_prob, cutoff_top_n, _ext_scorer, blank_id):
        paddlespeech_ctcdecoders.CtcBeamSearchDecoderBatch.__init__(
            self, vocab_list, batch_size, beam_size, num_processes, cutoff_prob,
            cutoff_top_n, _ext_scorer, blank_id)
