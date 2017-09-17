"""Contains various CTC decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import groupby
import numpy as np
from math import log
import multiprocessing


def ctc_greedy_decoder(probs_seq, vocabulary):
    """CTC greedy (best path) decoder.

    Path consisting of the most probable tokens are further post-processed to
    remove consecutive repetitions and all blanks.

    :param probs_seq: 2-D list of probabilities over the vocabulary for each
                      character. Each element is a list of float probabilities
                      for one character.
    :type probs_seq: list
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :return: Decoding result string.
    :rtype: baseline
    """
    # dimension verification
    for probs in probs_seq:
        if not len(probs) == len(vocabulary) + 1:
            raise ValueError("probs_seq dimension mismatchedd with vocabulary")
    # argmax to get the best index for each time step
    max_index_list = list(np.array(probs_seq).argmax(axis=1))
    # remove consecutive duplicate indexes
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    # remove blank indexes
    blank_index = len(vocabulary)
    index_list = [index for index in index_list if index != blank_index]
    # convert index list to string
    return ''.join([vocabulary[index] for index in index_list])


def ctc_beam_search_decoder(probs_seq,
                            beam_size,
                            vocabulary,
                            cutoff_prob=1.0,
                            cutoff_top_n=40,
                            ext_scoring_func=None,
                            nproc=False):
    """CTC Beam search decoder.

    It utilizes beam search to approximately select top best decoding
    labels and returning results in the descending order.
    The implementation is based on Prefix Beam Search
    (https://arxiv.org/abs/1408.2873), and the unclear part is
    redesigned. Two important modifications: 1) in the iterative computation
    of probabilities, the assignment operation is changed to accumulation for
    one prefix may comes from different paths; 2) the if condition "if l^+ not
    in A_prev then" after probabilities' computation is deprecated for it is
    hard to understand and seems unnecessary.

    :param probs_seq: 2-D list of probability distributions over each time
                      step, with each element being a list of normalized
                      probabilities over vocabulary and blank.
    :type probs_seq: 2-D list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param cutoff_prob: Cutoff probability in pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
    :param ext_scoring_func: External scoring function for
                            partially decoded sentence, e.g. word count
                            or language model.
    :type external_scoring_func: callable
    :param nproc: Whether the decoder used in multiprocesses.
    :type nproc: bool
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    :rtype: list
    """
    # dimension check
    for prob_list in probs_seq:
        if not len(prob_list) == len(vocabulary) + 1:
            raise ValueError("The shape of prob_seq does not match with the "
                             "shape of the vocabulary.")

    # blank_id assign
    blank_id = len(vocabulary)

    # If the decoder called in the multiprocesses, then use the global scorer
    # instantiated in ctc_beam_search_decoder_batch().
    if nproc is True:
        global ext_nproc_scorer
        ext_scoring_func = ext_nproc_scorer

    ## initialize
    # prefix_set_prev: the set containing selected prefixes
    # probs_b_prev: prefixes' probability ending with blank in previous step
    # probs_nb_prev: prefixes' probability ending with non-blank in previous step
    prefix_set_prev = {'\t': 1.0}
    probs_b_prev, probs_nb_prev = {'\t': 1.0}, {'\t': 0.0}

    ## extend prefix in loop
    for time_step in xrange(len(probs_seq)):
        # prefix_set_next: the set containing candidate prefixes
        # probs_b_cur: prefixes' probability ending with blank in current step
        # probs_nb_cur: prefixes' probability ending with non-blank in current step
        prefix_set_next, probs_b_cur, probs_nb_cur = {}, {}, {}

        prob_idx = list(enumerate(probs_seq[time_step]))
        cutoff_len = len(prob_idx)
        #If pruning is enabled
        if cutoff_prob < 1.0 or cutoff_top_n < cutoff_len:
            prob_idx = sorted(prob_idx, key=lambda asd: asd[1], reverse=True)
            cutoff_len, cum_prob = 0, 0.0
            for i in xrange(len(prob_idx)):
                cum_prob += prob_idx[i][1]
                cutoff_len += 1
                if cum_prob >= cutoff_prob:
                    break
            cutoff_len = min(cutoff_len, cutoff_top_n)
            prob_idx = prob_idx[0:cutoff_len]

        for l in prefix_set_prev:
            if not prefix_set_next.has_key(l):
                probs_b_cur[l], probs_nb_cur[l] = 0.0, 0.0

            # extend prefix by travering prob_idx
            for index in xrange(cutoff_len):
                c, prob_c = prob_idx[index][0], prob_idx[index][1]

                if c == blank_id:
                    probs_b_cur[l] += prob_c * (
                        probs_b_prev[l] + probs_nb_prev[l])
                else:
                    last_char = l[-1]
                    new_char = vocabulary[c]
                    l_plus = l + new_char
                    if not prefix_set_next.has_key(l_plus):
                        probs_b_cur[l_plus], probs_nb_cur[l_plus] = 0.0, 0.0

                    if new_char == last_char:
                        probs_nb_cur[l_plus] += prob_c * probs_b_prev[l]
                        probs_nb_cur[l] += prob_c * probs_nb_prev[l]
                    elif new_char == ' ':
                        if (ext_scoring_func is None) or (len(l) == 1):
                            score = 1.0
                        else:
                            prefix = l[1:]
                            score = ext_scoring_func(prefix)
                        probs_nb_cur[l_plus] += score * prob_c * (
                            probs_b_prev[l] + probs_nb_prev[l])
                    else:
                        probs_nb_cur[l_plus] += prob_c * (
                            probs_b_prev[l] + probs_nb_prev[l])
                    # add l_plus into prefix_set_next
                    prefix_set_next[l_plus] = probs_nb_cur[
                        l_plus] + probs_b_cur[l_plus]
            # add l into prefix_set_next
            prefix_set_next[l] = probs_b_cur[l] + probs_nb_cur[l]
        # update probs
        probs_b_prev, probs_nb_prev = probs_b_cur, probs_nb_cur

        ## store top beam_size prefixes
        prefix_set_prev = sorted(
            prefix_set_next.iteritems(), key=lambda asd: asd[1], reverse=True)
        if beam_size < len(prefix_set_prev):
            prefix_set_prev = prefix_set_prev[:beam_size]
        prefix_set_prev = dict(prefix_set_prev)

    beam_result = []
    for seq, prob in prefix_set_prev.items():
        if prob > 0.0 and len(seq) > 1:
            result = seq[1:]
            # score last word by external scorer
            if (ext_scoring_func is not None) and (result[-1] != ' '):
                prob = prob * ext_scoring_func(result)
            log_prob = log(prob)
            beam_result.append((log_prob, result))
        else:
            beam_result.append((float('-inf'), ''))

    ## output top beam_size decoding results
    beam_result = sorted(beam_result, key=lambda asd: asd[0], reverse=True)
    return beam_result


def ctc_beam_search_decoder_batch(probs_split,
                                  beam_size,
                                  vocabulary,
                                  num_processes,
                                  cutoff_prob=1.0,
                                  cutoff_top_n=40,
                                  ext_scoring_func=None):
    """CTC beam search decoder using multiple processes.

    :param probs_seq: 3-D list with each element as an instance of 2-D list
                      of probabilities used by ctc_beam_search_decoder().
    :type probs_seq: 3-D list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param num_processes: Number of parallel processes.
    :type num_processes: int
    :param cutoff_prob: Cutoff probability in pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
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
    if not num_processes > 0:
        raise ValueError("Number of processes must be positive!")

    # use global variable to pass the externnal scorer to beam search decoder
    global ext_nproc_scorer
    ext_nproc_scorer = ext_scoring_func
    nproc = True

    pool = multiprocessing.Pool(processes=num_processes)
    results = []
    for i, probs_list in enumerate(probs_split):
        args = (probs_list, beam_size, vocabulary, cutoff_prob, cutoff_top_n,
                None, nproc)
        results.append(pool.apply_async(ctc_beam_search_decoder, args))

    pool.close()
    pool.join()
    beam_search_results = [result.get() for result in results]
    return beam_search_results
