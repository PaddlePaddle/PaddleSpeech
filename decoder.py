"""
    CTC-like decoder utilitis.
"""

import os
from itertools import groupby
import numpy as np
import copy
import kenlm
import multiprocessing


def ctc_best_path_decode(probs_seq, vocabulary):
    """
    Best path decoding, also called argmax decoding or greedy decoding.
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


class Scorer(object):
    """
    External defined scorer to evaluate a sentence in beam search
               decoding, consisting of language model and word count.

    :param alpha: Parameter associated with language model.
    :type alpha: float
    :param beta: Parameter associated with word count.
    :type beta: float
    :model_path: Path to load language model.
    :type model_path: basestring
    """

    def __init__(self, alpha, beta, model_path):
        self._alpha = alpha
        self._beta = beta
        if not os.path.isfile(model_path):
            raise IOError("Invaid language model path: %s" % model_path)
        self._language_model = kenlm.LanguageModel(model_path)

    # n-gram language model scoring
    def language_model_score(self, sentence):
        #log prob of last word
        log_cond_prob = list(
            self._language_model.full_scores(sentence, eos=False))[-1][0]
        return np.power(10, log_cond_prob)

    # word insertion term
    def word_count(self, sentence):
        words = sentence.strip().split(' ')
        return len(words)

    # execute evaluation
    def evaluate(self, sentence):
        lm = self.language_model_score(sentence)
        word_cnt = self.word_count(sentence)
        score = np.power(lm, self._alpha) \
                * np.power(word_cnt, self._beta)
        return score


def ctc_beam_search_decoder(probs_seq,
                            beam_size,
                            vocabulary,
                            ext_scoring_func=None,
                            blank_id=0):
    '''
    Beam search decoder for CTC-trained network, using beam search with width
    beam_size to find many paths to one label, return  beam_size labels in
    the order of probabilities. The implementation is based on Prefix Beam
    Search(https://arxiv.org/abs/1408.2873), and the unclear part is
    redesigned, need to be verified.

    :param probs_seq: 2-D list with length num_time_steps, each element
                      is a list of normalized probabilities over vocabulary
                      and blank for one time step.
    :type probs_seq: 2-D list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param ext_scoring_func: External defined scoring function for
                            partially decoded sentence, e.g. word count
                            and language model.
    :type external_scoring_function: function
    :param blank_id: id of blank, default 0.
    :type blank_id: int
    :return: Decoding log probability and result string.
    :rtype: list

    '''
    # dimension check
    for prob_list in probs_seq:
        if not len(prob_list) == len(vocabulary) + 1:
            raise ValueError("probs dimension mismatchedd with vocabulary")
    num_time_steps = len(probs_seq)

    # blank_id check
    probs_dim = len(probs_seq[0])
    if not blank_id < probs_dim:
        raise ValueError("blank_id shouldn't be greater than probs dimension")

    ## initialize
    # the set containing selected prefixes
    prefix_set_prev = {'\t': 1.0}
    probs_b_prev, probs_nb_prev = {'\t': 1.0}, {'\t': 0.0}

    ## extend prefix in loop
    for time_step in range(num_time_steps):
        # the set containing candidate prefixes
        prefix_set_next = {}
        probs_b_cur, probs_nb_cur = {}, {}
        for l in prefix_set_prev:
            prob = probs_seq[time_step]
            if not prefix_set_next.has_key(l):
                probs_b_cur[l], probs_nb_cur[l] = 0.0, 0.0

            # extend prefix by travering vocabulary
            for c in range(0, probs_dim):
                if c == blank_id:
                    probs_b_cur[l] += prob[c] * (
                        probs_b_prev[l] + probs_nb_prev[l])
                else:
                    last_char = l[-1]
                    new_char = vocabulary[c]
                    l_plus = l + new_char
                    if not prefix_set_next.has_key(l_plus):
                        probs_b_cur[l_plus], probs_nb_cur[l_plus] = 0.0, 0.0

                    if new_char == last_char:
                        probs_nb_cur[l_plus] += prob[c] * probs_b_prev[l]
                        probs_nb_cur[l] += prob[c] * probs_nb_prev[l]
                    elif new_char == ' ':
                        if (ext_scoring_func is None) or (len(l) == 1):
                            score = 1.0
                        else:
                            prefix = l[1:]
                            score = ext_scoring_func(prefix)
                        probs_nb_cur[l_plus] += score * prob[c] * (
                            probs_b_prev[l] + probs_nb_prev[l])
                    else:
                        probs_nb_cur[l_plus] += prob[c] * (
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
    for (seq, prob) in prefix_set_prev.items():
        if prob > 0.0:
            result = seq[1:]
            log_prob = np.log(prob)
            beam_result.append([log_prob, result])

    ## output top beam_size decoding results
    beam_result = sorted(beam_result, key=lambda asd: asd[0], reverse=True)
    return beam_result


def ctc_beam_search_decoder_nproc(probs_split,
                                  beam_size,
                                  vocabulary,
                                  ext_scoring_func=None,
                                  blank_id=0,
                                  num_processes=None):
    '''
    Beam search decoder using multiple processes.

    :param probs_seq: 3-D list with length batch_size, each element
                      is a 2-D list of  probabilities can be used by
                      ctc_beam_search_decoder.

    :type probs_seq: 3-D list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param ext_scoring_func: External defined scoring function for
                            partially decoded sentence, e.g. word count
                            and language model.
    :type external_scoring_function: function
    :param blank_id: id of blank, default 0.
    :type blank_id: int
    :param num_processes: Number of processes, default None, equal to the
                 number of CPUs.
    :type num_processes: int
    :return: Decoding log probability and result string.
    :rtype: list

    '''

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    if not num_processes > 0:
        raise ValueError("Number of processes must be positive!")

    pool = multiprocessing.Pool(processes=num_processes)
    results = []
    for i, probs_list in enumerate(probs_split):
        args = (probs_list, beam_size, vocabulary, ext_scoring_func, blank_id)
        results.append(pool.apply_async(ctc_beam_search_decoder, args))

    pool.close()
    pool.join()
    beam_search_results = []
    for result in results:
        beam_search_results.append(result.get())
    return beam_search_results
