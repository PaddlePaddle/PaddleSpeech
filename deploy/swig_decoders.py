"""Wrapper for various CTC decoders in SWIG."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import swig_ctc_decoders
#import multiprocessing
from pathos.multiprocessing import Pool


def ctc_best_path_decoder(probs_seq, vocabulary):
    """Wrapper for ctc best path decoder in swig.

    :param probs_seq: 2-D list of probability distributions over each time
                      step, with each element being a list of normalized
                      probabilities over vocabulary and blank.
    :type probs_seq: 2-D list
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :return: Decoding result string.
    :rtype: basestring
    """
    return swig_ctc_decoders.ctc_best_path_decoder(probs_seq.tolist(),
                                                   vocabulary)


def ctc_beam_search_decoder(
        probs_seq,
        beam_size,
        vocabulary,
        blank_id,
        cutoff_prob=1.0,
        ext_scoring_func=None, ):
    """Wrapper for CTC Beam Search Decoder.

    :param probs_seq: 2-D list of probability distributions over each time
                      step, with each element being a list of normalized
                      probabilities over vocabulary and blank.
    :type probs_seq: 2-D list
    :param beam_size: Width for beam search.
    :type beam_size: int
    :param vocabulary: Vocabulary list.
    :type vocabulary: list
    :param blank_id: ID of blank.
    :type blank_id: int
    :param cutoff_prob: Cutoff probability in pruning,
                        default 1.0, no pruning.
    :type cutoff_prob: float
    :param ext_scoring_func: External scoring function for
                            partially decoded sentence, e.g. word count
                            or language model.
    :type external_scoring_func: callable
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    :rtype: list
    """
    return swig_ctc_decoders.ctc_beam_search_decoder(
        probs_seq.tolist(), beam_size, vocabulary, blank_id, cutoff_prob,
        ext_scoring_func)


def ctc_beam_search_decoder_batch(probs_split,
                                  beam_size,
                                  vocabulary,
                                  blank_id,
                                  num_processes,
                                  cutoff_prob=1.0,
                                  ext_scoring_func=None):
    """Wrapper for CTC beam search decoder in batch
    """

    # TODO: to resolve PicklingError

    if not num_processes > 0:
        raise ValueError("Number of processes must be positive!")

    pool = Pool(processes=num_processes)
    results = []
    args_list = []
    for i, probs_list in enumerate(probs_split):
        args = (probs_list, beam_size, vocabulary, blank_id, cutoff_prob,
                ext_scoring_func)
        args_list.append(args)
        results.append(pool.apply_async(ctc_beam_search_decoder, args))

    pool.close()
    pool.join()
    beam_search_results = [result.get() for result in results]
    """
    len_args = len(probs_split)
    beam_search_results = pool.map(ctc_beam_search_decoder,
                                  probs_split,
                                  [beam_size for i in xrange(len_args)],
                                  [vocabulary for i in xrange(len_args)],
                                  [blank_id for i in xrange(len_args)],
                                  [cutoff_prob for i in xrange(len_args)],
                                  [ext_scoring_func for i in xrange(len_args)]
                                  )
    """
    '''
    processes = [mp.Process(target=ctc_beam_search_decoder,
                args=(probs_list, beam_size, vocabulary, blank_id, cutoff_prob,
                    ext_scoring_func) for probs_list in probs_split]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    beam_search_results = []
    '''
    return beam_search_results
