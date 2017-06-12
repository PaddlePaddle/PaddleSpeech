# -*- coding: utf-8 -*-
"""
    This module provides functions to calculate error rate in different level.
    e.g. wer for word-level, cer for char-level.
"""

import numpy as np


def levenshtein_distance(ref, hyp):
    ref_len = len(ref)
    hyp_len = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if ref_len == 0:
        return hyp_len
    if hyp_len == 0:
        return ref_len

    distance = np.zeros((ref_len + 1, hyp_len + 1), dtype=np.int32)

    # initialize distance matrix
    for j in xrange(hyp_len + 1):
        distance[0][j] = j
    for i in xrange(ref_len + 1):
        distance[i][0] = i

    # calculate levenshtein distance
    for i in xrange(1, ref_len + 1):
        for j in xrange(1, hyp_len + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                s_num = distance[i - 1][j - 1] + 1
                i_num = distance[i][j - 1] + 1
                d_num = distance[i - 1][j] + 1
                distance[i][j] = min(s_num, i_num, d_num)

    return distance[ref_len][hyp_len]


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """
    Calculate word error rate (WER). WER compares reference text and 
    hypothesis text in word-level. WER is defined as:

    .. math::
        WER = (Sw + Dw + Iw) / Nw

    where

    .. code-block:: text

        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference

    We can use levenshtein distance to calculate WER. Please draw an attention that 
    empty items will be removed when splitting sentences by delimiter.

    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = filter(None, reference.split(delimiter))
    hyp_words = filter(None, hypothesis.split(delimiter))

    if len(ref_words) == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    edit_distance = levenshtein_distance(ref_words, hyp_words)
    wer = float(edit_distance) / len(ref_words)
    return wer


def cer(reference, hypothesis, ignore_case=False):
    """
    Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:

    .. math::
        CER = (Sc + Dc + Ic) / Nc

    where

    .. code-block:: text

        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference

    We can use levenshtein distance to calculate CER. Chinese input should be 
    encoded to unicode. Please draw an attention that the leading and tailing 
    white space characters will be truncated and multiple consecutive white 
    space characters in a sentence will be replaced by one white space character.

    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :return: Character error rate.
    :rtype: float
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    reference = ' '.join(filter(None, reference.split(' ')))
    hypothesis = ' '.join(filter(None, hypothesis.split(' ')))

    if len(reference) == 0:
        raise ValueError("Length of reference should be greater than 0.")

    edit_distance = levenshtein_distance(reference, hypothesis)
    cer = float(edit_distance) / len(reference)
    return cer
