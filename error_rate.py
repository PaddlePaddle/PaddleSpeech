# -- * -- coding: utf-8 -- * --
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

    distance = np.zeros((ref_len + 1) * (hyp_len + 1), dtype=np.int64)
    distance = distance.reshape((ref_len + 1, hyp_len + 1))

    # initialization distance matrix
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


def wer(reference, hypophysis, delimiter=' ', filter_none=True):
    """
    Calculate word error rate (WER). WER is a popular evaluation metric used
    in speech recognition. It compares a reference to an hypophysis and
    is defined like this:

    .. math::
        WER = (Sw + Dw + Iw) / Nw

    where

    .. code-block:: text

        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference

    We can use levenshtein distance to calculate WER. Take an attention that 
    this function will truncate the beginning and ending delimiter for 
    reference and hypophysis sentences before calculating WER.

    :param reference: The reference sentence.
    :type reference: str
    :param hypophysis: The hypophysis sentence.
    :type reference: str
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :param filter_none: Whether to remove None value when splitting sentence.
    :type filter_none: bool
    :return: WER
    :rtype: float
    """

    if len(reference.strip(delimiter)) == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    if filter_none == True:
        ref_words = filter(None, reference.strip(delimiter).split(delimiter))
        hyp_words = filter(None, hypophysis.strip(delimiter).split(delimiter))
    else:
        ref_words = reference.strip(delimiter).split(delimiter)
        hyp_words = reference.strip(delimiter).split(delimiter)

    edit_distance = levenshtein_distance(ref_words, hyp_words)
    wer = float(edit_distance) / len(ref_words)
    return wer


def cer(reference, hypophysis, squeeze=True, ignore_case=False, strip_char=''):
    """
    Calculate charactor error rate (CER). CER will compare reference text and
    hypophysis text in char-level. CER is defined as:

    .. math::
        CER = (Sc + Dc + Ic) / Nc

    where

    .. code-block:: text

        Sc is the number of character substituted,
        Dc is the number of deleted,
        Ic is the number of inserted
        Nc is the number of characters in the reference

    We can use levenshtein distance to calculate CER. Chinese input should be 
    encoded to unicode.

    :param reference: The reference sentence.
    :type reference: str
    :param hypophysis: The hypophysis sentence.
    :type reference: str
    :param squeeze: If set true, consecutive space character 
    will be squeezed to one
    :type squeezed: bool
    :param ignore_case: Whether ignoring character case.
    :type ignore_case: bool
    :param strip_char: If not set to '', strip_char in beginning and ending of
    sentence will be truncated.
    :type strip_char: char
    :return: CER
    :rtype: float
    """
    if ignore_case == True:
        reference = reference.lower()
        hypophysis = hypophysis.lower()
    if strip_char != '':
        reference = reference.strip(strip_char)
        hypophysis = hypophysis.strip(strip_char)
    if squeeze == True:
        reference = ' '.join(filter(None, reference.split(' ')))
        hypophysis = ' '.join(filter(None, hypophysis.split(' ')))

    if len(reference) == 0:
        raise ValueError("Length of reference should be greater than 0.")
    edit_distance = levenshtein_distance(reference, hypophysis)
    cer = float(edit_distance) / len(reference)
    return cer
