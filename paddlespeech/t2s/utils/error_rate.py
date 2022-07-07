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
"""This module provides functions to calculate error rate in different level.
e.g. wer for word-level, cer for char-level.
"""
import numpy as np

__all__ = ['word_errors', 'char_errors', 'wer', 'cer']


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.

    Args:
        reference (str): 
            The reference sentence.
        hypothesis (str): 
            The hypothesis sentence.
        ignore_case (bool): 
            Whether case-sensitive or not.
        delimiter (char(str)): 
            Delimiter of input sentences.

    Returns:
        list: Levenshtein distance and word number of reference sentence.
    """
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = list(filter(None, reference.split(delimiter)))
    hyp_words = list(filter(None, hypothesis.split(delimiter)))

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.

    Args:
        reference (str): The reference sentence.
        hypothesis (str): The hypothesis sentence.
        ignore_case (bool): Whether case-sensitive or not.
        remove_space (bool): Whether remove internal space characters

    Returns:
        list: Levenshtein distance and length of reference sentence.
    """
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space:
        join_char = ''

    reference = join_char.join(list(filter(None, reference.split(' '))))
    hypothesis = join_char.join(list(filter(None, hypothesis.split(' '))))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.

    Args:
        reference (str): The reference sentence.
        hypothesis (str): The hypothesis sentence.
        ignore_case (bool): Whether case-sensitive or not.
        delimiter (char): Delimiter of input sentences.

    Returns: 
        float: Word error rate.

    Raises:
        ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
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
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.

    Args:
        reference (str): The reference sentence.
        hypothesis (str): The hypothesis sentence.
        ignore_case (bool): Whether case-sensitive or not.
        remove_space (bool): Whether remove internal space characters

    Returns: 
        float: Character error rate.

    Raises: 
        ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


if __name__ == "__main__":
    reference = [
        'j', 'iou4', 'zh', 'e4', 'iang5', 'x', 'v2', 'b', 'o1', 'k', 'ai1',
        'sh', 'iii3', 'l', 'e5', 'b', 'ei3', 'p', 'iao1', 'sh', 'eng1', 'ia2'
    ]
    hypothesis = [
        'j', 'iou4', 'zh', 'e4', 'iang4', 'x', 'v2', 'b', 'o1', 'k', 'ai1',
        'sh', 'iii3', 'l', 'e5', 'b', 'ei3', 'p', 'iao1', 'sh', 'eng1', 'ia2'
    ]
    reference = " ".join(reference)
    hypothesis = " ".join(hypothesis)
    print(wer(reference, hypothesis))
