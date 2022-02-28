"""Edit distance and WER computation.

Authors
 * Aku Rouhe 2020
"""
import collections

EDIT_SYMBOLS = {
    "eq": "=",  # when tokens are equal
    "ins": "I",
    "del": "D",
    "sub": "S",
}


# NOTE: There is a danger in using mutables as default arguments, as they are
# only initialized once, and not every time the function is run. However,
# here the default is not actually ever mutated,
# and simply serves as an empty Counter.
def accumulatable_wer_stats(refs, hyps, stats=collections.Counter()):
    """Computes word error rate and the related counts for a batch.

    Can also be used to accumulate the counts over many batches, by passing
    the output back to the function in the call for the next batch.

    Arguments
    ----------
    ref : iterable
        Batch of reference sequences.
    hyp : iterable
        Batch of hypothesis sequences.
    stats : collections.Counter
        The running statistics.
        Pass the output of this function back as this parameter
        to accumulate the counts. It may be cleanest to initialize
        the stats yourself; then an empty collections.Counter() should
        be used.

    Returns
    -------
    collections.Counter
        The updated running statistics, with keys:

        * "WER" - word error rate
        * "insertions" - number of insertions
        * "deletions" - number of deletions
        * "substitutions" - number of substitutions
        * "num_ref_tokens" - number of reference tokens

    Example
    -------
    >>> import collections
    >>> batches = [[[[1,2,3],[4,5,6]], [[1,2,4],[5,6]]],
    ...             [[[7,8], [9]],     [[7,8],  [10]]]]
    >>> stats = collections.Counter()
    >>> for batch in batches:
    ...     refs, hyps = batch
    ...     stats = accumulatable_wer_stats(refs, hyps, stats)
    >>> print("%WER {WER:.2f}, {num_ref_tokens} ref tokens".format(**stats))
    %WER 33.33, 9 ref tokens
    """
    updated_stats = stats + _batch_stats(refs, hyps)
    if updated_stats["num_ref_tokens"] == 0:
        updated_stats["WER"] = float("nan")
    else:
        num_edits = sum(
            [
                updated_stats["insertions"],
                updated_stats["deletions"],
                updated_stats["substitutions"],
            ]
        )
        updated_stats["WER"] = (
            100.0 * num_edits / updated_stats["num_ref_tokens"]
        )
    return updated_stats


def _batch_stats(refs, hyps):
    """Internal function which actually computes the counts.

    Used by accumulatable_wer_stats

    Arguments
    ----------
    ref : iterable
        Batch of reference sequences.
    hyp : iterable
        Batch of hypothesis sequences.

    Returns
    -------
    collections.Counter
        Edit statistics over the batch, with keys:

        * "insertions" - number of insertions
        * "deletions" - number of deletions
        * "substitutions" - number of substitutions
        * "num_ref_tokens" - number of reference tokens

    Example
    -------
    >>> from speechbrain.utils.edit_distance import _batch_stats
    >>> batch = [[[1,2,3],[4,5,6]], [[1,2,4],[5,6]]]
    >>> refs, hyps = batch
    >>> print(_batch_stats(refs, hyps))
    Counter({'num_ref_tokens': 6, 'substitutions': 1, 'deletions': 1})
    """
    if len(refs) != len(hyps):
        raise ValueError(
            "The reference and hypothesis batches are not of the same size"
        )
    stats = collections.Counter()
    for ref_tokens, hyp_tokens in zip(refs, hyps):
        table = op_table(ref_tokens, hyp_tokens)
        edits = count_ops(table)
        stats += edits
        stats["num_ref_tokens"] += len(ref_tokens)
    return stats


def op_table(a, b):
    """Table of edit operations between a and b.

    Solves for the table of edit operations, which is mainly used to
    compute word error rate. The table is of size ``[|a|+1, |b|+1]``,
    and each point ``(i, j)`` in the table has an edit operation. The
    edit operations can be deterministically followed backwards to
    find the shortest edit path to from ``a[:i-1] to b[:j-1]``. Indexes
    of zero (``i=0`` or ``j=0``) correspond to an empty sequence.

    The algorithm itself is well known, see

    `Levenshtein distance <https://en.wikipedia.org/wiki/Levenshtein_distance>`_

    Note that in some cases there are multiple valid edit operation
    paths which lead to the same edit distance minimum.

    Arguments
    ---------
    a : iterable
        Sequence for which the edit operations are solved.
    b : iterable
        Sequence for which the edit operations are solved.

    Returns
    -------
    list
        List of lists, Matrix, Table of edit operations.

    Example
    -------
    >>> ref = [1,2,3]
    >>> hyp = [1,2,4]
    >>> for row in op_table(ref, hyp):
    ...     print(row)
    ['=', 'I', 'I', 'I']
    ['D', '=', 'I', 'I']
    ['D', 'D', '=', 'I']
    ['D', 'D', 'D', 'S']
    """
    # For the dynamic programming algorithm, only two rows are really needed:
    # the one currently being filled in, and the previous one
    # The following is also the right initialization
    prev_row = [j for j in range(len(b) + 1)]
    curr_row = [0] * (len(b) + 1)  # Just init to zero
    # For the edit operation table we will need the whole matrix.
    # We will initialize the table with no-ops, so that we only need to change
    # where an edit is made.
    table = [
        [EDIT_SYMBOLS["eq"] for j in range(len(b) + 1)]
        for i in range(len(a) + 1)
    ]
    # We already know the operations on the first row and column:
    for i in range(len(a) + 1):
        table[i][0] = EDIT_SYMBOLS["del"]
    for j in range(len(b) + 1):
        table[0][j] = EDIT_SYMBOLS["ins"]
    table[0][0] = EDIT_SYMBOLS["eq"]
    # The rest of the table is filled in row-wise:
    for i, a_token in enumerate(a, start=1):
        curr_row[0] += 1  # This trick just deals with the first column.
        for j, b_token in enumerate(b, start=1):
            # The dynamic programming algorithm cost rules
            insertion_cost = curr_row[j - 1] + 1
            deletion_cost = prev_row[j] + 1
            substitution = 0 if a_token == b_token else 1
            substitution_cost = prev_row[j - 1] + substitution
            # Here copying the Kaldi compute-wer comparison order, which in
            # ties prefers:
            # insertion > deletion > substitution
            if (
                substitution_cost < insertion_cost
                and substitution_cost < deletion_cost
            ):
                curr_row[j] = substitution_cost
                # Again, note that if not substitution, the edit table already
                # has the correct no-op symbol.
                if substitution:
                    table[i][j] = EDIT_SYMBOLS["sub"]
            elif deletion_cost < insertion_cost:
                curr_row[j] = deletion_cost
                table[i][j] = EDIT_SYMBOLS["del"]
            else:
                curr_row[j] = insertion_cost
                table[i][j] = EDIT_SYMBOLS["ins"]
        # Move to the next row:
        prev_row[:] = curr_row[:]
    return table


def alignment(table):
    """Get the edit distance alignment from an edit op table.

    Walks back an edit operations table, produced by calling ``table(a, b)``,
    and collects the edit distance alignment of a to b. The alignment
    shows which token in a corresponds to which token in b. Note that the
    alignment is monotonic, one-to-zero-or-one.

    Arguments
    ----------
    table : list
        Edit operations table from ``op_table(a, b)``.

    Returns
    -------
    list
        Schema: ``[(str <edit-op>, int-or-None <i>, int-or-None <j>),]``
        List of edit operations, and the corresponding indices to a and b.
        See the EDIT_SYMBOLS dict for the edit-ops.
        The i indexes a, j indexes b, and the indices can be None, which means
        aligning to nothing.

    Example
    -------
    >>> # table for a=[1,2,3], b=[1,2,4]:
    >>> table = [['I', 'I', 'I', 'I'],
    ...          ['D', '=', 'I', 'I'],
    ...          ['D', 'D', '=', 'I'],
    ...          ['D', 'D', 'D', 'S']]
    >>> print(alignment(table))
    [('=', 0, 0), ('=', 1, 1), ('S', 2, 2)]
    """
    # The alignment will be the size of the longer sequence.
    # form: [(op, a_index, b_index)], index is None when aligned to empty
    alignment = []
    # Now we'll walk back the table to get the alignment.
    i = len(table) - 1
    j = len(table[0]) - 1
    while not (i == 0 and j == 0):
        if i == 0:
            j -= 1
            alignment.insert(0, (EDIT_SYMBOLS["ins"], None, j))
        elif j == 0:
            i -= 1
            alignment.insert(0, (EDIT_SYMBOLS["del"], i, None))
        else:
            if table[i][j] == EDIT_SYMBOLS["ins"]:
                j -= 1
                alignment.insert(0, (EDIT_SYMBOLS["ins"], None, j))
            elif table[i][j] == EDIT_SYMBOLS["del"]:
                i -= 1
                alignment.insert(0, (EDIT_SYMBOLS["del"], i, None))
            elif table[i][j] == EDIT_SYMBOLS["sub"]:
                i -= 1
                j -= 1
                alignment.insert(0, (EDIT_SYMBOLS["sub"], i, j))
            else:
                i -= 1
                j -= 1
                alignment.insert(0, (EDIT_SYMBOLS["eq"], i, j))
    return alignment


def count_ops(table):
    """Count the edit operations in the shortest edit path in edit op table.

    Walks back an edit operations table produced by table(a, b) and
    counts the number of insertions, deletions, and substitutions in the
    shortest edit path. This information is typically used in speech
    recognition to report the number of different error types separately.

    Arguments
    ----------
    table : list
        Edit operations table from ``op_table(a, b)``.

    Returns
    -------
    collections.Counter
        The counts of the edit operations, with keys:

        * "insertions"
        * "deletions"
        * "substitutions"

        NOTE: not all of the keys might appear explicitly in the output,
        but for the missing keys collections. The counter will return 0.

    Example
    -------
    >>> table = [['I', 'I', 'I', 'I'],
    ...          ['D', '=', 'I', 'I'],
    ...          ['D', 'D', '=', 'I'],
    ...          ['D', 'D', 'D', 'S']]
    >>> print(count_ops(table))
    Counter({'substitutions': 1})
    """
    edits = collections.Counter()
    # Walk back the table, gather the ops.
    i = len(table) - 1
    j = len(table[0]) - 1
    while not (i == 0 and j == 0):
        if i == 0:
            edits["insertions"] += 1
            j -= 1
        elif j == 0:
            edits["deletions"] += 1
            i -= 1
        else:
            if table[i][j] == EDIT_SYMBOLS["ins"]:
                edits["insertions"] += 1
                j -= 1
            elif table[i][j] == EDIT_SYMBOLS["del"]:
                edits["deletions"] += 1
                i -= 1
            else:
                if table[i][j] == EDIT_SYMBOLS["sub"]:
                    edits["substitutions"] += 1
                i -= 1
                j -= 1
    return edits


def _batch_to_dict_format(ids, seqs):
    # Used by wer_details_for_batch
    return dict(zip(ids, seqs))


def wer_details_for_batch(ids, refs, hyps, compute_alignments=False):
    """Convenient batch interface for ``wer_details_by_utterance``.

    ``wer_details_by_utterance`` can handle missing hypotheses, but
    sometimes (e.g. CTC training with greedy decoding) they are not needed,
    and this is a convenient interface in that case.

    Arguments
    ---------
    ids : list, torch.tensor
        Utterance ids for the batch.
    refs : list, torch.tensor
        Reference sequences.
    hyps : list, torch.tensor
        Hypothesis sequences.
    compute_alignments : bool, optional
        Whether to compute alignments or not. If computed, the details
        will also store the refs and hyps. (default: False)

    Returns
    -------
    list
        See ``wer_details_by_utterance``

    Example
    -------
    >>> ids = [['utt1'], ['utt2']]
    >>> refs = [[['a','b','c']], [['d','e']]]
    >>> hyps = [[['a','b','d']], [['d','e']]]
    >>> wer_details = []
    >>> for ids_batch, refs_batch, hyps_batch in zip(ids, refs, hyps):
    ...     details = wer_details_for_batch(ids_batch, refs_batch, hyps_batch)
    ...     wer_details.extend(details)
    >>> print(wer_details[0]['key'], ":",
    ...     "{:.2f}".format(wer_details[0]['WER']))
    utt1 : 33.33
    """
    refs = _batch_to_dict_format(ids, refs)
    hyps = _batch_to_dict_format(ids, hyps)
    return wer_details_by_utterance(
        refs, hyps, compute_alignments=compute_alignments, scoring_mode="strict"
    )


def wer_details_by_utterance(
    ref_dict, hyp_dict, compute_alignments=False, scoring_mode="strict"
):
    """Computes a wealth WER info about each single utterance.

    This info can then be used to compute summary details (WER, SER).

    Arguments
    ---------
    ref_dict : dict
        Should be indexable by utterance ids, and return the reference tokens
        for each utterance id as iterable
    hyp_dict : dict
        Should be indexable by utterance ids, and return
        the hypothesis tokens for each utterance id as iterable
    compute_alignments : bool
        Whether alignments should also be saved.
        This also saves the tokens themselves, as they are probably
        required for printing the alignments.
    scoring_mode : {'strict', 'all', 'present'}
        How to deal with missing hypotheses (reference utterance id
        not found in hyp_dict).

        * 'strict': Raise error for missing hypotheses.
        * 'all': Score missing hypotheses as empty.
        * 'present': Only score existing hypotheses.

    Returns
    -------
    list
        A list with one entry for every reference utterance. Each entry is a
        dict with keys:

        * "key": utterance id
        * "scored": (bool) Whether utterance was scored.
        * "hyp_absent": (bool) True if a hypothesis was NOT found.
        * "hyp_empty": (bool) True if hypothesis was considered empty
          (either because it was empty, or not found and mode 'all').
        * "num_edits": (int) Number of edits in total.
        * "num_ref_tokens": (int) Number of tokens in the reference.
        * "WER": (float) Word error rate of the utterance.
        * "insertions": (int) Number of insertions.
        * "deletions": (int) Number of deletions.
        * "substitutions": (int) Number of substitutions.
        * "alignment": If compute_alignments is True, alignment as list,
          see ``speechbrain.utils.edit_distance.alignment``.
          If compute_alignments is False, this is None.
        * "ref_tokens": (iterable) The reference tokens
          only saved if alignments were computed, else None.
        * "hyp_tokens": (iterable) the hypothesis tokens,
          only saved if alignments were computed, else None.

    Raises
    ------
    KeyError
        If scoring mode is 'strict' and a hypothesis is not found.
    """
    details_by_utterance = []
    for key, ref_tokens in ref_dict.items():
        # Initialize utterance_details
        utterance_details = {
            "key": key,
            "scored": False,
            "hyp_absent": None,
            "hyp_empty": None,
            "num_edits": None,
            "num_ref_tokens": len(ref_tokens),
            "WER": None,
            "insertions": None,
            "deletions": None,
            "substitutions": None,
            "alignment": None,
            "ref_tokens": ref_tokens if compute_alignments else None,
            "hyp_tokens": None,
        }
        if key in hyp_dict:
            utterance_details.update({"hyp_absent": False})
            hyp_tokens = hyp_dict[key]
        elif scoring_mode == "all":
            utterance_details.update({"hyp_absent": True})
            hyp_tokens = []
        elif scoring_mode == "present":
            utterance_details.update({"hyp_absent": True})
            details_by_utterance.append(utterance_details)
            continue  # Skip scoring this utterance
        elif scoring_mode == "strict":
            raise KeyError(
                "Key "
                + key
                + " in reference but missing in hypothesis and strict mode on."
            )
        else:
            raise ValueError("Invalid scoring mode: " + scoring_mode)
        # Compute edits for this utterance
        table = op_table(ref_tokens, hyp_tokens)
        ops = count_ops(table)
        # Update the utterance-level details if we got this far:
        utterance_details.update(
            {
                "scored": True,
                "hyp_empty": True
                if len(hyp_tokens) == 0
                else False,  # This also works for e.g. torch tensors
                "num_edits": sum(ops.values()),
                "num_ref_tokens": len(ref_tokens),
                "WER": 100.0 * sum(ops.values()) / len(ref_tokens),
                "insertions": ops["insertions"],
                "deletions": ops["deletions"],
                "substitutions": ops["substitutions"],
                "alignment": alignment(table) if compute_alignments else None,
                "ref_tokens": ref_tokens if compute_alignments else None,
                "hyp_tokens": hyp_tokens if compute_alignments else None,
            }
        )
        details_by_utterance.append(utterance_details)
    return details_by_utterance


def wer_summary(details_by_utterance):
    """
    Computes summary stats from the output of details_by_utterance

    Summary stats like WER

    Arguments
    ---------
    details_by_utterance : list
        See the output of wer_details_by_utterance

    Returns
    -------
    dict
        Dictionary with keys:

        * "WER": (float) Word Error Rate.
        * "SER": (float) Sentence Error Rate (percentage of utterances
          which had at least one error).
        * "num_edits": (int) Total number of edits.
        * "num_scored_tokens": (int) Total number of tokens in scored
          reference utterances (a missing hypothesis might still
          have been scored with 'all' scoring mode).
        * "num_erraneous_sents": (int) Total number of utterances
          which had at least one error.
        * "num_scored_sents": (int) Total number of utterances
          which were scored.
        * "num_absent_sents": (int) Hypotheses which were not found.
        * "num_ref_sents": (int) Number of all reference utterances.
        * "insertions": (int) Total number of insertions.
        * "deletions": (int) Total number of deletions.
        * "substitutions": (int) Total number of substitutions.

        NOTE: Some cases lead to ambiguity over number of
        insertions, deletions and substitutions. We
        aim to replicate Kaldi compute_wer numbers.
    """
    # Build the summary details:
    ins = dels = subs = 0
    num_scored_tokens = (
        num_scored_sents
    ) = num_edits = num_erraneous_sents = num_absent_sents = num_ref_sents = 0
    for dets in details_by_utterance:
        num_ref_sents += 1
        if dets["scored"]:
            num_scored_sents += 1
            num_scored_tokens += dets["num_ref_tokens"]
            ins += dets["insertions"]
            dels += dets["deletions"]
            subs += dets["substitutions"]
            num_edits += dets["num_edits"]
            if dets["num_edits"] > 0:
                num_erraneous_sents += 1
        if dets["hyp_absent"]:
            num_absent_sents += 1
    wer_details = {
        "WER": 100.0 * num_edits / num_scored_tokens,
        "SER": 100.0 * num_erraneous_sents / num_scored_sents,
        "num_edits": num_edits,
        "num_scored_tokens": num_scored_tokens,
        "num_erraneous_sents": num_erraneous_sents,
        "num_scored_sents": num_scored_sents,
        "num_absent_sents": num_absent_sents,
        "num_ref_sents": num_ref_sents,
        "insertions": ins,
        "deletions": dels,
        "substitutions": subs,
    }
    return wer_details


def wer_details_by_speaker(details_by_utterance, utt2spk):
    """Compute word error rate and another salient info grouping by speakers.

    Arguments
    ---------
    details_by_utterance : list
        See the output of wer_details_by_utterance
    utt2spk : dict
        Map from utterance id to speaker id


    Returns
    -------
    dict
        Maps speaker id to a dictionary of the statistics, with keys:

        * "speaker": Speaker id,
        * "num_edits": (int) Number of edits in total by this speaker.
        * "insertions": (int) Number insertions by this speaker.
        * "dels": (int) Number of deletions by this speaker.
        * "subs": (int) Number of substitutions by this speaker.
        * "num_scored_tokens": (int) Number of scored reference
          tokens by this speaker (a missing hypothesis might still
          have been scored with 'all' scoring mode).
        * "num_scored_sents": (int) number of scored utterances
          by this speaker.
        * "num_erraneous_sents": (int) number of utterance with at least
          one error, by this speaker.
        * "num_absent_sents": (int) number of utterances for which no
          hypotheses was found, by this speaker.
        * "num_ref_sents": (int) number of utterances by this speaker
          in total.
    """
    # Build the speakerwise details:
    details_by_speaker = {}
    for dets in details_by_utterance:
        speaker = utt2spk[dets["key"]]
        spk_dets = details_by_speaker.setdefault(
            speaker,
            collections.Counter(
                {
                    "speaker": speaker,
                    "insertions": 0,
                    "dels": 0,
                    "subs": 0,
                    "num_scored_tokens": 0,
                    "num_scored_sents": 0,
                    "num_edits": 0,
                    "num_erraneous_sents": 0,
                    "num_absent_sents": 0,
                    "num_ref_sents": 0,
                }
            ),
        )
        utt_stats = collections.Counter()
        if dets["hyp_absent"]:
            utt_stats.update({"num_absent_sents": 1})
        if dets["scored"]:
            utt_stats.update(
                {
                    "num_scored_sents": 1,
                    "num_scored_tokens": dets["num_ref_tokens"],
                    "insertions": dets["insertions"],
                    "dels": dets["deletions"],
                    "subs": dets["substitutions"],
                    "num_edits": dets["num_edits"],
                }
            )
            if dets["num_edits"] > 0:
                utt_stats.update({"num_erraneous_sents": 1})
        spk_dets.update(utt_stats)
    # We will in the end return a list of normal dicts
    # We want the output to be sortable
    details_by_speaker_dicts = []
    # Now compute speakerwise summary details
    for speaker, spk_dets in details_by_speaker.items():
        spk_dets["speaker"] = speaker
        if spk_dets["num_scored_sents"] > 0:
            spk_dets["WER"] = (
                100.0 * spk_dets["num_edits"] / spk_dets["num_scored_tokens"]
            )
            spk_dets["SER"] = (
                100.0
                * spk_dets["num_erraneous_sents"]
                / spk_dets["num_scored_sents"]
            )
        else:
            spk_dets["WER"] = None
            spk_dets["SER"] = None
        details_by_speaker_dicts.append(spk_dets)
    return details_by_speaker_dicts


def top_wer_utts(details_by_utterance, top_k=20):
    """
    Finds the k utterances with highest word error rates.

    Useful for diagnostic purposes, to see where the system
    is making the most mistakes.
    Returns results utterances which were not empty
    i.e. had to have been present in the hypotheses, with output produced

    Arguments
    ---------
    details_by_utterance : list
        See output of wer_details_by_utterance.
    top_k : int
        Number of utterances to return.

    Returns
    -------
    list
        List of at most K utterances,
        with the highest word error rates, which were not empty.
        The utterance dict has the same keys as
        details_by_utterance.
    """
    scored_utterances = [
        dets for dets in details_by_utterance if dets["scored"]
    ]
    utts_by_wer = sorted(
        scored_utterances, key=lambda d: d["WER"], reverse=True
    )
    top_non_empty = []
    top_empty = []
    while utts_by_wer and (
        len(top_non_empty) < top_k or len(top_empty) < top_k
    ):
        utt = utts_by_wer.pop(0)
        if utt["hyp_empty"] and len(top_empty) < top_k:
            top_empty.append(utt)
        elif not utt["hyp_empty"] and len(top_non_empty) < top_k:
            top_non_empty.append(utt)
    return top_non_empty, top_empty


def top_wer_spks(details_by_speaker, top_k=10):
    """
    Finds the K speakers with the highest word error rates.

    Useful for diagnostic purposes.

    Arguments
    ---------
    details_by_speaker : list
        See output of wer_details_by_speaker.
    top_k : int
        Number of seakers to return.

    Returns
    -------
    list
        List of at most K dicts (with the same keys as details_by_speaker)
        of speakers sorted by WER.
    """
    scored_speakers = [
        dets for dets in details_by_speaker if dets["num_scored_sents"] > 0
    ]
    spks_by_wer = sorted(scored_speakers, key=lambda d: d["WER"], reverse=True)
    if len(spks_by_wer) >= top_k:
        return spks_by_wer[:top_k]
    else:
        return spks_by_wer
