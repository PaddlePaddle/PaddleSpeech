from speechbrain.utils.metric_stats import MetricStats


def merge_words(sequences):
    """Merge successive words into phrase, putting space between each word

    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains a word sequence.
    Returns
    -------
    The list contains phrase sequences.
    """
    results = []
    for seq in sequences:
        words = " ".join(seq)
        results.append(words)
    return results


class BLEUStats(MetricStats):
    """A class for tracking BLEU (https://www.aclweb.org/anthology/P02-1040.pdf).
    Arguments
    ---------
    merge_words: bool
        Whether to merge the successive words to create sentences.
    Example
    -------
    >>> bleu = BLEUStats()
    >>> i2l = {0: 'a', 1: 'b'}
    >>> bleu.append(
    ...     ids=['utterance1'],
    ...     predict=[[0, 1, 1]],
    ...     targets=[[[0, 1, 0]], [[0, 1, 1]], [[1, 1, 0]]],
    ...     ind2lab=lambda batch: [[i2l[int(x)] for x in seq] for seq in batch],
    ... )
    >>> stats = bleu.summarize()
    >>> stats['BLEU']
    0.0
    """

    def __init__(
        self, lang="en", merge_words=True,
    ):

        self.clear()
        self.merge_words = merge_words

        self.predicts = []
        self.targets = None

    def append(
        self, ids, predict, targets, ind2lab=None,
    ):
        """Add stats to the relevant containers.
        * See MetricStats.append()
        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : torch.tensor
            A predicted output, for comparison with the target output
        targets : list
            list of references (when measuring BLEU, one sentence could have more
                                than one target translation).
        ind2lab : callable
            Callable that maps from indices to labels, operating on batches,
            for writing alignments.
        """
        self.ids.extend(ids)

        if ind2lab is not None:
            predict = ind2lab(predict)
            targets = [ind2lab(t) for t in targets]

        if self.merge_words:
            predict = merge_words(predict)
            targets = [merge_words(t) for t in targets]

        self.predicts.extend(predict)
        if self.targets is None:
            self.targets = targets
        else:
            assert len(self.targets) == len(targets)
            for i in range(len(self.targets)):
                self.targets[i].extend(targets[i])

    def summarize(self, field=None):
        """Summarize the BLEU and return relevant statistics.
        * See MetricStats.summarize()
        """

        # Check extra-dependency for computing the bleu score
        try:
            import sacrebleu
        except ImportError:
            print(
                "Please install sacrebleu (https://github.com/mjpost/sacreble) in order to use the BLEU metric"
            )

        scores = sacrebleu.corpus_bleu(self.predicts, self.targets)
        details = {}
        details["BLEU"] = scores.score
        details["BP"] = scores.bp
        details["ratio"] = scores.sys_len / scores.ref_len
        details["hyp_len"] = scores.sys_len
        details["ref_len"] = scores.ref_len
        details["precisions"] = scores.precisions

        self.scores = scores
        self.summary = details

        # Add additional, more generic key
        self.summary["bleu_score"] = self.summary["BLEU"]

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        if not self.summary:
            self.summarize()

        print(self.scores, file=filestream)
