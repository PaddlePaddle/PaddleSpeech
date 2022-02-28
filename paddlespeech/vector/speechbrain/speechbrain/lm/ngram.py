"""
N-gram language model query interface

Authors
 * Aku Rouhe 2020
"""
import collections

NEGINFINITY = float("-inf")


class BackoffNgramLM:
    """
    Query interface for backoff N-gram language models

    The ngrams format is best explained by an example query: P( world | <s>,
    hello ), i.e. trigram model, probability of "world" given "<s> hello", is:
    `ngrams[2][("<s>", "hello")]["world"]`

    On the top level, ngrams is a dict of different history lengths, and each
    order is a dict, with contexts (tuples) as keys and (log-)distributions
    (dicts) as values.

    The backoffs format is a little simpler. On the top level, backoffs is a
    list of different context-orders, and each order is a mapping (dict) from
    backoff context to backoff (log-)weight

    Arguments
    ---------
    ngrams : dict
        The N-gram log probabilities.
        This is a triply nested dict.
        The first layer is indexed by N-gram order (integer).
        The second layer is indexed by the context (tuple of tokens).
        The third layer is indexed by tokens, and maps to the log prob.
        Example:
        log(P(fox|a quick red)) = -5.3 is accessed by:
        `ngrams[4][('a', 'quick', 'red')]['fox']`
    backoffs : dict
        The backoff log weights.
        This is a doubly nested dict.
        The first layer is indexed by N-gram order (integer).
        The second layer is indexed by the backoff history (tuple of tokens)
        i.e. the context on which the probability distribution is conditioned
        on. This maps to the log weights.
        Example:
        If log(P(fox|a quick red)) is not listed, we find
        log(backoff(a quick red)) = -23.4, which is accessed:
        `backoffs[3][('a', 'quick', 'red')]`
        This dict needs to have entries for orders up to at least N-1 (even if
        they are empty). It may also have entries for order N, though those
        can never be accessed.

    Example
    -------
    >>> import math
    >>> ngrams = {1: {tuple(): {'a': -0.6931, 'b': -0.6931}},
    ...           2: {('a',): {'a': -0.6931, 'b': -0.6931},
    ...               ('b',): {'a': -0.6931}}}
    >>> backoffs = {1: {('b',): 0.}}
    >>> lm = BackoffNgramLM(ngrams, backoffs)
    >>> round(math.exp(lm.logprob('a', ('b',))), 1)
    0.5
    >>> round(math.exp(lm.logprob('b', ('b',))), 1)
    0.5

    """

    def __init__(self, ngrams, backoffs):
        # Backoffs of length equal to max N-gram order can never be used,
        # but interface-wise we support having that order specified as well.
        # This plays nice e.g. with ARPA model loading.
        order = len(ngrams)
        if not (len(backoffs) == order or len(backoffs) == order - 1):
            raise ValueError("Backoffs dict needs to be of order N or N-1")
        self.ngrams = ngrams
        self.backoffs = backoffs
        self.top_order = order

    def logprob(self, token, context=tuple()):
        # If a longer context is given than we can ever use,
        # just use less context.
        query_order = len(context) + 1
        if query_order > self.top_order:
            return self.logprob(token, context[1:])
        # Now, let's see if we have both:
        # a distribution for the query context at all
        # and if so, a probability for the token.
        # Then we'll just return that.
        if (
            context in self.ngrams[query_order]
            and token in self.ngrams[query_order][context]
        ):
            return self.ngrams[query_order][context][token]
        # If we're here, no direct probability stored for the query.
        # Missing unigram queries are a special case, the recursion will stop.
        if query_order == 1:
            return NEGINFINITY  # Zeroth order for not found
        # Otherwise, we'll backoff to lower order model.
        # First, we'll get add the backoff log weight
        context_order = query_order - 1
        backoff_log_weight = self.backoffs[context_order].get(context, 0.0)
        # And then just recurse:
        lp = self.logprob(token, context[1:])
        return lp + backoff_log_weight


def ngram_evaluation_details(data, LM):
    """
    Evaluates the N-gram LM on each sentence in data

    Call `ngram_perplexity` with the output of this function to compute the
    perplexity.

    Arguments
    ---------
    data : iterator
        An iterator over sentences, where each sentence should be an iterator
        as returned by `speechbrain.lm.counting.ngrams_for_evaluation`
    LM : BackoffNgramLM
        The language model to evaluate

    Returns
    -------
    list
        List of `collections.Counter`s which have the keys "num_tokens" and
        "neglogprob", giving the number of tokens and logprob of each sentence
        (in the same order as data).

    NOTE
    ----
    The `collections.Counter` cannot add negative numbers. Thus it is important
    to use negative log probabilities (always >=0).

    Example
    -------
    >>> class MockLM:
    ...     def __init__(self):
    ...         self.top_order = 3
    ...     def logprob(self, token, context):
    ...         return -1.0
    >>> LM = MockLM()
    >>> data = [[("S", ("<s>",)),
    ...          ("p", ("<s>", "S")),
    ...          ("e", ("S", "p")),
    ...          ("e", ("p", "e")),
    ...          ("c", ("e", "e")),
    ...          ("h", ("e", "c")),
    ...          ("</s>", ("c", "h"))],
    ...         [("B", ("<s>",)),
    ...          ("r", ("<s>", "B")),
    ...          ("a", ("B", "r")),
    ...          ("i", ("r", "a")),
    ...          ("n", ("a", "i")),
    ...          ("</s>", ("i", "n"))]]
    >>> sum(ngram_evaluation_details(data, LM), collections.Counter())
    Counter({'num_tokens': 13, 'neglogprob': 13.0})

    """
    details = []
    for sentence in data:
        counter = collections.Counter()
        for token, context in sentence:
            counter["num_tokens"] += 1
            counter["neglogprob"] += -LM.logprob(token, context)
        details.append(counter)
    return details


def ngram_perplexity(eval_details, logbase=10.0):
    """
    Computes perplexity from a list of individual sentence evaluations.

    Arguments
    ---------
    eval_details : list
        List of individual sentence evaluations. As returned by
        `ngram_evaluation_details`
    logbase : float
        The logarithm base to use.

    Returns
    -------
    float
        The computed perplexity.

    Example
    -------
    >>> eval_details = [
    ...     collections.Counter(neglogprob=5, num_tokens=5),
    ...     collections.Counter(neglogprob=15, num_tokens=15)]
    >>> ngram_perplexity(eval_details)
    10.0

    """
    counter = sum(eval_details, collections.Counter())
    exponent = counter["neglogprob"] / counter["num_tokens"]
    perplexity = logbase ** exponent
    return perplexity
