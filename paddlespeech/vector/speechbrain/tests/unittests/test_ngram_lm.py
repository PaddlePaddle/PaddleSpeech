def test_backofff_ngram_lm():
    from speechbrain.lm.ngram import BackoffNgramLM
    import math

    HALF = math.log(0.5)
    ngrams = {
        1: {tuple(): {"a": HALF, "b": HALF}},
        2: {("a",): {"a": HALF, "b": HALF}, ("b",): {"a": HALF}},
    }
    backoffs = {1: {("b",): 0.0}}
    lm = BackoffNgramLM(ngrams, backoffs)
    # The basic cases covered by the ngrams and backoffs:
    assert lm.logprob("a", ()) == HALF
    assert lm.logprob("b", ()) == HALF
    assert lm.logprob("a", ("a",)) == HALF
    assert lm.logprob("a", ("b",)) == HALF
    assert lm.logprob("b", ("a",)) == HALF
    assert lm.logprob("b", ("b",)) == HALF
    # Edge cases
    # Too large context:
    assert lm.logprob("a", ("a", "a")) == HALF
    assert lm.logprob("b", ("a", "b")) == HALF
    # OOV:
    assert lm.logprob("c", ()) == float("-inf")
    # OOV in context:
    assert lm.logprob("a", ("c",)) == HALF
