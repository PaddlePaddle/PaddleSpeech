import pytest


def test_read_arpa():
    from speechbrain.lm.arpa import read_arpa
    import io

    with io.StringIO() as f:
        print("Anything can be here", file=f)
        print("", file=f)
        print("\\data\\", file=f)
        print("ngram 1=2", file=f)
        print("ngram 2=3", file=f)
        print("", file=f)  # Ends data section
        print("\\1-grams:", file=f)
        print("-0.6931 a", file=f)
        print("-0.6931 b 0.", file=f)
        print("", file=f)  # Ends unigram section
        print("\\2-grams:", file=f)
        print("-0.6931 a a", file=f)
        print("-0.6931 a b", file=f)
        print("-0.6931 b a", file=f)
        print("", file=f)  # Ends bigram section
        print("\\end\\", file=f)  # Ends whole file
        f.seek(0)
        num_grams, ngrams, backoffs = read_arpa(f)
    assert num_grams[1] == 2
    assert num_grams[2] == 3
    assert ngrams[1][()]["a"] == -0.6931
    assert ngrams[1][()]["b"] == -0.6931
    assert ngrams[2][("a",)]["a"] == -0.6931
    assert ngrams[2][("a",)]["b"] == -0.6931
    assert ngrams[2][("b",)]["a"] == -0.6931
    assert backoffs[1][("b",)] == 0.0
    assert list(backoffs[1].keys()) == [("b",)]

    # Badly formatted ARPA file:
    with io.StringIO() as f:
        print("Anything can be here", file=f)
        print("", file=f)
        print("\\data\\", file=f)
        print("ngram 1=1", file=f)
        print("", file=f)  # Ends data section
        print("\\1-grams:", file=f)
        print("-0.6931 a", file=f)
        print("", file=f)  # Ends bigram section
        # BUT! \end\ is missing
        f.seek(0)

        with pytest.raises(ValueError):
            num_grams, ngrams, backoffs = read_arpa(f)

    # You can theoretically have many ARPA LMs in one file:
    with io.StringIO() as f:
        for _ in range(3):
            print("Anything can be here", file=f)
            print("", file=f)
            print("\\data\\", file=f)
            print("ngram 1=2", file=f)
            print("ngram 2=3", file=f)
            print("", file=f)  # Ends data section
            print("\\1-grams:", file=f)
            print("-0.6931 a", file=f)
            print("-0.6931 b 0.", file=f)
            print("", file=f)  # Ends unigram section
            print("\\2-grams:", file=f)
            print("-0.6931 a a", file=f)
            print("-0.6931 a b", file=f)
            print("-0.6931 b a", file=f)
            print("", file=f)  # Ends bigram section
            print("\\end\\", file=f)  # Ends whole file
        f.seek(0)
        # Now read three times:
        num_grams, ngrams, backoffs = read_arpa(f)
        num_grams, ngrams, backoffs = read_arpa(f)
        num_grams, ngrams, backoffs = read_arpa(f)

        # And it still worked:
        assert num_grams[1] == 2
        assert num_grams[2] == 3
        assert ngrams[1][()]["a"] == -0.6931
        assert ngrams[1][()]["b"] == -0.6931
        assert ngrams[2][("a",)]["a"] == -0.6931
        assert ngrams[2][("a",)]["b"] == -0.6931
        assert ngrams[2][("b",)]["a"] == -0.6931
        assert backoffs[1][("b",)] == 0.0
        assert list(backoffs[1].keys()) == [("b",)]

        # Try to read a fourth time, though, and it will fail
        with pytest.raises(ValueError):
            num_grams, ngrams, backoffs = read_arpa(f)


def test_weird_arpa_formats():
    # We've decided to not be picky about ARPA format
    from speechbrain.lm.arpa import read_arpa
    import io

    with io.StringIO() as f:
        print("Anything can be here", file=f)
        print("", file=f)
        print("\\data\\", file=f)
        print("ngram 1=2", file=f)
        print("ngram 2=3", file=f)
        # No empty line before next section
        # and starts off with 2 grams:
        print("\\2-grams:", file=f)
        print("-0.6931 a a", file=f)
        print("-0.6931 a b", file=f)
        print("-0.6931 b a", file=f)
        # No empty line before next section
        print("\\1-grams:", file=f)
        print("-0.6931 a", file=f)
        print("-0.6931 b 0.", file=f)
        # No empty line before next section
        print("\\end\\", file=f)  # Ends whole file, this is still required
        f.seek(0)
        num_grams, ngrams, backoffs = read_arpa(f)
    assert num_grams[1] == 2
    assert num_grams[2] == 3
    assert ngrams[1][()]["a"] == -0.6931
    assert ngrams[1][()]["b"] == -0.6931
    assert ngrams[2][("a",)]["a"] == -0.6931
    assert ngrams[2][("a",)]["b"] == -0.6931
    assert ngrams[2][("b",)]["a"] == -0.6931
    assert backoffs[1][("b",)] == 0.0
    assert list(backoffs[1].keys()) == [("b",)]
