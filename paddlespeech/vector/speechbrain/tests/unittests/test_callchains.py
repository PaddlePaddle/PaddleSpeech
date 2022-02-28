def test_lengths_arg_exists():
    from speechbrain.utils.callchains import lengths_arg_exists

    def non_len_func(x):
        return x + 1

    def len_func(x, lengths):
        return x + lengths

    assert not lengths_arg_exists(non_len_func)
    assert lengths_arg_exists(len_func)


def test_lengths_capable_chain():
    from speechbrain.utils.callchains import LengthsCapableChain

    def non_len_func(x):
        return x + 1

    def len_func(x, lengths):
        return x + lengths

    def tuple_func(x):
        return x, x + 1

    chain = LengthsCapableChain(non_len_func, len_func)
    assert chain(1, 2) == 4
    assert chain(lengths=2, x=1) == 4
    chain.append(non_len_func)
    assert chain(1, 2) == 5
    chain.append(tuple_func)
    assert chain(1, 2) == 5
