def test_accumulatable_wer_stats():
    from speechbrain.utils.edit_distance import accumulatable_wer_stats

    refs = [[[1, 2, 3], [4, 5, 6]], [[7, 8], [9]]]
    hyps = [[[1, 2, 4], [5, 6]], [[7, 8], [10]]]
    # Test basic functionality:
    stats = accumulatable_wer_stats(refs[0], hyps[0])
    assert stats["WER"] == 100.0 * 2 / 6
    stats = accumulatable_wer_stats(refs[1], hyps[1], stats)
    assert stats["WER"] == 100.0 * 3 / 9
    # Test edge cases:
    import math

    # No batches:
    stats = accumulatable_wer_stats([], [])
    assert stats["num_ref_tokens"] == 0
    assert math.isnan(stats["WER"])
    # Empty hyp sequence:
    stats = accumulatable_wer_stats([[1, 2, 3]], [[]])
    assert stats["num_ref_tokens"] == 3
    assert stats["WER"] == 100.0
    # Empty ref sequence:
    stats = accumulatable_wer_stats([[]], [[1, 2, 3]])
    assert stats["num_ref_tokens"] == 0
    assert stats["insertions"] == 3
    assert math.isnan(stats["WER"])


def test_op_table():
    from speechbrain.utils.edit_distance import op_table, EDIT_SYMBOLS

    assert len(op_table([1, 2, 3], [1, 2, 4])) == 4
    assert len(op_table([1, 2, 3], [1, 2, 4])[0]) == 4
    assert len(op_table([1, 2, 3], [])) == 4
    assert len(op_table([1, 2, 3], [])[0]) == 1
    assert op_table([1, 2, 3], [1, 2, 4])[3][3] == EDIT_SYMBOLS["sub"]
    assert op_table([1, 2, 3], [1, 2, 4])[2][2] == EDIT_SYMBOLS["eq"]
    assert op_table([1, 2, 3], [1, 2, 4])[0][0] == EDIT_SYMBOLS["eq"]


def test_alignment():
    from speechbrain.utils.edit_distance import alignment, EDIT_SYMBOLS

    I = EDIT_SYMBOLS["ins"]  # noqa: E741, here I is a good var name
    D = EDIT_SYMBOLS["del"]
    S = EDIT_SYMBOLS["sub"]
    E = EDIT_SYMBOLS["eq"]
    table = [[I, I, I, I], [D, E, I, I], [D, D, E, I], [D, D, D, S]]
    assert alignment(table) == [(E, 0, 0), (E, 1, 1), (S, 2, 2)]


def test_count_ops():
    from speechbrain.utils.edit_distance import count_ops, EDIT_SYMBOLS

    I = EDIT_SYMBOLS["ins"]  # noqa: E741, here I is a good var name
    D = EDIT_SYMBOLS["del"]
    S = EDIT_SYMBOLS["sub"]
    E = EDIT_SYMBOLS["eq"]
    table = [[I, I, I, I], [D, E, I, I], [D, D, E, I], [D, D, D, S]]
    assert count_ops(table)["insertions"] == 0
    assert count_ops(table)["deletions"] == 0
    assert count_ops(table)["substitutions"] == 1
