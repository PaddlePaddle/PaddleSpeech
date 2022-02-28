import os
import paddle


def test_tokenizer():
    from speechbrain.tokenizers.SentencePiece import SentencePiece

    gt = [
        ["HELLO", "MORNING", "MORNING", "HELLO"],
        ["HELLO", "MORNING", "HELLO"],
    ]

    # Word-level input test
    dict_int2lab = {1: "HELLO", 2: "MORNING"}

    spm = SentencePiece(
        os.path.abspath("tokenizer_data/"),
        100,
        annotation_train=os.path.abspath(
            "tests/unittests/tokenizer_data/dev-clean.csv"
        ),
        annotation_read="wrd",
        model_type="bpe",
    )
    encoded_seq_ids, encoded_seq_pieces = spm(
        paddle.Tensor([[1, 2, 2, 1], [1, 2, 1, 0]]),
        paddle.Tensor([1.0, 0.75]),
        dict_int2lab,
        task="encode",
    )
    lens = (encoded_seq_pieces * encoded_seq_ids.shape[1]).round().int()
    # decode from torch tensors (batch, batch_lens)
    words_seq = spm(encoded_seq_ids, encoded_seq_pieces, task="decode")
    assert words_seq == gt, "output not the same"
    # decode from a list of bpe sequence (without padding)
    hyps_list = [
        encoded_seq_ids[0].int().tolist(),
        encoded_seq_ids[1][: lens[1]].int().tolist(),
    ]
    words_seq = spm(hyps_list, task="decode_from_list")
    assert words_seq == gt, "output not the same"

    # Char-level input test
    dict_int2lab = {
        1: "H",
        2: "E",
        3: "L",
        4: "O",
        5: "M",
        6: "R",
        7: "N",
        8: "I",
        9: "G",
        10: "_",
    }

    spm = SentencePiece(
        os.path.abspath("tokenizer_data/"),
        100,
        annotation_train=os.path.abspath(
            "tests/unittests/tokenizer_data/dev-clean.csv"
        ),
        annotation_read="char",
        char_format_input=True,
        model_type="bpe",
    )
    encoded_seq_ids, encoded_seq_pieces = spm(
        paddle.Tensor(
            [
                [
                    1,
                    2,
                    3,
                    3,
                    4,
                    10,
                    5,
                    4,
                    6,
                    7,
                    8,
                    7,
                    9,
                    10,
                    5,
                    4,
                    6,
                    7,
                    8,
                    7,
                    9,
                    10,
                    1,
                    2,
                    3,
                    3,
                    4,
                ],
                [
                    1,
                    2,
                    3,
                    3,
                    4,
                    10,
                    5,
                    4,
                    6,
                    7,
                    8,
                    7,
                    9,
                    10,
                    1,
                    2,
                    3,
                    3,
                    4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        ),
        paddle.Tensor([1.0, 0.7037037037037037]),
        dict_int2lab,
        task="encode",
    )
    lens = (encoded_seq_pieces * encoded_seq_ids.shape[1]).round().int()
    # decode from torch tensors (batch, batch_lens)
    words_seq = spm(encoded_seq_ids, encoded_seq_pieces, task="decode")
    assert words_seq == gt, "output not the same"
    # decode from a list of bpe sequence (without padding)
    hyps_list = [
        encoded_seq_ids[0].int().tolist(),
        encoded_seq_ids[1][: lens[1]].int().tolist(),
    ]
    words_seq = spm(hyps_list, task="decode_from_list")
    assert words_seq == gt, "output not the same"
