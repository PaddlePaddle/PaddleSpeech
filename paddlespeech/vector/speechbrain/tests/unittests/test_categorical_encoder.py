import pytest
import paddle

def test_categorical_encoder(device):
    paddle.device.set_device(device)
    from speechbrain.dataio.encoder import CategoricalEncoder

    encoder = CategoricalEncoder()
    encoder.update_from_iterable("abcd")
    integers = encoder.encode_sequence("dcba")
    assert all(isinstance(i, int) for i in integers)
    assert encoder.is_continuous()
    with pytest.raises(KeyError):
        encoder.add_label("a")
    # Does NOT raise:
    encoder.ensure_label("a")
    with pytest.raises(KeyError):
        encoder.insert_label("a", -3)
    encoder.enforce_label("a", -3)
    assert encoder.encode_label("a") == -3
    assert not encoder.is_continuous()

    # Decoding:
    encoder = CategoricalEncoder()
    encoder.update_from_iterable("abcd")
    result = encoder.decode_paddle(
        paddle.to_tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype="float32")
    )
    assert result == [["a", "a"], ["b", "b"], ["c", "c"], ["d", "d"]]
    result = encoder.decode_ndim([[0, 0], [1, 1], [2, 2], [3, 3]])
    assert result == [["a", "a"], ["b", "b"], ["c", "c"], ["d", "d"]]
    result = encoder.decode_ndim(paddle.to_tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype="float32"))
    assert result == [["a", "a"], ["b", "b"], ["c", "c"], ["d", "d"]]
    result = encoder.decode_ndim([[[[[0, 0], [1, 1], [2, 2], [3, 3]]]]])
    assert result == [[[[["a", "a"], ["b", "b"], ["c", "c"], ["d", "d"]]]]]
    result = encoder.decode_paddle(
        paddle.to_tensor([[[[[0, 0], [1, 1], [2, 2], [3, 3]]]]],  dtype="float32")
    )
    assert result == [[[[["a", "a"], ["b", "b"], ["c", "c"], ["d", "d"]]]]]
    result = encoder.decode_ndim([[0, 0], [1], [2, 2, 2], []])
    assert result == [["a", "a"], ["b"], ["c", "c", "c"], []]

    encoder = CategoricalEncoder()
    encoder.limited_labelset_from_iterable("aabbbcccd", n_most_common=3)
    encoder.encode_sequence("abc")
    with pytest.raises(KeyError):
        encoder.encode_label("d")
    encoder = CategoricalEncoder()
    encoder.limited_labelset_from_iterable("aabbbcccd", min_count=3)
    encoder.encode_sequence("cbcb")
    with pytest.raises(KeyError):
        encoder.encode_label("a")
    with pytest.raises(KeyError):
        encoder.encode_label("d")
    encoder = CategoricalEncoder()
    encoder.limited_labelset_from_iterable(
        "aabbbcccd", n_most_common=3, min_count=3
    )
    encoder.encode_sequence("cbcb")
    with pytest.raises(KeyError):
        encoder.encode_label("a")
    with pytest.raises(KeyError):
        encoder.encode_label("d")

    encoder = CategoricalEncoder(unk_label="<unk>")
    encoder.update_from_iterable("abc")
    assert encoder.encode_label("a") == 1
    assert encoder.encode_label("d") == 0
    assert encoder.decode_ndim(encoder.encode_label("d")) == "<unk>"


def test_categorical_encoder_saving(tmpdir):
    paddle.device.set_device("cpu")
    from speechbrain.dataio.encoder import CategoricalEncoder

    encoder = CategoricalEncoder(starting_index=3)
    encoding_file = tmpdir / "char_encoding.txt"
    # First time this runs, the encoding is created:
    if not encoder.load_if_possible(encoding_file):
        encoder.update_from_iterable("abcd")
        encoder.save(encoding_file)
    else:
        assert False  # We should not get here!
    # Now, imagine a recovery:
    encoder = CategoricalEncoder()
    # The second time, the encoding is just loaded from file:
    if not encoder.load_if_possible(encoding_file):
        assert False  # We should not get here!
    integers = encoder.encode_sequence("dcba")
    assert all(isinstance(i, int) for i in integers)
    assert encoder.starting_index == 3  # This is also loaded

    # Also possible to encode tuples and load
    encoder = CategoricalEncoder()
    encoding_file = tmpdir / "tuple_encoding.txt"
    encoder.add_label((1, 2, 3))
    encoder.insert_label((1, 2), index=-1)
    encoder.save(encoding_file)
    # Reload
    encoder = CategoricalEncoder()
    assert encoder.load_if_possible(encoding_file)
    assert encoder.encode_label((1, 2)) == -1

    # Load unk:
    encoder = CategoricalEncoder(unk_label="UNKNOWN")
    encoding_file = tmpdir / "unk_encoding.txt"
    encoder.update_from_iterable("abc")
    encoder.save(encoding_file)
    encoder = CategoricalEncoder()
    assert encoder.load_if_possible(encoding_file)
    assert encoder.encode_label("a") == 1
    assert encoder.decode_ndim(encoder.encode_label("d")) == "UNKNOWN"
    # Even if set differently:
    encoder = CategoricalEncoder()
    encoder.add_unk()
    assert encoder.load_if_possible(encoding_file)
    assert encoder.encode_label("a") == 1
    assert encoder.decode_ndim(encoder.encode_label("d")) == "UNKNOWN"


def test_categorical_encoder_from_dataset():
    from speechbrain.dataio.encoder import CategoricalEncoder
    from speechbrain.dataio.dataset import DynamicItemDataset

    encoder = CategoricalEncoder()
    data = {
        "utt1": {"foo": -1, "bar": 0, "text": "hello world"},
        "utt2": {"foo": 1, "bar": 2, "text": "how are you world"},
        "utt3": {"foo": 3, "bar": 4, "text": "where are you world"},
        "utt4": {"foo": 5, "bar": 6, "text": "hello nation"},
    }
    dynamic_items = [
        {"func": lambda x: x.split(), "takes": ["text"], "provides": "words"},
        {
            "func": encoder.encode_sequence,
            "takes": ["words"],
            "provides": "words_t",
        },
    ]
    output_keys = ["words_t"]
    dataset = DynamicItemDataset(data, dynamic_items, output_keys)
    encoder.update_from_didataset(dataset, "words", sequence_input=True)
    assert dataset[0]["words_t"] == [0, 1]
    assert encoder.decode_ndim(dataset[0]["words_t"]) == ["hello", "world"]


def test_text_encoder(tmpdir):
    from speechbrain.dataio.encoder import TextEncoder

    encoder = TextEncoder()
    encoding_file = tmpdir / "text_encoding.txt"
    encoder.add_bos_eos()
    encoder.update_from_iterable(
        [["hello", "world"], ["how", "are", "you", "world"]],
        sequence_input=True,
    )
    encoded = encoder.encode_sequence(
        encoder.prepend_bos_label(["are", "you", "world"])
    )
    assert encoded[0] == 0
    encoded = encoder.append_eos_index(
        encoder.encode_sequence(["are", "you", "world"])
    )
    assert encoded[-1] == 1  # By default uses just one sentence_boundary marker
    encoder.save(encoding_file)
    encoder = TextEncoder()
    assert encoder.load_if_possible(encoding_file)
    encoded = encoder.encode_sequence(
        encoder.append_eos_label(["are", "you", "world"])
    )
    assert encoded[-1] == 1
    encoded = encoder.prepend_bos_index(
        encoder.encode_sequence(["are", "you", "world"])
    )
    assert encoded[0] == 0


def test_ctc_encoder(tmpdir):
    from speechbrain.dataio.encoder import CTCTextEncoder

    encoder = CTCTextEncoder()
    encoder.insert_bos_eos(
        bos_label="<s>", bos_index=0, eos_label="</s>", eos_index=1
    )
    encoder.insert_blank(blank_label="_", index=2)
    encoding_file = tmpdir / "ctc_encoding.txt"
    encoder.update_from_iterable(["abcd", "bcdef"], sequence_input=True)
    encoded = encoder.encode_sequence(encoder.prepend_bos_label(["a", "b"]))
    assert encoded[0] == 0
    encoder.save(encoding_file)
    encoder = CTCTextEncoder()
    assert encoder.load_if_possible(encoding_file)
    assert (
        "".join(encoder.collapse_labels("_bb_aaa___bbbbb_b_eeee_____"))
        == "babbe"
    )
    assert "".join(encoder.collapse_labels("babe")) == "babe"
    assert (
        "".join(
            encoder.collapse_labels(
                "_bb_aaa___bbbbb_b_eeee_____", merge_repeats=False
            )
        )
        == "bbaaabbbbbbeeee"
    )
    assert encoder.decode_ndim(
        (
            encoder.collapse_indices_ndim(
                [
                    [0, 2, 4, 4, 2, 3, 3, 3, 2, 2, 2, 4, 2, 4, 2, 7, 2, 1],
                    [[0, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]],
                ]
            )
        )
    ) == [
        ["<s>", "b", "a", "b", "b", "e", "</s>"],
        [["<s>", "a", "b", "c", "d", "c", "b", "a", "</s>"]],
    ]
