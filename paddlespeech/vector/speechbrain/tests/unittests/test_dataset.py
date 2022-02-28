def test_dynamic_item_dataset():
    from speechbrain.dataio.dataset import DynamicItemDataset
    import operator

    data = {
        "utt1": {"foo": -1, "bar": 0, "text": "hello world"},
        "utt2": {"foo": 1, "bar": 2, "text": "how are you world"},
        "utt3": {"foo": 3, "bar": 4, "text": "where are you world"},
        "utt4": {"foo": 5, "bar": 6, "text": "hello nation"},
    }
    dynamic_items = [
        {"provides": "foobar", "func": operator.add, "takes": ["foo", "bar"]}
    ]
    output_keys = ["text"]
    dataset = DynamicItemDataset(data, dynamic_items, output_keys)
    assert dataset[0] == {"text": "hello world"}
    dataset.set_output_keys(["id", "foobar"])
    assert dataset[1] == {"id": "utt2", "foobar": 3}
    dataset.add_dynamic_item(operator.sub, ["bar", "foo"], "barfoo")
    dataset.set_output_keys(["id", "barfoo"])
    assert dataset[1] == {"id": "utt2", "barfoo": 1}
    # Iterate:
    barfoosum = 0
    for data_point in iter(dataset):
        barfoosum += data_point["barfoo"]
    assert barfoosum == 4


def test_filtered_sorted_dynamic_item_dataset():
    from speechbrain.dataio.dataset import DynamicItemDataset
    import operator

    data = {
        "utt1": {"foo": -1, "bar": 0, "text": "hello world"},
        "utt2": {"foo": 1, "bar": 2, "text": "how are you world"},
        "utt3": {"foo": 3, "bar": 4, "text": "where are you world"},
        "utt4": {"foo": 5, "bar": 6, "text": "hello nation"},
    }
    dynamic_items = [
        {"provides": "foobar", "func": operator.add, "takes": ["foo", "bar"]}
    ]
    output_keys = ["text"]
    dataset = DynamicItemDataset(data, dynamic_items, output_keys)
    subset = dataset.filtered_sorted(key_min_value={"foo": 3})
    # Note: subset is not a shallow view!
    dataset.set_output_keys(["id", "foo"])
    assert subset[0] == {"text": "where are you world"}
    subset.set_output_keys(["id", "foo"])
    assert subset[0] == {"id": "utt3", "foo": 3}

    # Note: now making a subset from a version which had id and foo as output keys
    subset = dataset.filtered_sorted(key_max_value={"bar": 2})
    assert len(subset) == 2
    assert subset[0] == {"id": "utt1", "foo": -1}

    dataset.add_dynamic_item(operator.sub, ["bar", "foo"], "barfoo")
    subset = dataset.filtered_sorted(key_test={"barfoo": lambda x: x == 1})
    assert len(subset) == 4
    assert subset[3] == {"id": "utt4", "foo": 5}
    subset = dataset.filtered_sorted(key_min_value={"foo": 3, "bar": 2})
    assert subset[0]["id"] == "utt3"
    subset = dataset.filtered_sorted(
        key_min_value={"foo": 3}, key_max_value={"foobar": 7}
    )
    assert len(subset) == 1
    subset = dataset.filtered_sorted(
        key_min_value={"foo": 3}, key_max_value={"foobar": 3}
    )
    assert len(subset) == 0
    subset = dataset.filtered_sorted(select_n=1, key_min_value={"foo": 3})
    assert len(subset) == 1
    assert subset[0]["id"] == "utt3"

    # Can filter twice!
    subset = dataset.filtered_sorted(key_min_value={"foo": 3})
    subsetsubset = subset.filtered_sorted(key_max_value={"bar": 4})
    assert len(subset) == 2
    assert len(subsetsubset) == 1

    # Can sort:
    subset = dataset.filtered_sorted(sort_key="foo", reverse=True)
    assert subset[0]["id"] == "utt4"

    # Can filter and sort at the same time:
    subset = dataset.filtered_sorted(
        key_max_value={"foo": 1}, sort_key="foo", reverse=True
    )
    assert subset[0]["id"] == "utt2"
