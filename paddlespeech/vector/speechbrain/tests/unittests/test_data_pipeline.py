import pytest


def test_data_pipeline():
    from speechbrain.utils.data_pipeline import DataPipeline

    pipeline = DataPipeline(
        ["text"],
        dynamic_items=[
            {"func": lambda x: x.lower(), "takes": ["text"], "provides": "foo"},
            {"func": lambda x: x[::-1], "takes": "foo", "provides": ["bar"]},
        ],
        output_keys=["text", "foo", "bar"],
    )
    result = pipeline({"text": "Test"})
    print(result)
    assert result["bar"] == "tset"
    pipeline = DataPipeline(["foo", "bar"])
    pipeline.add_dynamic_item(
        func=lambda x, y: x + y, takes=["foo", "bar"], provides="foobar"
    )
    pipeline.set_output_keys(["bar", "foobar"])
    result = pipeline({"foo": 1, "bar": 2})
    assert result["foobar"] == 3
    pipeline = DataPipeline(["foo", "bar"])
    from unittest.mock import MagicMock, Mock

    watcher = Mock()
    pipeline.add_dynamic_item(
        provides="foobar", func=watcher, takes=["foo", "bar"]
    )
    result = pipeline({"foo": 1, "bar": 2})
    assert not watcher.called
    pipeline = DataPipeline(["foo", "bar"])
    watcher = MagicMock(return_value=3)
    pipeline.add_dynamic_item(watcher, ["foo", "bar"], "foobar")
    pipeline.add_dynamic_item(lambda x: x, ["foobar"], ["truebar"])
    pipeline.set_output_keys(("truebar",))
    result = pipeline({"foo": 1, "bar": 2})
    assert watcher.called
    assert result["truebar"] == 3
    pipeline = DataPipeline(["foo", "bar"])
    watcher = MagicMock(return_value=3)
    pipeline.add_dynamic_item(
        func=watcher, takes=["foo", "bar"], provides="foobar"
    )
    pipeline.add_dynamic_item(
        func=lambda x: x, takes=["foo"], provides="truebar"
    )
    pipeline.set_output_keys(("truebar",))
    result = pipeline({"foo": 1, "bar": 2})
    assert not watcher.called
    assert result["truebar"] == 1

    pipeline = DataPipeline(["foo", "bar"])
    watcher = MagicMock(return_value=3)
    pipeline.add_dynamic_item(
        func=watcher, takes=["foo", "bar"], provides="foobar"
    )
    pipeline.set_output_keys(("foobar", "foo"))
    result = pipeline({"foo": 1, "bar": 2})
    assert watcher.called
    assert "foo" in result
    assert "foobar" in result
    assert "bar" not in result
    # Can change the outputs (continues previous tests)
    watcher.reset_mock()
    pipeline.set_output_keys(["bar"])
    result = pipeline({"foo": 1, "bar": 2})
    assert not watcher.called
    assert "foo" not in result
    assert "foobar" not in result
    assert "bar" in result
    # Finally, can also still request any specific key:
    computed = pipeline.compute_specific(["foobar"], {"foo": 1, "bar": 2})
    assert watcher.called
    assert computed["foobar"] == 3

    # Output can be a mapping:
    # (Key appears outside, value is internal)
    pipeline.set_output_keys({"signal": "foobar"})
    result = pipeline({"foo": 1, "bar": 2})
    assert result["signal"] == 3


def test_takes_provides():
    from speechbrain.utils.data_pipeline import takes, provides

    @takes("a")
    @provides("b")
    def a_to_b(a):
        return a + 1

    assert a_to_b(1) == 2
    a_to_b.reset()
    # Normal dynamic item can be called twice:
    assert a_to_b(1) == 2
    assert a_to_b(1) == 2
    # And it knows what it needs:
    assert a_to_b.next_takes() == ("a",)
    # And it knows what it gives:
    assert a_to_b.next_provides() == ("b",)


def test_MIMO_pipeline():
    from speechbrain.utils.data_pipeline import DataPipeline, takes, provides

    @takes("text", "other-text")
    @provides("reversed", "concat")
    def text_pipeline(text, other):
        return text[::-1], text + other

    @takes("reversed", "concat")
    @provides("reversed_twice", "double_concat")
    def second_pipeline(rev, concat):
        yield rev[::-1]
        yield concat + concat

    @provides("hello-world")
    def provider():
        yield "hello-world"

    @takes("hello-world", "reversed_twice")
    @provides("message")
    def messenger(hello, name):
        return f"{hello}, {name}"

    pipeline = DataPipeline(
        ["text", "other-text"],
        dynamic_items=[second_pipeline, text_pipeline],
        output_keys=["text", "reversed", "reversed_twice"],
    )
    result = pipeline({"text": "abc", "other-text": "def"})
    assert result["reversed"] == "cba"
    assert result["reversed_twice"] == "abc"
    result = pipeline.compute_specific(
        ["concat"], {"text": "abc", "other-text": "def"}
    )
    assert result["concat"] == "abcdef"
    result = pipeline.compute_specific(
        ["double_concat"], {"text": "abc", "other-text": "def"}
    )
    assert result["double_concat"] == "abcdefabcdef"
    assert "concat" not in result

    # Add messenger but not provider, so "hello-world" is unaccounted for:
    pipeline.add_dynamic_item(messenger)
    with pytest.raises(RuntimeError):
        pipeline.compute_specific(
            ["message"], {"text": "abc", "other-text": "def"}
        )
    # Now add provider, so that the unaccounted for hello-world key gets accounted for.
    pipeline.add_dynamic_item(provider)
    result = pipeline.compute_specific(
        ["message"], {"text": "abc", "other-text": "def"}
    )
    assert result["message"] == "hello-world, abc"
