import pytest
import paddle
import numpy as np


def test_batch_pad_right_to(device):
    from speechbrain.utils.data_utils import batch_pad_right
    import random
    paddle.device.set_device("cpu")
    n_channels = 40
    batch_lens = [1, 5]

    for b in batch_lens:
        rand_lens = [random.randint(10, 53) for x in range(b)]
        tensors = [
            paddle.ones((rand_lens[x], n_channels))
            for x in range(b)
        ]
        batched, lens = batch_pad_right(tensors)
        assert batched.shape[0] == b
        np.testing.assert_almost_equal(
            lens, [x / max(rand_lens) for x in rand_lens], decimal=3
        )

    for b in batch_lens:
        rand_lens = [random.randint(10, 53) for x in range(b)]
        tensors = [paddle.ones([rand_lens[x]]) for x in range(b)]
        batched, lens = batch_pad_right(tensors)
        assert batched.shape[0] == b
        np.testing.assert_almost_equal(
            lens, [x / max(rand_lens) for x in rand_lens], decimal=3
        )


def test_paddedbatch(device):
    from speechbrain.dataio.batch import PaddedBatch
    paddle.device.set_device("cpu")
    batch = PaddedBatch(
        [
            {
                "id": "ex1",
                "foo": paddle.to_tensor([1.0]),
                "bar": paddle.to_tensor([1.0, 2.0, 3.0]),
            },
            {
                "id": "ex2",
                "foo": paddle.to_tensor([2.0, 1.0]),
                "bar": paddle.to_tensor([2.0]),
            },
        ]
    )
    # batch = paddle.to_tensor(batch, dtype="float16")
    batch.to(dtype="float16")
    assert batch.foo.data.dtype == paddle.float16
    assert batch["foo"][1].dtype == paddle.float16
    assert batch.bar.lengths.dtype == paddle.float16
    assert batch.foo.data.shape == [2, 2]
    assert batch.bar.data.shape == [2, 3]
    ids, foos, bars = batch
    assert ids == ["ex1", "ex2"]


@pytest.mark.skipif(not paddle.is_compiled_with_cuda, reason="Requires CUDA")
def test_pin_memory():
    from speechbrain.dataio.batch import PaddedBatch

    batch = PaddedBatch(
        [
            {
                "id": "ex1",
                "foo": paddle.to_tensor([1.0]),
                "bar": paddle.to_tensor([1.0, 2.0, 3.0]),
            },
            {
                "id": "ex2",
                "foo": paddle.to_tensor([2.0, 1.0]),
                "bar": paddle.to_tensor([2.0]),
            },
        ]
    )
    batch.pin_memory()
    ref = paddle.to_tensor([1.0]).pin_memory()
    print("place type: {}".format(type(ref.place)))
    print("paddle pind: {}".format(type(batch.foo.data.place)))
    # todo: 需要确定 CUDAPinnedPlace 没有通过
    # assert batch.foo.data.place == ref.place
