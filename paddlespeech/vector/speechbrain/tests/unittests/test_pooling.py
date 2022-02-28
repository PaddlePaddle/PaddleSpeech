import paddle
import paddle.nn


def test_pooling1d(device):
    paddle.device.set_device(device)
    from speechbrain.nnet.pooling import Pooling1d

    input = (
        paddle.to_tensor([1, 3, 2], )
        .unsqueeze(0)
        .unsqueeze(-1)
        .astype("float32")
    )
    pool = Pooling1d("max", 3).to(device)
    output = pool(input)
    assert output == 3

    pool = Pooling1d("avg", 3).to(device)
    output = pool(input)
    assert output == 2

    # assert paddle.jit.trace(pool, input)


def test_pooling2d(device):
    paddle.device.set_device(device)
    from speechbrain.nnet.pooling import Pooling2d

    input = (
        paddle.to_tensor([[1, 3, 2], [4, 6, 5]], ).astype("float32").unsqueeze(0)
    )
    pool = Pooling2d("max", (2, 3)).to(device)
    output = pool(input)
    assert output == 6

    input = (
        paddle.to_tensor([[1, 3, 2], [4, 6, 5]], ).astype("float32").unsqueeze(0)
    )
    pool = Pooling2d("max", (1, 3)).to(device)
    output = pool(input)
    assert output[0][0] == 3
    assert output[0][1] == 6

    input = (
        paddle.to_tensor([[1, 3, 2], [4, 6, 5]], ).astype("float32").unsqueeze(0)
    )
    pool = Pooling2d("avg", (2, 3)).to(device)
    output = pool(input)
    assert output == 3.5

    input = (
        paddle.to_tensor([[1, 3, 2], [4, 6, 5]], ).astype("float32").unsqueeze(0)
    )
    pool = Pooling2d("avg", (1, 3)).to(device)
    output = pool(input)
    assert output[0][0] == 2
    assert output[0][1] == 5

    # assert paddle.jit.trace(pool, input)
