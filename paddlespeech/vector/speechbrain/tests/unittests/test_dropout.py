import paddle
import paddle.nn


def test_dropout(device):

    from speechbrain.nnet.dropout import Dropout2d

    inputs = paddle.rand([4, 10, 32], )
    drop = Dropout2d(drop_rate=0.0)
    outputs = drop(inputs)
    assert torch.all(torch.eq(inputs, outputs))

    drop = Dropout2d(drop_rate=1.0)
    outputs = drop(inputs)
    assert paddle.all(
        torch.equal(torch.zeros(inputs.shape, ), outputs)
    )

    # assert torch.jit.trace(drop, inputs)
