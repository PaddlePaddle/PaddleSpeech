import paddle
import paddle.nn


def test_linear(device):
    paddle.device.set_device(device)
    from speechbrain.nnet.linear import Linear

    inputs = paddle.rand([1, 2, 4], dtype="float32")
    lin_t = Linear(n_neurons=4, input_size=inputs.shape[-1], bias=False)
    # 使用set_value，去掉了创建Parameter的麻烦
    lin_t.w.weight.set_value(paddle.eye(inputs.shape[-1], dtype="float32"))
    
    outputs = lin_t(inputs)
    assert paddle.all(paddle.equal(inputs, outputs))

    # assert torch.jit.trace(lin_t, inputs)
