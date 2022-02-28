import paddle
import paddle.nn


def test_softmax(device):
    paddle.device.set_device(device)
    from speechbrain.nnet.activations import Softmax

    inputs = paddle.to_tensor([1, 2, 3], dtype="float32")
    act = Softmax(apply_log=False)
    outputs = act(inputs)
    assert paddle.argmax(outputs) == 2

    # assert torch.jit.trace(act, inputs)
