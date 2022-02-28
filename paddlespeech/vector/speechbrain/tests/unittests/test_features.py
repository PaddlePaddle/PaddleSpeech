import paddle


def test_deltas(device):
    paddle.device.set_device(device)
    from speechbrain.processing.features import Deltas

    size = [10, 101, 20]
    inp = paddle.ones(shape=size)
    compute_deltas = Deltas(input_size=20)
    out = paddle.zeros(size)
    assert paddle.sum(compute_deltas(inp) == out) == out.numel()
    
    # 暂时不考虑动转静
    # assert paddle.jit.trace(compute_deltas, inp)


def test_context_window(device):
    paddle.device.set_device(device)
    from speechbrain.processing.features import ContextWindow

    inp = (
        paddle.to_tensor([1, 2, 3])
        .unsqueeze(0)
        .unsqueeze(-1)
        .astype("float32")
    )
    compute_cw = ContextWindow(left_frames=1, right_frames=1)
    out = (
        paddle.to_tensor([[0, 1, 2], [1, 2, 3], [2, 3, 0]])
        .unsqueeze(0)
        .astype("float32")
    )
    assert paddle.sum(compute_cw(inp) == out) == 9

    inp = paddle.rand([2, 10, 5])
    compute_cw = ContextWindow(left_frames=0, right_frames=0)
    assert paddle.sum(compute_cw(inp) == inp) == inp.numel()

    # assert paddle.jit.trace(compute_cw, inp)


def test_istft(device):
    paddle.device.set_device(device)
    from speechbrain.processing.features import STFT
    from speechbrain.processing.features import ISTFT

    fs = 16000
    inp = paddle.randn([10, 16000])
    inp = paddle.stack(3 * [inp], -1)

    compute_stft = STFT(sample_rate=fs)
    compute_istft = ISTFT(sample_rate=fs)
    out = compute_istft(compute_stft(inp), sig_length=16000)

    assert paddle.sum(paddle.abs(inp - out) < 5e-5) >= inp.numel() - 5

    # assert paddle.jit.trace(compute_stft, inp)
    # assert paddle.jit.trace(compute_istft, compute_stft(inp))


def test_filterbank(device):
    paddle.device.set_device(device)
    from speechbrain.processing.features import Filterbank

    compute_fbanks = Filterbank()
    inputs = paddle.ones([10, 101, 201])
    # assert paddle.jit.trace(compute_fbanks, inputs)

    # Check amin (-100 dB)
    inputs = paddle.zeros([10, 101, 201])
    fbanks = compute_fbanks(inputs)
    # assert 0
    assert paddle.equal_all(fbanks, paddle.ones_like(fbanks) * -100)

    # Check top_db
    fbanks = paddle.zeros([1, 1, 1])
    # 这里为什么得到的是int64类型
    expected = paddle.to_tensor([[[-100]]], dtype=fbanks.dtype)
    fbanks_db = compute_fbanks._amplitude_to_DB(fbanks)
    assert paddle.equal_all(fbanks_db, expected)

    # Making sure independent computation gives same results
    # as the batch computation
    input1 = paddle.rand([1, 101, 201]) * 10
    input2 = paddle.rand([1, 101, 201])
    input3 = paddle.concat([input1, input2], axis=0)
    fbank1 = compute_fbanks(input1)
    fbank2 = compute_fbanks(input2)
    fbank3 = compute_fbanks(input3)
    assert paddle.sum(paddle.abs(fbank1[0] - fbank3[0])) < 8e-05
    assert paddle.sum(paddle.abs(fbank2[0] - fbank3[1])) < 8e-05


def test_dtc(device):
    paddle.device.set_device(device)
    from speechbrain.processing.features import DCT

    compute_dct = DCT(input_size=40)
    inputs = paddle.randn([10, 101, 40])
    # 这里校验的是动转静
    # assert paddle.equal_all(compute_dct, inputs)
    # assert paddle.jit.trace(compute_dct, inputs)


def test_input_normalization(device):
    paddle.device.set_device(device)
    from speechbrain.processing.features import InputNormalization

    # norm = InputNormalization()
    # inputs = paddle.randn([10, 101, 20])
    # inp_len = paddle.ones([10])
    # assert paddle.jit.trace(norm, (inputs, inp_len))

    # norm = InputNormalization(norm_type="sentence")
    norm = InputNormalization()
    inputs = (
        paddle.to_tensor([1, 2, 3, 0, 0, 0], dtype="float32")
        .unsqueeze(0)
        .unsqueeze(2)
    )
    inp_len = paddle.to_tensor([0.5], dtype="float32")
    out_norm = norm(inputs, inp_len).squeeze()
    target = paddle.to_tensor([-1, 0, 1, -2, -2, -2], dtype="float32")
    assert paddle.equal_all(out_norm, target)


def test_features_multimic(device):
    paddle.device.set_device(device)
    from speechbrain.processing.features import Filterbank

    compute_fbanks = Filterbank()
    inputs = paddle.rand([10, 101, 201])
    output = compute_fbanks(inputs)
    inputs_ch2 = paddle.stack((inputs, inputs), -1)
    output_ch2 = compute_fbanks(inputs_ch2)
    output_ch2 = output_ch2[..., 0]
    assert paddle.sum(output - output_ch2) < 5e-05
