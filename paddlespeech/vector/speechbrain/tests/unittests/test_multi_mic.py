import paddle


def test_gccphat(device):
    paddle.device.set_device(device)
    from speechbrain.processing.features import STFT
    from speechbrain.processing.multi_mic import Covariance, GccPhat

    # Creating the test signal
    fs = 16000

    delay = 60

    sig = paddle.randn([10, fs], )
    sig_delayed = paddle.concat(
        (paddle.zeros([10, delay], ), sig[:, 0:-delay]), 1
    )

    xs = paddle.stack((sig_delayed, sig), -1)

    stft = STFT(sample_rate=fs)
    Xs = stft(xs)

    # Computing the covariance matrix for GCC-PHAT
    cov = Covariance().to(device)
    gccphat = GccPhat().to(device)

    XXs = cov(Xs).to(device)
    tdoas = paddle.abs(gccphat(XXs))

    n_valid_tdoas = paddle.sum(paddle.abs(tdoas[..., 1] - delay) < 1e-3)
    assert n_valid_tdoas == Xs.shape[0] * Xs.shape[1]
    # assert paddle.jit.trace(stft, xs)
    # assert paddle.jit.trace(cov, Xs)
    # assert paddle.jit.trace(gccphat, XXs)
