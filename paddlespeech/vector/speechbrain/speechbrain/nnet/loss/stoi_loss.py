# ################################
# From paper: "End-to-End Waveform Utterance Enhancement for Direct Evaluation
# Metrics Optimization by Fully Convolutional Neural Networks", TASLP, 2018
# Authors: Szu-Wei, Fu 2020
# ################################

import paddle
import numpy as np
from speechbrain.utils.torch_audio_backend import check_torchaudio_backend

check_torchaudio_backend()
smallVal = np.finfo("float").eps  # To avoid divide by zero


def thirdoct(fs, nfft, num_bands, min_freq):
    """Returns the 1/3 octave band matrix.

    Arguments
    ---------
    fs : int
        Sampling rate.
    nfft : int
        FFT size.
    num_bands : int
        Number of 1/3 octave bands.
    min_freq : int
        Center frequency of the lowest 1/3 octave band.

    Returns
    -------
    obm : tensor
        Octave Band Matrix.
    """

    f = torch.linspace(0, fs, nfft + 1)
    f = f[: int(nfft / 2) + 1]
    k = torch.from_numpy(np.array(range(num_bands)).astype(float))
    cf = torch.pow(2.0 ** (1.0 / 3), k) * min_freq
    freq_low = min_freq * torch.pow(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * torch.pow(2.0, (2 * k + 1) / 6)
    obm = torch.zeros(num_bands, len(f))  # a verifier

    for i in range(len(cf)):
        # Match 1/3 oct band freq with fft frequency bin
        f_bin = torch.argmin(torch.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = torch.argmin(torch.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        # Assign to the octave band matrix
        obm[i, fl_ii:fh_ii] = 1
    return obm


def removeSilentFrames(x, y, dyn_range=40, N=256, K=128):
    w = torch.unsqueeze(torch.from_numpy(np.hanning(256)), 0).to(torch.float)

    X1 = x[0 : int(x.shape[0]) // N * N].reshape(int(x.shape[0]) // N, N).T
    X2 = (
        x[128 : (int(x.shape[0]) - 128) // N * N + 128]
        .reshape((int(x.shape[0]) - 128) // N, N)
        .T
    )
    X = torch.zeros(N, X1.shape[1] + X2.shape[1])
    X[:, 0::2] = X1
    X[:, 1::2] = X2

    energy = 20 * torch.log10(
        torch.sqrt(torch.matmul(w ** 2, X ** 2)) / 16.0 + smallVal
    )

    Max_energy = torch.max(energy)
    msk = torch.squeeze((energy - Max_energy + dyn_range > 0))

    Y1 = y[0 : int(y.shape[0]) // N * N].reshape(int(y.shape[0]) // N, N).T
    Y2 = (
        y[128 : (int(y.shape[0]) - 128) // N * N + 128]
        .reshape((int(y.shape[0]) - 128) // N, N)
        .T
    )
    Y = torch.zeros(N, Y1.shape[1] + Y2.shape[1])
    Y[:, 0::2] = Y1
    Y[:, 1::2] = Y2

    x_sil = w.T.repeat(1, X[:, msk].shape[-1]) * X[:, msk]
    y_sil = w.T.repeat(1, X[:, msk].shape[-1]) * Y[:, msk]

    x_sil = torch.cat(
        (
            x_sil[0:128, 0],
            (x_sil[0:128, 1:] + x_sil[128:, 0:-1]).T.flatten(),
            x_sil[128:256, -1],
        ),
        axis=0,
    )
    y_sil = torch.cat(
        (
            y_sil[0:128, 0],
            (y_sil[0:128, 1:] + y_sil[128:, 0:-1]).T.flatten(),
            y_sil[128:256, -1],
        ),
        axis=0,
    )

    return [x_sil, y_sil]


def stoi_loss(y_pred_batch, y_true_batch, lens, reduction="mean"):
    """Compute the STOI score and return -1 * that score.

    This function can be used as a loss function for training
    with SGD-based updates.

    Arguments
    ---------
    y_pred_batch : paddle.Tensor
        The degraded (enhanced) waveforms.
    y_true_batch : paddle.Tensor
        The clean (reference) waveforms.
    lens : paddle.Tensor
        The relative lengths of the waveforms within the batch.
    reduction : str
        The type of reduction ("mean" or "batch") to use.

    Example
    -------
    >>> a = torch.sin(torch.arange(16000, dtype=torch.float32)).unsqueeze(0)
    >>> b = a + 0.001
    >>> -stoi_loss(b, a, torch.ones(1))
    tensor(0.7...)
    """

    y_pred_batch = torch.squeeze(y_pred_batch, dim=-1)
    y_true_batch = torch.squeeze(y_true_batch, dim=-1)

    batch_size = y_pred_batch.shape[0]

    fs = 16000  # Sampling rate
    N = 30  # length of temporal envelope vectors
    J = 15.0  # Number of one-third octave bands

    octave_band = thirdoct(fs=10000, nfft=512, num_bands=15, min_freq=150)
    c = 5.62341325  # 10^(-Beta/20) with Beta = -15
    D = torch.zeros(batch_size)
    resampler = torchaudio.transforms.Resample(fs, 10000).to(
        y_pred_batch.device
    )

    for i in range(0, batch_size):  # Run over mini-batches
        y_true = y_true_batch[i, 0 : int(lens[i] * y_pred_batch.shape[1])]
        y_pred = y_pred_batch[i, 0 : int(lens[i] * y_pred_batch.shape[1])]

        y_true, y_pred = resampler(y_true), resampler(y_pred)

        [y_sil_true, y_sil_pred] = removeSilentFrames(y_true, y_pred)

        stft_true = torchaudio.transforms.Spectrogram(
            n_fft=512, win_length=256, hop_length=128, power=2
        )(y_sil_true)
        stft_pred = torchaudio.transforms.Spectrogram(
            n_fft=512, win_length=256, hop_length=128, power=2
        )(y_sil_pred)

        OCT_true = torch.sqrt(torch.matmul(octave_band, stft_true) + 1e-14)
        OCT_pred = torch.sqrt(torch.matmul(octave_band, stft_pred) + 1e-14)

        M = int(
            stft_pred.shape[-1] - (N - 1)
        )  # number of temporal envelope vectors

        X = torch.zeros(15 * M, 30)
        Y = torch.zeros(15 * M, 30)
        for m in range(0, M):  # Run over temporal envelope vectors
            X[m * 15 : (m + 1) * 15, :] = OCT_true[:, m : m + N]
            Y[m * 15 : (m + 1) * 15, :] = OCT_pred[:, m : m + N]

        alpha = torch.norm(X, dim=-1, keepdim=True) / (
            torch.norm(Y, dim=-1, keepdim=True) + smallVal
        )

        ay = Y * alpha
        y = torch.min(ay, X + X * c)

        xn = X - torch.mean(X, dim=-1, keepdim=True)
        xn = xn / (torch.norm(xn, dim=-1, keepdim=True) + smallVal)

        yn = y - torch.mean(y, dim=-1, keepdim=True)
        yn = yn / (torch.norm(yn, dim=-1, keepdim=True) + smallVal)
        d = torch.sum(xn * yn)
        D[i] = d / (J * M)

    if reduction == "mean":
        return -D.mean()

    return -D
