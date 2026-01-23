import numpy as np
import paddle
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-05):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-05):
    return paddle.log(paddle.clip(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return paddle.exp(x=x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    y = paddle.to_tensor(y.detach().cpu().numpy())
    if paddle.min(paddle.to_tensor(y)) < -1.0:
        print("min value is ", paddle.min(paddle.to_tensor(y)))
    if paddle.max(paddle.to_tensor(y)) > 1.0:
        print("max value is ", paddle.max(paddle.to_tensor(y)))
    global mel_basis, hann_window
    if f"{str(fmax)}_{str(y.place)}" not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.place)] = (
            paddle.to_tensor(mel).float().to(y.place)
        )
        hann_window[str(y.place)] = paddle.audio.functional.get_window(
            win_length=win_size, dtype="float32", window="hann"
        ).to(y.place)
    
    y = paddle.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    window = paddle.load("/root/paddlejob/workspace/zhangjinghong/test/PaddleSpeech/matcha_window.pdparams").cuda()

    stft = paddle.signal.stft(
        y.cuda(), 
        n_fft=1920,
        hop_length=480,
        window=window
    )
    
    spec = paddle.as_real(
        stft
    )
    spec = paddle.sqrt(spec.pow(2).sum(-1) + 1e-09)
    spec = paddle.matmul(mel_basis[str(fmax) + "_" + str(y.place)], spec)
    spec = spectral_normalize_torch(spec)
    return spec
