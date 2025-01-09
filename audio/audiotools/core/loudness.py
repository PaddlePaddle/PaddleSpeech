# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/core/loudness.py)
import copy
import math
import typing

import numpy as np
import paddle
import paddle.nn.functional as F
import scipy

from . import _julius


def _unfold1d(x, kernel_size, stride):
    # https://github.com/PaddlePaddle/Paddle/pull/70102
    """1D only unfolding similar to the one from Paddlepaddle.

    Given an _input tensor of size `[*, T]` this will return
    a tensor `[*, F, K]` with `K` the kernel size, and `F` the number
    of frames. The i-th frame is a view onto `i * stride: i * stride + kernel_size`.
    This will automatically pad the _input to cover at least once all entries in `_input`.

    Args:
        _input (Tensor): tensor for which to return the frames.
        kernel_size (int): size of each frame.
        stride (int): stride between each frame.

    Shape:

        - Inputs: `_input` is `[*, T]`
        - Output: `[*, F, kernel_size]` with `F = 1 + ceil((T - kernel_size) / stride)`
    """

    if 3 != x.dim():
        raise NotImplementedError

    N, C, length = x.shape
    x = x.reshape([N * C, 1, length])

    n_frames = math.ceil((max(length, kernel_size) - kernel_size) / stride) + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    x = F.pad(x, (0, tgt_length - length), data_format="NCL")

    x = x.unsqueeze(-1)

    unfolded = paddle.nn.functional.unfold(
        x,
        kernel_sizes=[kernel_size, 1],
        strides=[stride, 1], )

    unfolded = unfolded.transpose([0, 2, 1])
    unfolded = unfolded.reshape([N, C, *unfolded.shape[1:]])
    return unfolded


class Meter(paddle.nn.Layer):
    """Tensorized version of pyloudnorm.Meter. Works with batched audio tensors.

    Parameters
    ----------
    rate : int
        Sample rate of audio.
    filter_class : str, optional
        Class of weighting filter used.
        K-weighting' (default), 'Fenton/Lee 1'
        'Fenton/Lee 2', 'Dash et al.'
        by default "K-weighting"
    block_size : float, optional
        Gating block size in seconds, by default 0.400
    zeros : int, optional
         Number of zeros to use in FIR approximation of
         IIR filters, by default 512
    use_fir : bool, optional
        Whether to use FIR approximation or exact IIR formulation.
        If computing on GPU, ``use_fir=True`` will be used, as its
        much faster, by default False
    """

    def __init__(
            self,
            rate: int,
            filter_class: str="K-weighting",
            block_size: float=0.400,
            zeros: int=512,
            use_fir: bool=False, ):
        super().__init__()

        self.rate = rate
        self.filter_class = filter_class
        self.block_size = block_size
        self.use_fir = use_fir

        G = paddle.to_tensor(
            np.array([1.0, 1.0, 1.0, 1.41, 1.41]), stop_gradient=True)
        self.register_buffer("G", G)

        # Compute impulse responses so that filtering is fast via
        # a convolution at runtime, on GPU, unlike lfilter.
        impulse = np.zeros((zeros, ))
        impulse[..., 0] = 1.0

        firs = np.zeros((len(self._filters), 1, zeros))
        # passband_gain = torch.zeros(len(self._filters))
        passband_gain = paddle.zeros([len(self._filters)], dtype="float32")

        for i, (_, filter_stage) in enumerate(self._filters.items()):
            firs[i] = scipy.signal.lfilter(filter_stage.b, filter_stage.a,
                                           impulse)
            passband_gain[i] = filter_stage.passband_gain

        firs = paddle.to_tensor(
            firs[..., ::-1].copy(), dtype="float32", stop_gradient=True)

        self.register_buffer("firs", firs)
        self.register_buffer("passband_gain", passband_gain)

    def apply_filter_gpu(self, data: paddle.Tensor):
        """Performs FIR approximation of loudness computation.

        Parameters
        ----------
        data : paddle.Tensor
            Audio data of shape (nb, nch, nt).

        Returns
        -------
        paddle.Tensor
            Filtered audio data.
        """
        # Data is of shape (nb, nch, nt)
        # Reshape to (nb*nch, 1, nt)
        nb, nt, nch = data.shape
        data = data.transpose([0, 2, 1])
        data = data.reshape([nb * nch, 1, nt])

        # Apply padding
        pad_length = self.firs.shape[-1]

        # Apply filtering in sequence
        for i in range(self.firs.shape[0]):
            data = F.pad(data, (pad_length, pad_length), data_format="NCL")
            data = _julius.fft_conv1d(data, self.firs[i, None, ...])
            data = self.passband_gain[i] * data
            data = data[..., 1:nt + 1]

        data = data.transpose([0, 2, 1])
        data = data[:, :nt, :]
        return data

    @staticmethod
    def scipy_lfilter(waveform, a_coeffs, b_coeffs, clamp: bool=True):
        # 使用 scipy.signal.lfilter 进行滤波（处理三维数据）
        output = np.zeros_like(waveform)
        for batch_idx in range(waveform.shape[0]):
            for channel_idx in range(waveform.shape[2]):
                output[batch_idx, :, channel_idx] = scipy.signal.lfilter(
                    b_coeffs, a_coeffs, waveform[batch_idx, :, channel_idx])
        return output

    def apply_filter_cpu(self, data: paddle.Tensor):
        """Performs IIR formulation of loudness computation.

        Parameters
        ----------
        data : paddle.Tensor
            Audio data of shape (nb, nch, nt).

        Returns
        -------
        paddle.Tensor
            Filtered audio data.
        """
        _data = data.cpu().numpy().copy()
        for _, filter_stage in self._filters.items():
            passband_gain = filter_stage.passband_gain

            a_coeffs = filter_stage.a
            b_coeffs = filter_stage.b

            filtered = self.scipy_lfilter(_data, a_coeffs, b_coeffs)
            _data[:] = passband_gain * filtered
        data = paddle.to_tensor(_data)
        return data

    def apply_filter(self, data: paddle.Tensor):
        """Applies filter on either CPU or GPU, depending
        on if the audio is on GPU or is on CPU, or if
        ``self.use_fir`` is True.

        Parameters
        ----------
        data : paddle.Tensor
            Audio data of shape (nb, nch, nt).

        Returns
        -------
        paddle.Tensor
            Filtered audio data.
        """
        # if data.place.is_gpu_place() or self.use_fir:
        #     data = self.apply_filter_gpu(data)
        # else:
        #     data = self.apply_filter_cpu(data)
        data = self.apply_filter_cpu(data)
        return data

    def forward(self, data: paddle.Tensor):
        """Computes integrated loudness of data.

        Parameters
        ----------
        data : paddle.Tensor
            Audio data of shape (nb, nch, nt).

        Returns
        -------
        paddle.Tensor
            Filtered audio data.
        """
        return self.integrated_loudness(data)

    def _unfold(self, input_data):
        T_g = self.block_size
        overlap = 0.75  # overlap of 75% of the block duration
        step = 1.0 - overlap  # step size by percentage

        kernel_size = int(T_g * self.rate)
        stride = int(T_g * self.rate * step)
        unfolded = _unfold1d(
            input_data.transpose([0, 2, 1]), kernel_size, stride)
        unfolded = unfolded.transpose([0, 1, 3, 2])

        return unfolded

    def integrated_loudness(self, data: paddle.Tensor):
        """Computes integrated loudness of data.

        Parameters
        ----------
        data : paddle.Tensor
            Audio data of shape (nb, nch, nt).

        Returns
        -------
        paddle.Tensor
            Filtered audio data.
        """
        if not paddle.is_tensor(data):
            data = paddle.to_tensor(data, dtype="float32")
        else:
            data = data.astype("float32")

        input_data = data.clone()
        # Data always has a batch and channel dimension.
        # Is of shape (nb, nt, nch)
        if input_data.ndim < 2:
            input_data = input_data.unsqueeze(-1)
        if input_data.ndim < 3:
            input_data = input_data.unsqueeze(0)

        nb, nt, nch = input_data.shape

        # Apply frequency weighting filters - account
        # for the acoustic respose of the head and auditory system
        input_data = self.apply_filter(input_data)

        G = self.G  # channel gains
        T_g = self.block_size  # 400 ms gating block standard
        Gamma_a = -70.0  # -70 LKFS = absolute loudness threshold

        unfolded = self._unfold(input_data)

        z = (1.0 / (T_g * self.rate)) * unfolded.square().sum(2)
        l = -0.691 + 10.0 * paddle.log10(
            (G[None, :nch, None] * z).sum(1, keepdim=True))
        l = l.expand_as(z)

        # find gating block indices above absolute threshold
        z_avg_gated = z
        z_avg_gated[l <= Gamma_a] = 0
        masked = l > Gamma_a
        z_avg_gated = z_avg_gated.sum(2) / masked.sum(2).astype("float32")

        # calculate the relative threshold value (see eq. 6)
        Gamma_r = -0.691 + 10.0 * paddle.log10(
            (z_avg_gated * G[None, :nch]).sum(-1)) - 10.0
        Gamma_r = Gamma_r[:, None, None]
        Gamma_r = Gamma_r.expand([nb, nch, l.shape[-1]])

        # find gating block indices above relative and absolute thresholds  (end of eq. 7)
        z_avg_gated = z
        z_avg_gated[l <= Gamma_a] = 0
        z_avg_gated[l <= Gamma_r] = 0
        masked = (l > Gamma_a) * (l > Gamma_r)
        z_avg_gated = z_avg_gated.sum(2) / (masked.sum(2) + 10e-6)

        # TODO Currently, paddle has a segmentation fault bug in this section of the code
        # z_avg_gated = paddle.nan_to_num(z_avg_gated)
        # z_avg_gated = paddle.where(
        #     paddle.isnan(z_avg_gated),
        #     paddle.zeros_like(z_avg_gated), z_avg_gated)
        z_avg_gated[z_avg_gated == float("inf")] = float(
            np.finfo(np.float32).max)
        z_avg_gated[z_avg_gated == -float("inf")] = float(
            np.finfo(np.float32).min)

        LUFS = -0.691 + 10.0 * paddle.log10(
            (G[None, :nch] * z_avg_gated).sum(1))
        return LUFS.astype("float32")

    @property
    def filter_class(self):
        return self._filter_class

    @filter_class.setter
    def filter_class(self, value):
        from pyloudnorm import Meter

        meter = Meter(self.rate)
        meter.filter_class = value
        self._filter_class = value
        self._filters = meter._filters


class LoudnessMixin:
    _loudness = None
    MIN_LOUDNESS = -70
    """Minimum loudness possible."""

    def loudness(self,
                 filter_class: str="K-weighting",
                 block_size: float=0.400,
                 **kwargs):
        """Calculates loudness using an implementation of ITU-R BS.1770-4.
        Allows control over gating block size and frequency weighting filters for
        additional control. Measure the integrated gated loudness of a signal.

        API is derived from PyLoudnorm, but this implementation is ported to PyTorch
        and is tensorized across batches. When on GPU, an FIR approximation of the IIR
        filters is used to compute loudness for speed.

        Uses the weighting filters and block size defined by the meter
        the integrated loudness is measured based upon the gating algorithm
        defined in the ITU-R BS.1770-4 specification.

        Parameters
        ----------
        filter_class : str, optional
            Class of weighting filter used.
            K-weighting' (default), 'Fenton/Lee 1'
            'Fenton/Lee 2', 'Dash et al.'
            by default "K-weighting"
        block_size : float, optional
            Gating block size in seconds, by default 0.400
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.loudness.Meter`.

        Returns
        -------
        paddle.Tensor
            Loudness of audio data.
        """
        if self._loudness is not None:
            return self._loudness  # .to(self.device)
        original_length = self.signal_length
        if self.signal_duration < 0.5:
            pad_len = int((0.5 - self.signal_duration) * self.sample_rate)
            self.zero_pad(0, pad_len)

        # create BS.1770 meter
        meter = Meter(
            self.sample_rate,
            filter_class=filter_class,
            block_size=block_size,
            **kwargs)
        # meter = meter.to(self.device)
        # measure loudness
        loudness = meter.integrated_loudness(
            self.audio_data.transpose([0, 2, 1]))
        self.truncate_samples(original_length)
        min_loudness = paddle.ones_like(loudness) * self.MIN_LOUDNESS
        self._loudness = paddle.maximum(loudness, min_loudness)

        return self._loudness  # .to(self.device)
