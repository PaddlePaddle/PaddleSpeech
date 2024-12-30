# Copyright (c) 2023 speechbrain Authors. All Rights Reserved.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from speechbrain(https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/processing/speech_augmentation.py)
"""Classes for mutating speech data for data augmentation.
This module provides classes that produce realistic distortions of speech
data for the purpose of training speech processing models. The list of
distortions includes adding noise, adding reverberation, changing speed,
and more. All the classes are of type `torch.nn.Module`. This gives the
possibility to have end-to-end differentiability and
backpropagate the gradient through them. In addition, all operations
are expected to be performed on the GPU (where available) for efficiency.

Authors
 * Peter Plantinga 2020
"""
import math

import paddle
import paddle.nn as nn

from paddlespeech.s2t.models.wav2vec2.processing.signal_processing import compute_amplitude
from paddlespeech.s2t.models.wav2vec2.processing.signal_processing import convolve1d
from paddlespeech.s2t.models.wav2vec2.processing.signal_processing import notch_filter


class SpeedPerturb(nn.Layer):
    """Slightly speed up or slow down an audio signal.
    Resample the audio signal at a rate that is similar to the original rate,
    to achieve a slightly slower or slightly faster signal. This technique is
    outlined in the paper: "Audio Augmentation for Speech Recognition"
    Arguments
    ---------
    orig_freq : int
        The frequency of the original signal.
    speeds : list
        The speeds that the signal should be changed to, as a percentage of the
        original signal (i.e. `speeds` is divided by 100 to get a ratio).
    perturb_prob : float
        The chance that the batch will be speed-
        perturbed. By default, every batch is perturbed.
    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> perturbator = SpeedPerturb(orig_freq=16000, speeds=[90])
    >>> clean = signal.unsqueeze(0)
    >>> perturbed = perturbator(clean)
    >>> clean.shape
    paddle.shape([1, 52173])
    >>> perturbed.shape
    paddle.shape([1, 46956])
    """

    def __init__(
            self,
            orig_freq,
            speeds=[90, 100, 110],
            perturb_prob=1.0, ):
        super().__init__()
        self.orig_freq = orig_freq
        self.speeds = speeds
        self.perturb_prob = perturb_prob

        # Initialize index of perturbation
        self.samp_index = 0
        # Initialize resamplers
        self.resamplers = []
        for speed in self.speeds:
            config = {
                "orig_freq": self.orig_freq,
                "new_freq": self.orig_freq * speed // 100,
            }
            self.resamplers.append(Resample(**config))

    def forward(self, waveform):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.
        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't perturb (return early) 1-`perturb_prob` portion of the batches
        if paddle.rand([1]) > self.perturb_prob:
            return waveform.clone()
        # Perform a random perturbation
        self.samp_index = paddle.randint(len(self.speeds), shape=(1, ))[0]
        perturbed_waveform = self.resamplers[self.samp_index](waveform)

        return perturbed_waveform


class Resample(nn.Layer):
    """This class resamples an audio signal using sinc-based interpolation.

    It is a modification of the `resample` function from torchaudio
    (https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html)

    Arguments
    ---------
    orig_freq : int
        the sampling frequency of the input signal.
    new_freq : int
        the new sampling frequency after this operation is performed.
    lowpass_filter_width : int
        Controls the sharpness of the filter, larger numbers result in a
        sharper filter, but they are less efficient. Values from 4 to 10 are allowed.
    """

    def __init__(
            self,
            orig_freq=16000,
            new_freq=16000,
            lowpass_filter_width=6, ):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.lowpass_filter_width = lowpass_filter_width

        # Compute rate for striding
        self._compute_strides()
        assert self.orig_freq % self.conv_stride == 0
        assert self.new_freq % self.conv_transpose_stride == 0

    def _compute_strides(self):
        """Compute the phases in polyphase filter.

        (almost directly from torchaudio.compliance.kaldi)
        """

        # Compute new unit based on ratio of in/out frequencies
        base_freq = math.gcd(self.orig_freq, self.new_freq)
        input_samples_in_unit = self.orig_freq // base_freq
        self.output_samples = self.new_freq // base_freq

        # Store the appropriate stride based on the new units
        self.conv_stride = input_samples_in_unit
        self.conv_transpose_stride = self.output_samples

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        if not hasattr(self, "first_indices"):
            self._indices_and_weights(waveforms)

        # Don't do anything if the frequencies are the same
        if self.orig_freq == self.new_freq:
            return waveforms
        unsqueezed = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(1)
            unsqueezed = True
        elif len(waveforms.shape) == 3:
            waveforms = waveforms.transpose([0, 2, 1])
        else:
            raise ValueError("Input must be 2 or 3 dimensions")

        # Do resampling
        resampled_waveform = self._perform_resample(waveforms)

        if unsqueezed:
            resampled_waveform = resampled_waveform.squeeze(1)
        else:
            resampled_waveform = resampled_waveform.transpose([0, 2, 1])

        return resampled_waveform

    def _perform_resample(self, waveforms):
        """Resamples the waveform at the new frequency.

        This matches Kaldi's OfflineFeatureTpl ResampleWaveform which uses a
        LinearResample (resample a signal at linearly spaced intervals to
        up/downsample a signal). LinearResample (LR) means that the output
        signal is at linearly spaced intervals (i.e the output signal has a
        frequency of `new_freq`). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        https://ccrma.stanford.edu/~jos/resample/
        Theory_Ideal_Bandlimited_Interpolation.html

        https://github.com/kaldi-asr/kaldi/blob/master/src/feat/resample.h#L56

        Arguments
        ---------
        waveforms : tensor
            The batch of audio signals to resample.

        Returns
        -------
        The waveforms at the new frequency.
        """

        # Compute output size and initialize
        batch_size, num_channels, wave_len = waveforms.shape
        window_size = self.weights.shape[1]
        tot_output_samp = self._output_samples(wave_len)
        resampled_waveform = paddle.zeros(
            (batch_size, num_channels, tot_output_samp))
        # self.weights = self.weights.to(waveforms.device)

        # Check weights are on correct device
        # if waveforms.device != self.weights.device:
        #     self.weights = self.weights.to(waveforms.device)

        # eye size: (num_channels, num_channels, 1)
        eye = paddle.eye(num_channels).unsqueeze(2)

        # Iterate over the phases in the polyphase filter
        for i in range(self.first_indices.shape[0]):
            wave_to_conv = waveforms
            first_index = int(self.first_indices[i].item())
            if first_index >= 0:
                # trim the signal as the filter will not be applied
                # before the first_index
                wave_to_conv = wave_to_conv[..., first_index:]

            # pad the right of the signal to allow partial convolutions
            # meaning compute values for partial windows (e.g. end of the
            # window is outside the signal length)
            max_index = (tot_output_samp - 1) // self.output_samples
            end_index = max_index * self.conv_stride + window_size
            current_wave_len = wave_len - first_index
            right_padding = max(0, end_index + 1 - current_wave_len)
            left_padding = max(0, -first_index)
            wave_to_conv = paddle.nn.functional.pad(
                wave_to_conv, (left_padding, right_padding), data_format='NCL')
            conv_wave = paddle.nn.functional.conv1d(
                x=wave_to_conv,
                weight=self.weights[i].repeat(num_channels, 1, 1),
                stride=self.conv_stride,
                groups=num_channels, )

            # we want conv_wave[:, i] to be at
            # output[:, i + n*conv_transpose_stride]
            dilated_conv_wave = paddle.nn.functional.conv1d_transpose(
                conv_wave, eye, stride=self.conv_transpose_stride)

            # pad dilated_conv_wave so it reaches the output length if needed.
            left_padding = i
            previous_padding = left_padding + dilated_conv_wave.shape[-1]
            right_padding = max(0, tot_output_samp - previous_padding)
            dilated_conv_wave = paddle.nn.functional.pad(
                dilated_conv_wave, (left_padding, right_padding),
                data_format='NCL')
            dilated_conv_wave = dilated_conv_wave[..., :tot_output_samp]

            resampled_waveform += dilated_conv_wave

        return resampled_waveform

    def _output_samples(self, input_num_samp):
        """Based on LinearResample::GetNumOutputSamples.

        LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a
        frequency of ``new_freq``). It uses sinc/bandlimited
        interpolation to upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        Arguments
        ---------
        input_num_samp : int
            The number of samples in each example in the batch.

        Returns
        -------
        Number of samples in the output waveform.
        """

        # For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
        # where tick_freq is the least common multiple of samp_in and
        # samp_out.
        samp_in = int(self.orig_freq)
        samp_out = int(self.new_freq)

        tick_freq = abs(samp_in * samp_out) // math.gcd(samp_in, samp_out)
        ticks_per_input_period = tick_freq // samp_in

        # work out the number of ticks in the time interval
        # [ 0, input_num_samp/samp_in ).
        interval_length = input_num_samp * ticks_per_input_period
        if interval_length <= 0:
            return 0
        ticks_per_output_period = tick_freq // samp_out

        # Get the last output-sample in the closed interval,
        # i.e. replacing [ ) with [ ]. Note: integer division rounds down.
        # See http://en.wikipedia.org/wiki/Interval_(mathematics) for an
        # explanation of the notation.
        last_output_samp = interval_length // ticks_per_output_period

        # We need the last output-sample in the open interval, so if it
        # takes us to the end of the interval exactly, subtract one.
        if last_output_samp * ticks_per_output_period == interval_length:
            last_output_samp -= 1

        # First output-sample index is zero, so the number of output samples
        # is the last output-sample plus one.
        num_output_samp = last_output_samp + 1

        return num_output_samp

    def _indices_and_weights(self, waveforms):
        """Based on LinearResample::SetIndexesAndWeights

        Retrieves the weights for resampling as well as the indices in which
        they are valid. LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a frequency
        of ``new_freq``). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        Returns
        -------
        - the place where each filter should start being applied
        - the filters to be applied to the signal for resampling
        """

        # Lowpass filter frequency depends on smaller of two frequencies
        min_freq = min(self.orig_freq, self.new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq
        window_width = self.lowpass_filter_width / (2.0 * lowpass_cutoff)

        assert lowpass_cutoff < min(self.orig_freq, self.new_freq) / 2
        output_t = paddle.arange(
            start=0.0, end=self.output_samples, dtype='int64')
        output_t /= self.new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = paddle.ceil(min_t * self.orig_freq)
        max_input_index = paddle.floor(max_t * self.orig_freq)
        num_indices = max_input_index - min_input_index + 1

        max_weight_width = num_indices.max()
        j = paddle.arange(max_weight_width)
        input_index = min_input_index.unsqueeze(1) + j.unsqueeze(0)
        delta_t = (input_index / self.orig_freq) - output_t.unsqueeze(1)

        weights = paddle.zeros_like(delta_t)

        inside_window_indices = delta_t.abs() < (window_width)
        # raised-cosine (Hanning) window with width `window_width`
        weights[inside_window_indices] = 0.5 * (1 + paddle.cos(
            2 * math.pi * lowpass_cutoff / self.lowpass_filter_width *
            delta_t[inside_window_indices]))
        t_eq_zero_indices = delta_t == 0.0
        t_not_eq_zero_indices = ~t_eq_zero_indices

        # sinc filter function
        weights[t_not_eq_zero_indices] *= paddle.sin(
            2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]) / (
                math.pi * delta_t[t_not_eq_zero_indices])

        # limit of the function at t = 0
        weights[t_eq_zero_indices] *= 2 * lowpass_cutoff

        # size (output_samples, max_weight_width)
        weights /= self.orig_freq

        self.first_indices = min_input_index
        self.weights = weights


class DropFreq(nn.Layer):
    """This class drops a random frequency from the signal.
    The purpose of this class is to teach models to learn to rely on all parts
    of the signal, not just a few frequency bands.
    Arguments
    ---------
    drop_freq_low : float
        The low end of frequencies that can be dropped,
        as a fraction of the sampling rate / 2.
    drop_freq_high : float
        The high end of frequencies that can be
        dropped, as a fraction of the sampling rate / 2.
    drop_count_low : int
        The low end of number of frequencies that could be dropped.
    drop_count_high : int
        The high end of number of frequencies that could be dropped.
    drop_width : float
        The width of the frequency band to drop, as
        a fraction of the sampling_rate / 2.
    drop_prob : float
        The probability that the batch of signals will  have a frequency
        dropped. By default, every batch has frequencies dropped.
    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> dropper = DropFreq()
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> dropped_signal = dropper(signal.unsqueeze(0))
    """

    def __init__(
            self,
            drop_freq_low=1e-14,
            drop_freq_high=1,
            drop_count_low=1,
            drop_count_high=2,
            drop_width=0.05,
            drop_prob=1, ):
        super().__init__()
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_width = drop_width
        self.drop_prob = drop_prob

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        dropped_waveform = waveforms.clone()
        if paddle.rand([1]) > self.drop_prob:
            return dropped_waveform

        # Add channels dimension
        if len(waveforms.shape) == 2:
            dropped_waveform = dropped_waveform.unsqueeze(-1)

        # Pick number of frequencies to drop
        drop_count = paddle.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            shape=(1, ), )

        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # Start with delta function
        drop_filter = paddle.zeros([1, filter_length, 1])
        drop_filter[0, pad, 0] = 1

        if drop_count.shape == 0:
            # Pick a frequency to drop
            drop_range = self.drop_freq_high - self.drop_freq_low
            drop_frequency = (
                paddle.rand(drop_count) * drop_range + self.drop_freq_low)
            # Subtract each frequency
            for frequency in drop_frequency:
                notch_kernel = notch_filter(
                    frequency,
                    filter_length,
                    self.drop_width, )
                drop_filter = convolve1d(drop_filter, notch_kernel, pad)

        # Apply filter
        dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

        # Remove channels dimension if added
        return dropped_waveform.squeeze(-1)


class DropChunk(nn.Layer):
    """This class drops portions of the input signal.
    Using `DropChunk` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.
    Arguments
    ---------
    drop_length_low : int
        The low end of lengths for which to set the
        signal to zero, in samples.
    drop_length_high : int
        The high end of lengths for which to set the
        signal to zero, in samples.
    drop_count_low : int
        The low end of number of times that the signal
        can be dropped to zero.
    drop_count_high : int
        The high end of number of times that the signal
        can be dropped to zero.
    drop_start : int
        The first index for which dropping will be allowed.
    drop_end : int
        The last index for which dropping will be allowed.
    drop_prob : float
        The probability that the batch of signals will
        have a portion dropped. By default, every batch
        has portions dropped.
    noise_factor : float
        The factor relative to average amplitude of an utterance
        to use for scaling the white noise inserted. 1 keeps
        the average amplitude the same, while 0 inserts all 0's.
    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> dropper = DropChunk(drop_start=100, drop_end=200, noise_factor=0.)
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> signal = signal.unsqueeze(0) # [batch, time, channels]
    >>> length = paddle.ones([1])
    >>> dropped_signal = dropper(signal, length)
    >>> float(dropped_signal[:, 150])
    0.0
    """

    def __init__(
            self,
            drop_length_low=100,
            drop_length_high=1000,
            drop_count_low=1,
            drop_count_high=10,
            drop_start=0,
            drop_end=None,
            drop_prob=1,
            noise_factor=0.0, ):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.drop_prob = drop_prob
        self.noise_factor = noise_factor

        # Validate low < high
        if drop_length_low > drop_length_high:
            raise ValueError("Low limit must not be more than high limit")
        if drop_count_low > drop_count_high:
            raise ValueError("Low limit must not be more than high limit")

        # Make sure the length doesn't exceed end - start
        if drop_end is not None and drop_end >= 0:
            if drop_start > drop_end:
                raise ValueError("Low limit must not be more than high limit")

            drop_range = drop_end - drop_start
            self.drop_length_low = min(drop_length_low, drop_range)
            self.drop_length_high = min(drop_length_high, drop_range)

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.
        Returns
        -------
        Tensor of shape `[batch, time]` or
            `[batch, time, channels]`
        """

        # Reading input list
        lengths = (lengths * waveforms.shape[1]).long()
        batch_size = waveforms.shape[0]
        dropped_waveform = waveforms.clone()

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        if paddle.rand([1]) > self.drop_prob:
            return dropped_waveform

        # Store original amplitude for computing white noise amplitude
        clean_amplitude = compute_amplitude(waveforms, lengths.unsqueeze(1))

        # Pick a number of times to drop
        drop_times = paddle.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            shape=(batch_size, ), )

        # Iterate batch to set mask
        for i in range(batch_size):
            if drop_times[i] == 0:
                continue

            # Pick lengths
            length = paddle.randint(
                low=self.drop_length_low,
                high=self.drop_length_high + 1,
                shape=(drop_times[i], ), )

            # Compute range of starting locations
            start_min = self.drop_start
            if start_min < 0:
                start_min += lengths[i]
            start_max = self.drop_end
            if start_max is None:
                start_max = lengths[i]
            if start_max < 0:
                start_max += lengths[i]
            start_max = max(0, start_max - length.max())

            # Pick starting locations
            start = paddle.randint(
                low=start_min,
                high=start_max + 1,
                shape=(drop_times[i], ), )

            end = start + length

            # Update waveform
            if not self.noise_factor:
                for j in range(drop_times[i]):
                    dropped_waveform[i, start[j]:end[j]] = 0.0
            else:
                # Uniform distribution of -2 to +2 * avg amplitude should
                # preserve the average for normalization
                noise_max = 2 * clean_amplitude[i] * self.noise_factor
                for j in range(drop_times[i]):
                    # zero-center the noise distribution
                    noise_vec = paddle.rand([length[j]])
                    noise_vec = 2 * noise_max * noise_vec - noise_max
                    dropped_waveform[i, start[j]:end[j]] = noise_vec

        return dropped_waveform


class SpecAugment(paddle.nn.Layer):
    """An implementation of the SpecAugment algorithm.
    Reference:
        https://arxiv.org/abs/1904.08779
    Arguments
    ---------
    time_warp : bool
        Whether applying time warping.
    time_warp_window : int
        Time warp window.
    time_warp_mode : str
        Interpolation mode for time warping (default "bicubic").
    freq_mask : bool
        Whether applying freq mask.
    freq_mask_width : int or tuple
        Freq mask width range.
    n_freq_mask : int
        Number of freq mask.
    time_mask : bool
        Whether applying time mask.
    time_mask_width : int or tuple
        Time mask width range.
    n_time_mask : int
        Number of time mask.
    replace_with_zero : bool
        If True, replace masked value with 0, else replace masked value with mean of the input tensor.
    Example
    -------
    >>> aug = SpecAugment()
    >>> a = paddle.rand([8, 120, 80])
    >>> a = aug(a)
    >>> print(a.shape)
    paddle.Size([8, 120, 80])
    """

    def __init__(
            self,
            time_warp=True,
            time_warp_window=5,
            time_warp_mode="bicubic",
            freq_mask=True,
            freq_mask_width=(0, 20),
            n_freq_mask=2,
            time_mask=True,
            time_mask_width=(0, 100),
            n_time_mask=2,
            replace_with_zero=True, ):
        super().__init__()
        assert (
            time_warp or freq_mask or time_mask
        ), "at least one of time_warp, time_mask, or freq_mask should be applied"

        self.apply_time_warp = time_warp
        self.time_warp_window = time_warp_window
        self.time_warp_mode = time_warp_mode

        self.freq_mask = freq_mask
        if isinstance(freq_mask_width, int):
            freq_mask_width = (0, freq_mask_width)
        self.freq_mask_width = freq_mask_width
        self.n_freq_mask = n_freq_mask

        self.time_mask = time_mask
        if isinstance(time_mask_width, int):
            time_mask_width = (0, time_mask_width)
        self.time_mask_width = time_mask_width
        self.n_time_mask = n_time_mask

        self.replace_with_zero = replace_with_zero

    def forward(self, x):
        """Takes in input a tensors and returns an augmented one."""
        if self.apply_time_warp:
            x = self.time_warp(x)
        if self.freq_mask:
            x = self.mask_along_axis(x, dim=2)
        if self.time_mask:
            x = self.mask_along_axis(x, dim=1)
        return x

    def time_warp(self, x):
        """Time warping with paddle.nn.functional.interpolate"""
        original_size = x.shape
        window = self.time_warp_window

        # 2d interpolation requires 4D or higher dimension tensors
        # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        time = x.shape[2]
        if time - window <= window:
            return x.reshape([*original_size])

        # compute center and corresponding window
        c = paddle.randint(window, time - window, (1, ))[0]
        w = paddle.randint(c - window, c + window, (1, ))[0] + 1

        left = paddle.nn.functional.interpolate(
            x[:, :, :c],
            (w, x.shape[3]),
            mode=self.time_warp_mode,
            align_corners=True, )
        right = paddle.nn.functional.interpolate(
            x[:, :, c:],
            (time - w, x.shape[3]),
            mode=self.time_warp_mode,
            align_corners=True, )

        x[:, :, :w] = left
        x[:, :, w:] = right
        return x.reshape([*original_size])

    def mask_along_axis(self, x, dim):
        """Mask along time or frequency axis.
        Arguments
        ---------
        x : tensor
            Input tensor.
        dim : int
            Corresponding dimension to mask.
        """
        original_size = x.shape
        if x.dim() == 4:
            x = x.reshape([-1, x.shape[2], x.shape[3]])

        batch, time, fea = x.shape

        if dim == 1:
            D = time
            n_mask = self.n_time_mask
            width_range = self.time_mask_width
        else:
            D = fea
            n_mask = self.n_freq_mask
            width_range = self.freq_mask_width

        mask_len = paddle.randint(width_range[0], width_range[1],
                                  (batch, n_mask)).unsqueeze(2)

        mask_pos = paddle.randint(0, max(1, D - mask_len.max()),
                                  (batch, n_mask)).unsqueeze(2)

        # compute masks
        arange = paddle.arange(end=D).reshape([1, 1, -1])
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(axis=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        if self.replace_with_zero:
            val = 0.0
        else:
            val = x.mean()
        # same to x.masked_fill_(mask, val)
        y = paddle.full(x.shape, val, x.dtype)
        x = paddle.where(mask, y, x)
        return x.reshape([*original_size])


class TimeDomainSpecAugment(nn.Layer):
    """A time-domain approximation of the SpecAugment algorithm.
    This augmentation module implements three augmentations in
    the time-domain.
     1. Drop chunks of the audio (zero amplitude or white noise)
     2. Drop frequency bands (with band-drop filters)
     3. Speed peturbation (via resampling to slightly different rate)
    Arguments
    ---------
    perturb_prob : float from 0 to 1
        The probability that a batch will have speed perturbation applied.
    drop_freq_prob : float from 0 to 1
        The probability that a batch will have frequencies dropped.
    drop_chunk_prob : float from 0 to 1
        The probability that a batch will have chunks dropped.
    speeds : list of ints
        A set of different speeds to use to perturb each batch.
        See ``speechbrain.processing.speech_augmentation.SpeedPerturb``
    sample_rate : int
        Sampling rate of the input waveforms.
    drop_freq_count_low : int
        Lowest number of frequencies that could be dropped.
    drop_freq_count_high : int
        Highest number of frequencies that could be dropped.
    drop_chunk_count_low : int
        Lowest number of chunks that could be dropped.
    drop_chunk_count_high : int
        Highest number of chunks that could be dropped.
    drop_chunk_length_low : int
        Lowest length of chunks that could be dropped.
    drop_chunk_length_high : int
        Highest length of chunks that could be dropped.
    drop_chunk_noise_factor : float
        The noise factor used to scale the white noise inserted, relative to
        the average amplitude of the utterance. Default 0 (no noise inserted).
    Example
    -------
    >>> inputs = paddle.randn([10, 16000])
    >>> feature_maker = TimeDomainSpecAugment(speeds=[80])
    >>> feats = feature_maker(inputs, paddle.ones(10))
    >>> feats.shape
    paddle.shape([10, 12800])
    """

    def __init__(
            self,
            perturb_prob=1.0,
            drop_freq_prob=1.0,
            drop_chunk_prob=1.0,
            speeds=[95, 100, 105],
            sample_rate=16000,
            drop_freq_count_low=0,
            drop_freq_count_high=3,
            drop_chunk_count_low=0,
            drop_chunk_count_high=5,
            drop_chunk_length_low=1000,
            drop_chunk_length_high=2000,
            drop_chunk_noise_factor=0, ):
        super().__init__()
        self.speed_perturb = SpeedPerturb(
            perturb_prob=perturb_prob, orig_freq=sample_rate, speeds=speeds)
        self.drop_freq = DropFreq(
            drop_prob=drop_freq_prob,
            drop_count_low=drop_freq_count_low,
            drop_count_high=drop_freq_count_high, )
        self.drop_chunk = DropChunk(
            drop_prob=drop_chunk_prob,
            drop_count_low=drop_chunk_count_low,
            drop_count_high=drop_chunk_count_high,
            drop_length_low=drop_chunk_length_low,
            drop_length_high=drop_chunk_length_high,
            noise_factor=drop_chunk_noise_factor, )

    def forward(self, waveforms, lengths):
        """Returns the distorted waveforms.
        Arguments
        ---------
        waveforms : tensor
            The waveforms to distort
        """
        # Augmentation
        with paddle.no_grad():
            waveforms = self.speed_perturb(waveforms)
            waveforms = self.drop_freq(waveforms)
            waveforms = self.drop_chunk(waveforms, lengths)
        return waveforms
